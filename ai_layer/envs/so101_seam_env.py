"""SO-101 seam-following RL 환경 (IsaacLab DirectRLEnv).

설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.

⚠️ 이 파일은 IsaacLab이 필요해 이번 세션(샌드박스)에서 직접 실행 검증을 못했다 (IsaacLab 자체는
smoke_test_so101.py로 사용자 터미널에서 검증 완료 — ai_layer_welding_trajectory_design.md 메모리
참고). 최초 실행 시 API 세부사항(관측 스페이스 타입, IK 컨트롤러 연동 등)에서 디버깅이 필요할 수 있다.

1차 버전 범위 (명시적 단순화, docs 7장에 TODO로 반영):
  - 목표 경로(용접선)는 실제 카메라 렌더링/seam_cv 인식이 아니라 **에피소드마다 절차적으로 생성한
    3D 폴리라인**을 ground truth로 사용한다. 즉 environment_state(seam 특징)는 seam_cv.py를 거치지
    않고 절차적 경로에서 직접 계산한다. 실제 비전 파이프라인(카메라+seam_cv)을 루프에 넣는 것은
    후속 작업 — 지금은 "경로를 따라가는 방법(RL)"만 학습하고, 인식(perception)은 배포 시 BC/RL이
    아니라 3.1절 CV 모듈이 별도로 담당한다는 설계와 일치시키기 위함이기도 하다.
  - 카메라는 아직 붙이지 않았다 (SO-ARM101-USD.usd에 카메라가 포함돼 있지만 정확한 마운트 프림
    경로/오프셋을 확인 못해 연결 보류). observation.images.wrist는 지금은 0으로 채운 placeholder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets" / "so101_isaac"))
from so101_cfg import SO101_CFG  # noqa: E402

from ai_layer.configs.so101_sac import CONTINUOUS_ACTION_DIM, IMAGE_KEY, SEAM_FEATURE_DIM, STATE_DIM  # noqa: E402
from ai_layer.rl.reward import WeightSchedule, total_reward  # noqa: E402

EE_BODY_NAME = "gripper"  # so101_cfg.SO101_CFG body 목록: base/shoulder/upper_arm/lower_arm/wrist/gripper/jaw
PATH_NUM_POINTS = 20


@configclass
class SO101SeamEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    robot_cfg: ArticulationCfg = SO101_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # 7 = 연속 EEF-delta(6) + 이산 gripper(1, SAC의 num_discrete_actions로 별도 처리되지만
    # env에는 하나의 텐서로 들어옴 — configs/so101_sac.py DISCRETE_DIMENSION_INDEX=-1 참고)
    action_space = CONTINUOUS_ACTION_DIM + 1
    # 참고용 (실제로는 train_rl.py가 dict 관측을 직접 다룸, gym space 타입 강제 안 함)
    observation_space = STATE_DIM + SEAM_FEATURE_DIM
    state_space = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)

    # 절차적 경로 생성 범위 (로봇 베이스 기준, m) — 1차 추정치, 실측 작업공간에 맞춰 튜닝 필요
    path_x_range = (0.15, 0.35)
    path_y_range = (-0.15, 0.15)
    path_z = 0.05

    action_scale_pos = 0.02  # 최대 EEF 위치 delta (m/step)
    action_scale_rot = 0.05  # 최대 EEF 회전 delta (rad/step)


class SO101SeamEnv(DirectRLEnv):
    cfg: SO101SeamEnvCfg

    def __init__(self, cfg: SO101SeamEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=[EE_BODY_NAME])
        self._robot_entity_cfg.resolve(self.scene)
        self._ee_jacobi_idx = (
            self._robot_entity_cfg.body_ids[0] - 1
            if self.robot.is_fixed_base
            else self._robot_entity_cfg.body_ids[0]
        )

        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self._ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        self._target_polyline = torch.zeros(self.num_envs, PATH_NUM_POINTS, 3, device=self.device)
        self._path_curvature = torch.zeros(self.num_envs, device=self.device)
        self._path_thickness = torch.ones(self.num_envs, device=self.device)
        self._prev_progress = torch.zeros(self.num_envs, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, CONTINUOUS_ACTION_DIM, device=self.device)
        self._weight_schedule = WeightSchedule()
        self._bc_reference = None  # train_rl.py가 set_bc_reference()로 주입 (R_imitation용, 선택)

    def set_bc_reference(self, bc_policy) -> None:
        """3.4절 R_imitation용 BC(ACT) teacher 주입. None이면 모방항 0으로 학습."""
        self._bc_reference = bc_policy

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """base(root) frame 기준 EE pose (pos, quat)."""
        ee_pose_w = self.robot.data.body_state_w[:, self._robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return ee_pos_b, ee_quat_b

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        continuous = actions[:, :CONTINUOUS_ACTION_DIM].clamp(-1.0, 1.0)
        # 그리퍼/펌프 이산 신호(actions[:, -1])는 1차 버전에서 물리 연동 없음 — TODO
        self._continuous_action = continuous
        scale = torch.tensor(
            [self.cfg.action_scale_pos] * 3 + [self.cfg.action_scale_rot] * 3, device=self.device
        )
        self._ee_delta_cmd = continuous * scale

    def _apply_action(self) -> None:
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        self._ik_controller.set_command(self._ee_delta_cmd, ee_pos_b, ee_quat_b)

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._robot_entity_cfg.joint_ids
        ]
        joint_pos = self.robot.data.joint_pos[:, self._robot_entity_cfg.joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._robot_entity_cfg.joint_ids)

    def _get_observations(self) -> dict:
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        state = torch.cat([ee_pos_b, ee_quat_b[:, 1:4]], dim=-1)  # (N,6): pos(3) + quat벡터부(3), 1차 근사

        # ground-truth 경로 기반 seam 특징 (실제 CV 파이프라인 연결 전까지의 대체값, 모듈 docstring 참고)
        from ai_layer.rl.reward import _point_to_polyline

        perp_dist, progress, seg_idx = _point_to_polyline(ee_pos_b, self._target_polyline)
        lookahead_idx = (seg_idx + 2).clamp(max=PATH_NUM_POINTS - 1)
        lookahead_pt = self._target_polyline.gather(
            1, lookahead_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        rel = lookahead_pt - ee_pos_b
        env_state = torch.cat(
            [
                rel,
                self._path_curvature.unsqueeze(-1),
                self._path_thickness.unsqueeze(-1),
            ],
            dim=-1,
        )  # (N, 5) — perception/seam_cv.py 출력 차원과 동일하게 맞춤

        image_placeholder = torch.zeros(self.num_envs, 3, 240, 320, device=self.device)  # TODO: 실제 카메라 연결

        return {
            "observation.state": state,
            "observation.environment_state": env_state,
            IMAGE_KEY: image_placeholder,
        }

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_b, _ = self._get_ee_pose_b()
        progress_fraction = float(self.common_step_counter) / max(1, self.max_episode_length * 100)
        weights = self._weight_schedule.weights(progress_fraction)

        action_bc = None
        if self._bc_reference is not None:
            with torch.no_grad():
                bc_obs = self._get_observations()
                action_bc = self._bc_reference.select_action(bc_obs)[:, :CONTINUOUS_ACTION_DIM]

        reward, new_progress = total_reward(
            action_rl=self._continuous_action,
            action_tm1=self._prev_action,
            eef_pos=ee_pos_b,
            target_polyline=self._target_polyline,
            prev_progress=self._prev_progress,
            curvature_at_progress=self._path_curvature,
            thickness_at_progress=self._path_thickness,
            weights=weights,
            action_bc=action_bc,
        )
        self._prev_progress = new_progress
        self._prev_action = self._continuous_action
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._generate_procedural_path(env_ids)
        self._prev_progress[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0

    def _generate_procedural_path(self, env_ids: torch.Tensor) -> None:
        """에피소드별 랜덤 스무스 경로 생성 (직선 보간, 1차 버전 — 곡선/지그재그는 TODO).

        docs 3.2 "궤적 길이/경로 형태/끊김 빈도/선 굵기별 속도"까지 반영하려면 곡률 있는 곡선과
        선 굵기 랜덤화가 필요 — 지금은 랜덤 시작/끝점 직선 + 일정 곡률/굵기 상수로 단순화.
        """
        n = len(env_ids)
        x0 = sample_uniform(self.cfg.path_x_range[0], self.cfg.path_x_range[1], (n,), self.device)
        y0 = sample_uniform(self.cfg.path_y_range[0], self.cfg.path_y_range[1], (n,), self.device)
        x1 = sample_uniform(self.cfg.path_x_range[0], self.cfg.path_x_range[1], (n,), self.device)
        y1 = sample_uniform(self.cfg.path_y_range[0], self.cfg.path_y_range[1], (n,), self.device)

        t = torch.linspace(0, 1, PATH_NUM_POINTS, device=self.device).view(1, -1, 1)
        p0 = torch.stack([x0, y0, torch.full_like(x0, self.cfg.path_z)], dim=-1).unsqueeze(1)
        p1 = torch.stack([x1, y1, torch.full_like(x1, self.cfg.path_z)], dim=-1).unsqueeze(1)
        self._target_polyline[env_ids] = p0 + t * (p1 - p0)
        self._path_curvature[env_ids] = 0.0
        self._path_thickness[env_ids] = 1.0
