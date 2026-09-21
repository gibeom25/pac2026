"""SO-101 seam-following RL 환경 (IsaacLab DirectRLEnv).

설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.

⚠️ IsaacLab이 필요해 Claude/Grok 세션(샌드박스)에서는 실행 검증을 못 한다. 사용자 터미널에서
`./isaaclab.sh -p ai_layer/train_rl.py` 로 최초 실행 시 API 세부에서 디버깅이 필요할 수 있다.

BC(configs/so101_act_bc.py)와 맞춘 것 (2026-09-21):
  - observation.state = 9 (xyz + rot6d, base frame). 예전 quat 벡터부(3)는 BC의 표현과 달랐다.
  - 이미지 키 observation.images.wrist (BC와 동일).
  - BC teacher는 (policy, preprocessor, postprocessor)로 주입하고, 출력(m/rad)을 env 액션 스케일
    ([-1,1])로 나눠서 비교한다. 예전엔 단위가 달라 R_imitation이 무의미했다.
  - IK는 팔 관절 5개만 푼다 (예전 ".*"는 턱 Jaw까지 움직여 EE를 맞추려 했다).
  - 보상 목표속도(target_speed_base)는 action_scale_pos보다 작게 둔다.
  - 보상 가중치 스케줄은 train_rl이 set_total_env_steps()로 알려준 전체 길이에 비례한다.

1차 버전 범위 (명시적 단순화):
  - 목표 경로(용접선)는 에피소드마다 절차적으로 생성한 3D 폴리라인. seam_cv를 거치지 않고
    절차적 경로에서 직접 seam 특징을 계산한다. 실제 카메라+seam_cv 연결은 후속 작업.
  - 카메라 미연결. observation.images.wrist는 0 placeholder.
  - Isaac USD는 아직 기본 SO-101 (assets/so101_isaac). 실로봇(PAC_Supermoon, tcp_link) 끝단과
    다르다. EE 바디는 기본 SO-101의 "gripper". USD를 UMI+D405로 바꾸는 것은 후속 작업.
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
from isaaclab.utils.math import matrix_from_quat, sample_uniform, subtract_frame_transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets" / "so101_isaac"))
from so101_cfg import SO101_CFG  # noqa: E402

from ai_layer.configs.so101_sac import CONTINUOUS_ACTION_DIM, IMAGE_KEY, SEAM_FEATURE_DIM, STATE_DIM  # noqa: E402
from ai_layer.rl.reward import WeightSchedule, _point_to_polyline, total_reward  # noqa: E402

EE_BODY_NAME = "gripper"  # so101_cfg.SO101_CFG body 목록: base/shoulder/upper_arm/lower_arm/wrist/gripper/jaw
# IK가 움직일 관절: 팔 5개만. Jaw(그리퍼)는 제외.
ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
PATH_NUM_POINTS = 20


@configclass
class SO101SeamEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 8.0

    # 60 Hz 물리 × decimation 2 = 30 Hz 정책 스텝 (= 송지수 dt 33.3 ms)
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)

    robot_cfg: ArticulationCfg = SO101_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # 7 = 연속 EEF-delta(6) + 이산 gripper(1). SAC은 num_discrete_actions로 마지막 차원을 따로 다룬다.
    action_space = CONTINUOUS_ACTION_DIM + 1
    observation_space = STATE_DIM + SEAM_FEATURE_DIM  # 참고용 (train_rl은 dict 관측을 직접 다룸)
    state_space = 0

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=True)

    # 절차적 경로 생성 범위 (로봇 베이스 기준, m) — 1차 추정치, 실측 작업공간에 맞춰 튜닝 필요
    path_x_range = (0.15, 0.35)
    path_y_range = (-0.15, 0.15)
    path_z = 0.05

    action_scale_pos = 0.02  # 최대 EEF 위치 delta (m/step)
    action_scale_rot = 0.05  # 최대 EEF 회전 delta (rad/step)
    target_speed_base = 0.01  # 보상 목표 진행량 (m/step). action_scale_pos 보다 작아야 함.


class SO101SeamEnv(DirectRLEnv):
    cfg: SO101SeamEnvCfg

    def __init__(self, cfg: SO101SeamEnvCfg, render_mode: str | None = None, **kwargs):
        assert cfg.target_speed_base < cfg.action_scale_pos, "target_speed_base는 action_scale_pos보다 작아야 한다"
        super().__init__(cfg, render_mode, **kwargs)

        self._robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=[EE_BODY_NAME])
        self._robot_entity_cfg.resolve(self.scene)
        self._ee_jacobi_idx = (
            self._robot_entity_cfg.body_ids[0] - 1
            if self.robot.is_fixed_base
            else self._robot_entity_cfg.body_ids[0]
        )

        ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")
        self._ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        self._action_scale = torch.tensor(
            [self.cfg.action_scale_pos] * 3 + [self.cfg.action_scale_rot] * 3, device=self.device
        )
        self._target_polyline = torch.zeros(self.num_envs, PATH_NUM_POINTS, 3, device=self.device)
        self._path_curvature = torch.zeros(self.num_envs, device=self.device)
        self._path_thickness = torch.ones(self.num_envs, device=self.device)
        self._prev_progress = torch.zeros(self.num_envs, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, CONTINUOUS_ACTION_DIM, device=self.device)
        self._continuous_action = torch.zeros(self.num_envs, CONTINUOUS_ACTION_DIM, device=self.device)
        self._ee_delta_cmd = torch.zeros(self.num_envs, CONTINUOUS_ACTION_DIM, device=self.device)
        self._weight_schedule = WeightSchedule()
        self._total_env_steps = max(1, int(self.max_episode_length) * 100)  # set_total_env_steps로 덮어씀
        self._bc = None  # (policy, preprocessor, postprocessor) — set_bc_reference()로 주입

    # ------------------------------------------------------------------ 외부 주입
    def set_bc_reference(self, policy, preprocessor, postprocessor) -> None:
        """3.4절 R_imitation용 BC(ACT) teacher. None이면 모방항 0으로 학습."""
        self._bc = None if policy is None else (policy, preprocessor, postprocessor)

    def set_total_env_steps(self, total_env_steps: int) -> None:
        """보상 가중치 스케줄(0→1)의 분모. train_rl이 num_steps // num_envs 를 넘긴다."""
        self._total_env_steps = max(1, int(total_env_steps))

    # ------------------------------------------------------------------ 씬
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.scene.clone_environments(copy_from_source=False)

        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """base(root) frame 기준 EE pose (pos, quat wxyz)."""
        ee_pose_w = self.robot.data.body_state_w[:, self._robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        return ee_pos_b, ee_quat_b

    # ------------------------------------------------------------------ 액션
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        continuous = actions[:, :CONTINUOUS_ACTION_DIM].clamp(-1.0, 1.0)
        # 그리퍼/펌프 이산 신호(actions[:, -1])는 1차 버전에서 물리 연동 없음 — TODO
        self._continuous_action = continuous
        self._ee_delta_cmd = continuous * self._action_scale

    def _apply_action(self) -> None:
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        self._ik_controller.set_command(self._ee_delta_cmd, ee_pos_b, ee_quat_b)

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._robot_entity_cfg.joint_ids
        ]
        joint_pos = self.robot.data.joint_pos[:, self._robot_entity_cfg.joint_ids]
        joint_pos_des = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self._robot_entity_cfg.joint_ids)

    # ------------------------------------------------------------------ 관측
    def _get_observations(self) -> dict:
        ee_pos_b, ee_quat_b = self._get_ee_pose_b()
        R = matrix_from_quat(ee_quat_b)  # (N, 3, 3)
        rot6d = torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)  # 회전행렬 앞 두 열 (kinematics.pose_to_state와 동일)
        state = torch.cat([ee_pos_b, rot6d], dim=-1)  # (N, 9)

        # ground-truth 경로 기반 seam 특징 (실제 CV 파이프라인 연결 전까지의 대체값)
        _, _, seg_idx = _point_to_polyline(ee_pos_b, self._target_polyline)
        lookahead_idx = (seg_idx + 2).clamp(max=PATH_NUM_POINTS - 1)
        lookahead_pt = self._target_polyline.gather(
            1, lookahead_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        rel = lookahead_pt - ee_pos_b
        env_state = torch.cat(
            [rel, self._path_curvature.unsqueeze(-1), self._path_thickness.unsqueeze(-1)], dim=-1
        )  # (N, 5) — perception/seam_cv.py 출력 차원과 동일하게 맞춤

        image_placeholder = torch.zeros(self.num_envs, 3, 240, 320, device=self.device)  # TODO: 실제 카메라 연결

        return {
            "observation.state": state,
            "observation.environment_state": env_state,
            IMAGE_KEY: image_placeholder,
        }

    # ------------------------------------------------------------------ 보상
    def _bc_action_scaled(self, obs: dict) -> torch.Tensor | None:
        """BC teacher 청크의 첫 스텝(m, rad) -> env 액션 스케일 [-1,1]로 변환."""
        if self._bc is None:
            return None
        policy, pre, post = self._bc
        with torch.no_grad():
            batch = pre(dict(obs))
            chunk = policy.predict_action_chunk(batch)  # (N, chunk, 7) 정규화 공간
            first = post(chunk[:, 0, :])  # (N, 7) 실제 단위, CPU
        delta = first[:, :CONTINUOUS_ACTION_DIM].to(self.device)
        return (delta / self._action_scale).clamp(-1.0, 1.0)

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_b, _ = self._get_ee_pose_b()
        progress_fraction = float(self.common_step_counter) / float(self._total_env_steps)
        weights = self._weight_schedule.weights(progress_fraction)

        action_bc = self._bc_action_scaled(self._get_observations()) if self._bc is not None else None

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
            lambda_consistency=self._weight_schedule.lambda_consistency,
            base_speed=self.cfg.target_speed_base,
        )
        self._prev_progress = new_progress
        self._prev_action = self._continuous_action.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    # ------------------------------------------------------------------ 리셋
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
        """에피소드별 랜덤 스무스 경로 생성 (직선 보간, 1차 버전 — 곡선/지그재그는 TODO)."""
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
