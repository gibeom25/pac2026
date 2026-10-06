"""SO-101 seam-following RL 환경 — ee_rig 기반 (2026-10-03 전면 재작성).

설계 문서: docs/AI_추론계층_프레임워크.md 3.4절. Gymnasium 표준 API(reset/step) 단일 환경.

2026-10-03 재작성 이유: 이전 버전은 5관절 로봇팔 MJCF(so101_new_calib.xml) + 실제 그리퍼
액추에이터 + 직선 2점 랜덤 경로를 썼는데, 실제 데이터 수집(tools/record_mujoco.py, 2026-09-23
EE-only 전환 이후)은 이미 EE-only 리그(assets/so101/ee_rig.xml: mocap+weld, 관절 없음) +
바이너리 트리거(실제 그리퍼 없음, "선 밖에서 접촉하면 실패") + 6형태×5variant A4 용접선 씬으로
완전히 바뀌어 있었다 — RL이 학습하는 물리 세계와 실제 배포 물리 세계가 달라서 RL이 배운 행동이
전이되지 않는 문제였다. 이제 데이터 수집과 똑같은 물리(ee_rig.xml, 같은 MJCF)를 쓰고, 액션도
같은 관례(위치/회전은 EEF-delta를 직접 적분, 관절 IK 없음)를 따른다.

관측(observation.state=9, environment_state=5, 이미지 placeholder)과 BC teacher 연동은 이전과
동일하게 유지했다 — configs/so101_act_bc.py / so101_sac.py와 맞춤.

Ground-truth 경로: 텍스처 PNG는 전부 같은 A4 평면 지오메트리를 공유하므로(텍스처 파일명만
다름 — 물리에는 영향 없음), MuJoCo 모델은 하나만 로드하고(reset마다 다시 불러올 필요 없음),
각 에피소드마다 envs/seam_ground_truth.py로 (형태, variant) 30가지 중 하나를 무작위로 골라
ground-truth target_polyline만 바꾼다 (텍스처 자체는 렌더링하지 않으므로 — 이미지는 여전히
placeholder, 아래 "1차 버전 범위" 참고 — 바꿀 필요가 없다).

안전성(2026-10-03, 기범 요청): "선을 잘 따라가고, 선 아닌 곳에서는 액체(비드)를 끊을 수 있어야
한다." reward.coverage_reward의 off_target_penalty는 소프트 패널티(그래디언트 신호)일 뿐 실제
행동을 막지는 않는다 — 정책이 아직 그 패널티를 충분히 학습하기 전에는 선 밖에서도 계속 도포할
수 있다. 그래서 여기서는 2단계로 분리했다:
  - coverage_radius(0.008m) 이내: 정상 도포 (soft reward, 점점 배우는 영역)
  - off_seam_safety_dist(cfg, 기본 0.03m) 밖: 정책이 트리거를 눌러도 gripper_active를 강제로
    0으로 덮어쓴다(step()의 safety override) — 학습 여부와 무관하게 항상 보장되는 하드 안전장치.
    실제로 적용된 gripper_active(override 반영값)를 보상/비드 판정 둘 다에 넣는다.
  - 막대가 바닥/용지에 물리적으로 닿으면(record_mujoco.py와 동일 규약, "표면에 닿음=실패") 그
    즉시 terminated=True + 큰 음의 보상 — teleoperation에서 자동 폐기되는 것과 같은 조건을
    RL에서는 에피소드 종료로 반영한다.

1차 버전 범위(이전 버전에서 유지): 카메라는 아직 관측에 안 쓴다(RL 스텝마다 렌더링하면 비용이
커서 observation.images.wrist는 0 placeholder) — BC teacher도 같은 키를 받으므로 형태만
맞추면 된다. 실제 렌더 연동은 후속 작업.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

from lerobot.utils.rotation import Rotation

from ai_layer.configs.so101_sac import CONTINUOUS_ACTION_DIM, IMAGE_KEY
from ai_layer.envs.seam_ground_truth import N_VARIANTS, SCENE_NAMES, SeamGroundTruth
from ai_layer.kinematics import pose_to_state, pose_to_xyzrotvec
from ai_layer.rl.reward import WeightSchedule, _point_to_polyline, total_reward

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "so101"
# 물리는 모든 scene_a4*.xml이 동일(ee_rig.xml 포함, A4 평면 크기/위치도 동일) — 텍스처 파일명만
# 다르고 RL은 그 텍스처를 렌더링하지 않으므로 아무 파일이나 대표로 하나만 로드하면 된다.
MJCF_PATH = ASSETS_DIR / "scene_a4.xml"
PATH_NUM_POINTS = 20

# ee_rig.xml 바디/지오메트리 이름. tools/record_mujoco.py, tools/check_ee.py와 동일 관례를 쓰되,
# 조이스틱/뷰어 종속 없는 RL 환경에 조이스틱 하드웨어 모듈을 끌어오지 않도록 여기서 독립적으로
# 정의한다(동일 상수를 3곳에 중복 — 셋 다 ee_rig.xml 자체의 이름과 반드시 일치해야 함).
MOCAP_BODY_NAME = "mocap_target"
EE_BODY_NAME = "ee_body"
ROD_GEOM_NAME = "tool_rod"
FLOOR_GEOM_NAME = "floor"
ROD_HALF_LENGTH = 0.025  # ee_rig.xml tool_rod size[1] (5cm 막대의 절반)
MOCAP_HOME = np.array([0.25, 0.0, 0.15])
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
WORKSPACE_X = (0.05, 0.45)
WORKSPACE_Y = (-0.20, 0.20)
WORKSPACE_Z = (-0.02, 0.35)


def _rod_tip_world(data, rod_gid: int) -> np.ndarray:
    R = data.geom_xmat[rod_gid].reshape(3, 3)
    return data.geom_xpos[rod_gid] + R @ np.array([0.0, 0.0, -ROD_HALF_LENGTH])


def _rotmat_to_mujoco_quat(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _ee_pose_xyzrotvec(data, ee_bid: int, rod_gid: int) -> np.ndarray:
    """도구 끝(rod tip) 기준 pose — record_mujoco.py와 동일 기준(관측/학습 일관성 필수)."""
    T = np.eye(4)
    T[:3, :3] = data.xmat[ee_bid].reshape(3, 3)
    T[:3, 3] = _rod_tip_world(data, rod_gid)
    return pose_to_xyzrotvec(T)


def _contact_pos(data, gid_a: int, gid_b: int) -> np.ndarray | None:
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == {gid_a, gid_b}:
            return c.pos.copy()
    return None


@dataclass
class SO101SeamEnvCfg:
    episode_length_s: float = 12.0
    physics_dt: float = 1.0 / 300.0  # ee_rig.xml 안정성 검증된 값 (assets/so101/ee_rig.xml 참고)
    decimation: int = 10  # policy dt = 1/30s (BC/제어 dt와 동일)

    action_scale_pos: float = 0.01  # 최대 EEF 위치 delta (m/step)
    action_scale_rot: float = 0.05  # 최대 EEF 회전 delta (rad/step)
    target_speed_base: float = 0.005  # 보상 목표 진행량 (m/step). action_scale_pos보다 작아야 함.
    coverage_radius: float = 0.008  # reward.coverage_reward와 동일 — "덮었다"로 치는 반경
    off_seam_safety_dist: float = 0.03  # 이거보다 멀면 트리거를 눌러도 비드 강제 OFF (안전 컷오프)
    floor_contact_penalty: float = 5.0  # 바닥/용지 접촉 시 추가 음의 보상(막대 하나 스케일보다 훨씬 큼)

    # 2026-10-06: "회전은 데이터셋에 저장은 하되 RL은 회전을 덜 참고하게" — imitation_reward/
    # smoothness_reward 계산에서 회전 성분(drx,dry,drz)에 곱할 가중치. 1.0이면 위치와 동등(이전
    # 동작), 작을수록 RL이 "BC 회전을 똑같이 따라하는지"/"회전이 매끄러운지"에 덜 민감해진다 —
    # 위치(선 추종)를 우선하고 회전은 참고만 하는 쪽으로. bc_action_distance 로그는 영향 안 받음
    # (그건 가중치 없는 원본 거리를 그대로 보여줌 — 순수 모니터링용).
    rotation_reward_weight: float = 0.3

    device: str = "cpu"


class SO101SeamEnv(gym.Env):
    """단일(비병렬) MuJoCo 환경. lerobot SACPolicy가 기대하는 관측 dict(배치 1)를 반환한다."""

    def __init__(self, cfg: SO101SeamEnvCfg | None = None):
        self.cfg = cfg or SO101SeamEnvCfg()
        assert self.cfg.target_speed_base < self.cfg.action_scale_pos, "target_speed_base는 action_scale_pos보다 작아야 한다"
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg.physics_dt

        self.mocap_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODY_NAME)
        self.mocap_idx = self.model.body_mocapid[self.mocap_bid]
        self.ee_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        self.rod_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, ROD_GEOM_NAME)
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)

        self._action_scale = np.array([self.cfg.action_scale_pos] * 3 + [self.cfg.action_scale_rot] * 3)
        self._action_dim_weights = torch.tensor(
            [1.0] * 3 + [self.cfg.rotation_reward_weight] * 3
        )  # imitation/smoothness 보상용 — 위치 1.0, 회전은 낮춤(위 rotation_reward_weight 참고)
        self.seam_gt = SeamGroundTruth(num_points=PATH_NUM_POINTS)
        self._rng = np.random.default_rng()

        self.num_envs = 1  # train_rl.py와의 인터페이스 호환용 (배치 크기 1)
        self.max_episode_length = int(self.cfg.episode_length_s / (self.cfg.physics_dt * self.cfg.decimation))

        self._episode_step = 0
        self._target_pos = MOCAP_HOME.copy()
        self._R_cmd = np.eye(3)  # mocap에 명령하는 누적 회전 (teleoperation의 rotation_rate 적분과 동일 상태)
        self._target_polyline = np.zeros((PATH_NUM_POINTS, 3), dtype=np.float32)
        self._path_curvature = 0.0
        self._path_thickness = 1.0
        self._prev_progress = 0.0
        self._coverage_mask = torch.zeros(1, PATH_NUM_POINTS, dtype=torch.bool)
        self._prev_action = np.zeros(CONTINUOUS_ACTION_DIM)
        self._weight_schedule = WeightSchedule()
        self._bc = None  # (policy, preprocessor, postprocessor)
        self._total_steps = 0
        self._total_env_steps = 200_000  # set_total_env_steps 로 덮어씀

    # ---- 외부 주입 ----
    def set_bc_reference(self, policy, preprocessor=None, postprocessor=None) -> None:
        """3.4절 R_imitation 용 BC(ACT) teacher. None 이면 모방항 0."""
        self._bc = None if policy is None else (policy, preprocessor, postprocessor)

    def set_total_env_steps(self, total_env_steps: int) -> None:
        """보상 가중치 스케줄(0→1)의 분모. train_rl 이 num_steps 를 넘긴다."""
        self._total_env_steps = max(1, int(total_env_steps))

    # ---- 내부 유틸 ----
    def _tip_pose(self) -> np.ndarray:
        """도구 끝(rod tip) 실측 pose (6,) [xyz, rotvec]."""
        return _ee_pose_xyzrotvec(self.data, self.ee_bid, self.rod_gid)

    def _get_observations(self, tip_pos: np.ndarray | None = None) -> dict[str, torch.Tensor]:
        pose = self._tip_pose()
        ee_pos = pose[:3] if tip_pos is None else tip_pos

        pt_t = torch.from_numpy(ee_pos).float().unsqueeze(0)
        poly_t = torch.from_numpy(self._target_polyline).float().unsqueeze(0)
        _, _, seg_idx = _point_to_polyline(pt_t, poly_t)
        lookahead_idx = min(int(seg_idx.item()) + 2, PATH_NUM_POINTS - 1)
        rel = self._target_polyline[lookahead_idx] - ee_pos

        env_state = np.concatenate([rel, [self._path_curvature, self._path_thickness]]).astype(np.float32)
        state = pose_to_state(pose).astype(np.float32)  # (9,) = [xyz, rot6d]
        image_placeholder = np.zeros((3, 240, 320), dtype=np.float32)

        return {
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "observation.environment_state": torch.from_numpy(env_state).unsqueeze(0),
            IMAGE_KEY: torch.from_numpy(image_placeholder).unsqueeze(0),
        }

    def _bc_action_scaled(self, obs: dict) -> torch.Tensor | None:
        """BC teacher 청크의 첫 스텝(m, rad) -> env 액션 스케일 [-1,1] (1, 6)."""
        if self._bc is None:
            return None
        policy, pre, post = self._bc
        with torch.no_grad():
            batch = pre(dict(obs)) if pre is not None else dict(obs)
            chunk = policy.predict_action_chunk(batch)  # (1, chunk, 7)
            first = post(chunk[:, 0, :]) if post is not None else chunk[:, 0, :].cpu()
        delta = first[:, :CONTINUOUS_ACTION_DIM].numpy()
        return torch.from_numpy(np.clip(delta / self._action_scale, -1.0, 1.0)).float()

    def _compute_reward(
        self, continuous_action: np.ndarray, gripper_active: float, tip_pos: np.ndarray
    ) -> tuple[float, float | None]:
        """reward와 함께 "RL이 BC를 얼마나 잘 따라가는지" 점수(bc_action_distance)도 반환한다.

        2026-10-06(기범): imitation_reward는 이미 total_reward 안에서 가중합으로만 쓰이고
        있었는데("RL이 BC를 얼마나 잘 따라갈지" 자체는 스칼라 보상에 섞여서 사라짐), 그걸
        학습 중 추이로 보고 싶다는 요청 — action_rl/action_bc 거리(정규화 [-1,1] 공간, L2)를
        따로 계산해 step()의 info에 실어서 train_rl.py가 로그 파일에 쌓게 한다. BC teacher가
        없으면(--bc-checkpoint 없음) None.
        """
        progress_fraction = self._total_steps / float(self._total_env_steps)
        weights = self._weight_schedule.weights(progress_fraction)

        action_bc = self._bc_action_scaled(self._get_observations(tip_pos)) if self._bc is not None else None
        bc_action_distance = (
            float(torch.linalg.norm(torch.from_numpy(continuous_action).float() - action_bc[0]))
            if action_bc is not None
            else None
        )

        reward_t, new_progress_t, new_coverage_mask = total_reward(
            action_rl=torch.from_numpy(continuous_action).float().unsqueeze(0),
            action_tm1=torch.from_numpy(self._prev_action).float().unsqueeze(0),
            eef_pos=torch.from_numpy(tip_pos).float().unsqueeze(0),
            target_polyline=torch.from_numpy(self._target_polyline).float().unsqueeze(0),
            prev_progress=torch.tensor([self._prev_progress]).float(),
            curvature_at_progress=torch.tensor([self._path_curvature]).float(),
            thickness_at_progress=torch.tensor([self._path_thickness]).float(),
            weights=weights,
            coverage_mask=self._coverage_mask,
            gripper_active=torch.tensor([gripper_active]).float(),
            action_bc=action_bc,
            lambda_consistency=self._weight_schedule.lambda_consistency,
            base_speed=self.cfg.target_speed_base,
            coverage_radius=self.cfg.coverage_radius,
            dim_weights=self._action_dim_weights,
        )
        self._prev_progress = new_progress_t.item()
        self._coverage_mask = new_coverage_mask
        return reward_t.item(), bc_action_distance

    # ---- Gymnasium API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._target_pos = MOCAP_HOME.copy()
        self._R_cmd = np.eye(3)
        self.data.mocap_pos[self.mocap_idx] = self._target_pos
        self.data.mocap_quat[self.mocap_idx] = IDENTITY_QUAT
        mujoco.mj_forward(self.model, self.data)

        scene = SCENE_NAMES[int(self._rng.integers(0, len(SCENE_NAMES)))]
        variant = int(self._rng.integers(0, N_VARIANTS))
        self._target_polyline, self._path_curvature, self._path_thickness = self.seam_gt.load(scene, variant)

        self._prev_progress = 0.0
        self._prev_action = np.zeros(CONTINUOUS_ACTION_DIM)
        self._coverage_mask = torch.zeros(1, PATH_NUM_POINTS, dtype=torch.bool)
        self._episode_step = 0

        obs = self._get_observations()
        return obs, {"scene": scene, "variant": variant}

    def step(self, action: torch.Tensor):
        action_np = action.squeeze(0).cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        continuous = np.clip(action_np[:CONTINUOUS_ACTION_DIM], -1.0, 1.0)
        gripper_signal = action_np[CONTINUOUS_ACTION_DIM]

        delta = continuous * self._action_scale  # [dx,dy,dz,drx,dry,drz], BC action과 동일 단위/관례

        self._target_pos = self._target_pos + delta[:3]
        self._target_pos[0] = float(np.clip(self._target_pos[0], *WORKSPACE_X))
        self._target_pos[1] = float(np.clip(self._target_pos[1], *WORKSPACE_Y))
        self._target_pos[2] = float(np.clip(self._target_pos[2], *WORKSPACE_Z))
        self.data.mocap_pos[self.mocap_idx] = self._target_pos

        if np.any(delta[3:6]):
            # kinematics.apply_pose_delta와 동일 관례: 월드 기준 왼쪽 곱 누적
            self._R_cmd = Rotation.from_rotvec(delta[3:6]).as_matrix() @ self._R_cmd
        self.data.mocap_quat[self.mocap_idx] = _rotmat_to_mujoco_quat(self._R_cmd)

        for _ in range(self.cfg.decimation):
            mujoco.mj_step(self.model, self.data)

        tip = _rod_tip_world(self.data, self.rod_gid)

        # 안전 컷오프: 선에서 off_seam_safety_dist보다 멀면 정책이 트리거를 켜도 실제로는 비활성.
        # 모듈 docstring의 "안전성" 설계 참고 — reward의 off_target 소프트 패널티와는 별개로, 학습
        # 여부와 무관하게 항상 보장되는 하드 제약이다.
        dist_to_seam = float(np.linalg.norm(tip[None, :] - self._target_polyline, axis=1).min())
        requested_gripper = gripper_signal > 0.5
        gripper_active = requested_gripper and dist_to_seam <= self.cfg.off_seam_safety_dist

        floor_touched = _contact_pos(self.data, self.rod_gid, self.floor_gid) is not None

        reward, bc_action_distance = self._compute_reward(continuous, gripper_active=float(gripper_active), tip_pos=tip)
        if floor_touched:
            reward -= self.cfg.floor_contact_penalty
        obs = self._get_observations(tip_pos=tip)

        self._episode_step += 1
        self._total_steps += 1
        self._prev_action = continuous.copy()
        truncated = self._episode_step >= self.max_episode_length
        terminated = bool(floor_touched)  # record_mujoco.py와 동일 규약: 표면 접촉 = 즉시 실패

        info = {} if bc_action_distance is None else {"bc_action_distance": bc_action_distance}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass
