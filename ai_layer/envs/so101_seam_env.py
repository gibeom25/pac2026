"""SO-101 seam-following RL 환경 — MuJoCo 기반.

설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.

2026-09-22: IsaacLab 버전(git 히스토리 참고, `git log -- ai_layer/envs/so101_seam_env.py`)에서
MuJoCo로 교체. 이유: 이 컴퓨터 사양(8GB VRAM 노트북 GPU)에서 Isaac Sim/Kit이 계속 불안정했고
(CUDA P2P 검증 행, GPU 드라이버 이슈 등), SAC는 off-policy라 PPO만큼 대규모 병렬환경이 필요 없으며
BC가 이미 기초 정책을 제공하므로 RL은 국소 탐색 위주라 무거운 병렬 시뮬레이터가 필수가 아님.
MuJoCo는 GPU 없이도 가볍고 안정적으로 동작하며, 이 세션에서 직접 테스트/검증 가능하다는 장점도 있다.

Gymnasium 표준 API(`reset`/`step`)를 따르는 단일 환경 (필요하면 `gymnasium.vector`로 나중에 병렬화 가능,
지금은 SAC 특성상 불필요).

1차 버전 범위 (docs 7장 TODO와 동일한 단순화, so101_seam_env.py IsaacLab 버전과 동일한 설계):
  - 목표 경로(용접선)는 실제 카메라 인식이 아니라 에피소드마다 절차적으로 생성한 3D 폴리라인(ground
    truth)을 사용. environment_state(seam 특징)는 seam_cv.py를 거치지 않고 절차적 경로에서 직접 계산.
  - 카메라는 아직 연결 안 함 (observation.images.wrist는 0으로 채운 placeholder) — MuJoCo는
    `mujoco.Renderer`로 오프스크린 렌더링이 Isaac보다 훨씬 가볍게 가능하므로, 실제 카메라+seam_cv를
    루프에 넣는 후속 작업은 IsaacLab보다 이쪽이 오히려 더 쉬울 것으로 예상.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

from ai_layer.configs.so101_sac import CONTINUOUS_ACTION_DIM, IMAGE_KEY
from ai_layer.kinematics import apply_pose_delta, build_arm_kinematics
from ai_layer.rl.reward import WeightSchedule, _point_to_polyline, total_reward

MJCF_PATH = Path(__file__).resolve().parents[2] / "assets" / "so101" / "so101_new_calib.xml"
PATH_NUM_POINTS = 20
GRIPPER_JOINT_IDX = 5  # qpos/ctrl 인덱스 (0~4: 팔 5축, 5: 그리퍼)


@dataclass
class SO101SeamEnvCfg:
    episode_length_s: float = 8.0
    physics_dt: float = 1.0 / 60.0
    decimation: int = 2  # 정책 스텝당 물리 서브스텝 수

    path_x_range: tuple[float, float] = (0.15, 0.35)
    path_y_range: tuple[float, float] = (-0.15, 0.15)
    path_z: float = 0.05

    action_scale_pos: float = 0.02  # 최대 EEF 위치 delta (m/step)
    action_scale_rot: float = 0.05  # 최대 EEF 회전 delta (rad/step)

    device: str = "cpu"


class SO101SeamEnv(gym.Env):
    """단일(비병렬) MuJoCo 환경. lerobot SACPolicy가 기대하는 관측 dict를 반환한다."""

    def __init__(self, cfg: SO101SeamEnvCfg | None = None):
        self.cfg = cfg or SO101SeamEnvCfg()
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg.physics_dt

        self.arm_kin = build_arm_kinematics()

        self.num_envs = 1  # train_rl.py와의 인터페이스 호환용 (배치 크기 1)
        self.max_episode_length = int(self.cfg.episode_length_s / (self.cfg.physics_dt * self.cfg.decimation))

        self._episode_step = 0
        self._target_polyline = np.zeros((PATH_NUM_POINTS, 3))
        self._path_curvature = 0.0
        self._path_thickness = 1.0
        self._prev_progress = 0.0
        self._prev_action = np.zeros(CONTINUOUS_ACTION_DIM)
        self._weight_schedule = WeightSchedule()
        self._bc_reference = None
        self._total_steps = 0  # progress_fraction 계산용 (외부에서 갱신 가능)

    def set_bc_reference(self, bc_policy) -> None:
        self._bc_reference = bc_policy

    # ---- 내부 유틸 ----
    def _arm_joint_deg(self) -> np.ndarray:
        return np.rad2deg(self.data.qpos[:5])

    def _eef_pose(self) -> np.ndarray:
        return self.arm_kin.forward_kinematics(self._arm_joint_deg())

    def _generate_procedural_path(self) -> None:
        x0, y0 = np.random.uniform(*self.cfg.path_x_range), np.random.uniform(*self.cfg.path_y_range)
        x1, y1 = np.random.uniform(*self.cfg.path_x_range), np.random.uniform(*self.cfg.path_y_range)
        t = np.linspace(0, 1, PATH_NUM_POINTS).reshape(-1, 1)
        p0 = np.array([x0, y0, self.cfg.path_z])
        p1 = np.array([x1, y1, self.cfg.path_z])
        self._target_polyline = p0 + t * (p1 - p0)
        self._path_curvature = 0.0
        self._path_thickness = 1.0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        T = self._eef_pose()
        ee_pos = T[:3, 3]

        pt_t = torch.from_numpy(ee_pos).float().unsqueeze(0)
        poly_t = torch.from_numpy(self._target_polyline).float().unsqueeze(0)
        _, progress, seg_idx = _point_to_polyline(pt_t, poly_t)
        lookahead_idx = min(int(seg_idx.item()) + 2, PATH_NUM_POINTS - 1)
        rel = self._target_polyline[lookahead_idx] - ee_pos

        env_state = np.concatenate([rel, [self._path_curvature, self._path_thickness]]).astype(np.float32)
        rot_part = T[:3, :3][:, 0]  # 회전행렬 첫 열로 방향 근사 (docs 3dim 근사와 동일한 1차 단순화)
        state = np.concatenate([ee_pos, rot_part]).astype(np.float32)
        image_placeholder = np.zeros((3, 240, 320), dtype=np.float32)

        return {
            "observation.state": torch.from_numpy(state).unsqueeze(0),
            "observation.environment_state": torch.from_numpy(env_state).unsqueeze(0),
            IMAGE_KEY: torch.from_numpy(image_placeholder).unsqueeze(0),
        }

    def _compute_reward(self, continuous_action: np.ndarray) -> float:
        ee_pos = self._eef_pose()[:3, 3]
        progress_fraction = self._total_steps / 200_000.0  # train_rl.py의 num_steps 기본값과 대략 맞춤
        weights = self._weight_schedule.weights(progress_fraction)

        action_bc = None
        if self._bc_reference is not None:
            with torch.no_grad():
                bc_obs = self._get_observations()
                action_bc = self._bc_reference.select_action(bc_obs)[:, :CONTINUOUS_ACTION_DIM]

        reward_t, new_progress_t = total_reward(
            action_rl=torch.from_numpy(continuous_action).float().unsqueeze(0),
            action_tm1=torch.from_numpy(self._prev_action).float().unsqueeze(0),
            eef_pos=torch.from_numpy(ee_pos).float().unsqueeze(0),
            target_polyline=torch.from_numpy(self._target_polyline).float().unsqueeze(0),
            prev_progress=torch.tensor([self._prev_progress]).float(),
            curvature_at_progress=torch.tensor([self._path_curvature]).float(),
            thickness_at_progress=torch.tensor([self._path_thickness]).float(),
            weights=weights,
            action_bc=action_bc,
        )
        self._prev_progress = new_progress_t.item()
        self._prev_action = continuous_action
        return reward_t.item()

    # ---- Gymnasium API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._generate_procedural_path()
        self._prev_progress = 0.0
        self._prev_action = np.zeros(CONTINUOUS_ACTION_DIM)
        self._episode_step = 0

        obs = self._get_observations()
        return obs, {}

    def step(self, action: torch.Tensor):
        action_np = action.squeeze(0).cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        continuous = np.clip(action_np[:CONTINUOUS_ACTION_DIM], -1.0, 1.0)
        gripper_signal = action_np[CONTINUOUS_ACTION_DIM]

        scale = np.array([self.cfg.action_scale_pos] * 3 + [self.cfg.action_scale_rot] * 3)
        delta = continuous * scale

        current_joint_deg = self._arm_joint_deg()
        T_current = self.arm_kin.forward_kinematics(current_joint_deg)
        T_target = apply_pose_delta(T_current, delta[:3], delta[3:6])
        target_joint_deg = self.arm_kin.inverse_kinematics(current_joint_deg, T_target)

        self.data.ctrl[:5] = np.deg2rad(target_joint_deg)
        lo, hi = self.model.actuator_ctrlrange[GRIPPER_JOINT_IDX]
        self.data.ctrl[GRIPPER_JOINT_IDX] = hi if gripper_signal > 0.5 else lo

        for _ in range(self.cfg.decimation):
            mujoco.mj_step(self.model, self.data)

        reward = self._compute_reward(continuous)
        obs = self._get_observations()

        self._episode_step += 1
        self._total_steps += 1
        truncated = self._episode_step >= self.max_episode_length
        terminated = False

        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        pass
