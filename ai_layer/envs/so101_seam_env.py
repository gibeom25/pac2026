"""SO-101 seam-following RL 환경 — MuJoCo 기반 (기범 선배님 2026-09-22 전환) + 2026-09-22 병합 정합.

설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.
IsaacLab → MuJoCo 전환 이유는 README "왜 IsaacLab 대신 MuJoCo인가" 참고. Gymnasium 표준 API(reset/step) 단일 환경.

BC(configs/so101_act_bc.py)와 맞춘 것 (병합 시 반영):
  - observation.state = 9 (xyz + rot6d, kinematics.pose_to_state 와 동일 정의). BC teacher 가 RL 관측을 그대로 읽는다.
  - 이미지 키 observation.images.wrist (configs/so101_sac.IMAGE_KEY = BC 와 동일).
  - BC teacher 는 (policy, preprocessor, postprocessor) 로 주입 (bc_inference.load_bc_checkpoint). 출력(m/rad)을
    env 액션 스케일([-1,1])로 나눠 R_imitation 에 넣는다 (단위가 달랐던 문제 수정).
  - 보상 가중치 스케줄 분모는 set_total_env_steps() 로 train_rl 이 알려준다.
  - 보상 목표속도(target_speed_base 0.01) < action_scale_pos(0.02).

FK/IK 는 실로봇 URDF(assets/pac_supermoon, tcp_link) 를 쓰고, MuJoCo(assets/so101/so101_new_calib.xml, 기본 SO-101)는
관절 동역학만 담당한다. 관측 EE pose 는 MuJoCo 기하가 아니라 qpos → 우리 FK 로 계산하므로 학습·추론과 일관된다.
(MJCF 의 시각/충돌 형상이 UMI+D405 와 다른 것은 후속 작업.)

1차 버전 범위: 목표 경로는 절차적 3D 폴리라인(ground truth), seam 특징도 거기서 직접 계산. 카메라 미연결(0 placeholder).
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
GRIPPER_JOINT_IDX = 5  # qpos/ctrl 인덱스 (0~4: 팔 5축 = kinematics.JOINT_NAMES[:5] 순서, 5: 그리퍼)


@dataclass
class SO101SeamEnvCfg:
    episode_length_s: float = 8.0
    # 물리 간격 1/300 s × 서브스텝 10 = 정책 스텝 1/30 s (= 송지수 dt 33.3 ms).
    # 주의: 1/60 s 로 두면 sts3215 위치 서보(kp≈998)가 수치적으로 불안정해 팔이 스스로 흔들리며 무너진다
    # (병합 검증에서 발견: ctrl 고정 상태로 1초 뒤 |qvel| 1.6 rad/s). MJCF 기본 0.002 s 나 그 근처를 쓸 것.
    physics_dt: float = 1.0 / 300.0
    decimation: int = 10

    path_x_range: tuple[float, float] = (0.15, 0.35)
    path_y_range: tuple[float, float] = (-0.15, 0.15)
    path_z: float = 0.05

    action_scale_pos: float = 0.02  # 최대 EEF 위치 delta (m/step)
    action_scale_rot: float = 0.05  # 최대 EEF 회전 delta (rad/step)
    target_speed_base: float = 0.01  # 보상 목표 진행량 (m/step). action_scale_pos 보다 작아야 함.
    ik_snap_m: float = 0.02  # IK 잔차가 이보다 크면(도달 불가) 명령 pose 를 실제 도달 pose 로 되돌림
    ik_iters: int = 5  # lerobot IK 는 호출당 1회 풀이 → 수렴할 때까지 반복 (큰 스텝에서 해가 튀는 것 방지)
    ik_orientation_weight: float = 0.01  # 위치 우선, 자세는 soft (민제씨 5D IK 와 같은 취지: yaw 자유).
    # 검증(2026-09-22): 도달 가능한 목표(작은 회전, 이동+회전)는 5회 반복으로 오차 0. 위치 고정 + pitch 0.5 rad 같은
    # 목표는 손목 한계(95°)에 닿아 물리적으로 불가 → IK 가 위치를 지키고 회전을 78% 만 채움 (정상 동작).
    # 시작 자세 (deg, JOINT_NAMES[:5]). qpos=0(완전히 뻗은 팔)에서 시작하면 자세 명령 몇 번에 관절 한계로 밀려 접힌다.
    # 손끝(tcp_link)이 경로 영역 중심 (0.25, 0, 0.10) 위에 오고 한계 여유 ≈ 24° 인 해 (IK 로 구함).
    home_joint_deg: tuple[float, float, float, float, float] = (0.0, -59.2, 35.7, 71.4, 0.0)

    device: str = "cpu"


class SO101SeamEnv(gym.Env):
    """단일(비병렬) MuJoCo 환경. lerobot SACPolicy가 기대하는 관측 dict(배치 1)를 반환한다."""

    def __init__(self, cfg: SO101SeamEnvCfg | None = None):
        self.cfg = cfg or SO101SeamEnvCfg()
        assert self.cfg.target_speed_base < self.cfg.action_scale_pos, "target_speed_base는 action_scale_pos보다 작아야 한다"
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg.physics_dt

        self.arm_kin = build_arm_kinematics()
        self._action_scale = np.array([self.cfg.action_scale_pos] * 3 + [self.cfg.action_scale_rot] * 3)

        self.num_envs = 1  # train_rl.py와의 인터페이스 호환용 (배치 크기 1)
        self.max_episode_length = int(self.cfg.episode_length_s / (self.cfg.physics_dt * self.cfg.decimation))

        self._episode_step = 0
        # 증분은 "직전 명령 pose" 에 누적한다 (송지수 ActionChunk 규약과 동일: p_k = p_{k-1} + dp).
        # 측정 pose 에 누적하면 중력 처짐이 매 스텝 목표에 흡수되어 팔이 계속 흘러내린다 (병합 검증에서 발견).
        self._T_cmd = np.eye(4)
        self._q_cmd_deg = np.zeros(5)
        self._target_polyline = np.zeros((PATH_NUM_POINTS, 3))
        self._path_curvature = 0.0
        self._path_thickness = 1.0
        self._prev_progress = 0.0
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
    def _arm_joint_deg(self) -> np.ndarray:
        return np.rad2deg(self.data.qpos[:5]).astype(np.float64)

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
        _, _, seg_idx = _point_to_polyline(pt_t, poly_t)
        lookahead_idx = min(int(seg_idx.item()) + 2, PATH_NUM_POINTS - 1)
        rel = self._target_polyline[lookahead_idx] - ee_pos

        env_state = np.concatenate([rel, [self._path_curvature, self._path_thickness]]).astype(np.float32)
        R = T[:3, :3]
        state = np.concatenate([ee_pos, R[:, 0], R[:, 1]]).astype(np.float32)  # (9,) = kinematics.pose_to_state
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

    def _compute_reward(self, continuous_action: np.ndarray) -> float:
        ee_pos = self._eef_pose()[:3, 3]
        progress_fraction = self._total_steps / float(self._total_env_steps)
        weights = self._weight_schedule.weights(progress_fraction)

        action_bc = self._bc_action_scaled(self._get_observations()) if self._bc is not None else None

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
            lambda_consistency=self._weight_schedule.lambda_consistency,
            base_speed=self.cfg.target_speed_base,
        )
        self._prev_progress = new_progress_t.item()
        self._prev_action = continuous_action.copy()
        return reward_t.item()

    # ---- Gymnasium API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        home = np.deg2rad(np.asarray(self.cfg.home_joint_deg, dtype=np.float64))
        self.data.qpos[:5] = home
        self.data.ctrl[:5] = home
        mujoco.mj_forward(self.model, self.data)

        self._generate_procedural_path()
        self._prev_progress = 0.0
        self._prev_action = np.zeros(CONTINUOUS_ACTION_DIM)
        self._episode_step = 0
        self._q_cmd_deg = self._arm_joint_deg()
        self._T_cmd = self.arm_kin.forward_kinematics(self._q_cmd_deg)

        obs = self._get_observations()
        return obs, {}

    def step(self, action: torch.Tensor):
        action_np = action.squeeze(0).cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        continuous = np.clip(action_np[:CONTINUOUS_ACTION_DIM], -1.0, 1.0)
        gripper_signal = action_np[CONTINUOUS_ACTION_DIM]

        delta = continuous * self._action_scale

        T_target = apply_pose_delta(self._T_cmd, delta[:3], delta[3:6])
        q = self._q_cmd_deg
        for _ in range(self.cfg.ik_iters):  # 반복 풀이 (warm start)
            q = np.asarray(self.arm_kin.inverse_kinematics(q, T_target, orientation_weight=self.cfg.ik_orientation_weight), dtype=np.float64)[:5]
        lo, hi = np.rad2deg(self.model.actuator_ctrlrange[:5, 0]), np.rad2deg(self.model.actuator_ctrlrange[:5, 1])
        self._q_cmd_deg = np.clip(q, lo, hi)
        # 다음 스텝 기준은 수학적으로 누적한 명령 pose (IK 잔차가 스텝마다 쌓이지 않게).
        # 단, IK 가 못 따라간(도달 불가) 목표는 실제 도달 pose 로 되돌려 무한히 벌어지지 않게 한다.
        T_reached = self.arm_kin.forward_kinematics(self._q_cmd_deg)
        if np.linalg.norm(T_reached[:3, 3] - T_target[:3, 3]) > self.cfg.ik_snap_m:
            self._T_cmd = T_reached
        else:
            self._T_cmd = T_target

        self.data.ctrl[:5] = np.deg2rad(self._q_cmd_deg)
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
