"""Piper 부스 시뮬레이션(envs/piper_sim.py) 위의 선 따라 그리기 RL 환경 (2026-10-09).

so101_seam_env.py(로봇 팔 없는 리그)와 같은 인터페이스(reset/step, set_bc_reference, set_total_env_steps,
num_envs)와 같은 보상(rl/reward.py total_reward)을 쓰지만, 세계는 실물 Piper 부스를 본뜬 것이다:
  - 액션: 플랜지 EEF-delta [-1,1] * action_scale (위치는 베이스 좌표, 회전은 월드 회전벡터 왼쪽 곱 —
    BC/실물 데이터와 같은 규약) + 트리거(이산 0/1). IK로 관절을 맞춘다(실물도 로컬 IK).
  - 관측: BC와 같은 규약 — observation.state = 플랜지 pose(9), observation.environment_state = 손목 이미지에서
    뽑은 seam CV 특징(5, perception/seam_features), observation.images.wrist = 실제 렌더한 손목 이미지.
    so101_seam_env.py는 정답 경로(lookahead 상대위치)를 관측으로 줬는데, 그건 실물에서 얻을 수 없는 값이라
    여기서는 쓰지 않는다. 정답 경로는 보상 계산에만 쓴다.
  - RL 정책 입력 이미지는 SAC_IMAGE_HW(120x160)로 줄인다 — 리플레이 버퍼(10만 전이 x 관측 2개)에 240x320
    float를 넣으면 수백 GB가 된다(버퍼는 uint8로 저장, rl/replay_buffer.py). BC teacher에는 원래 해상도
    (+고정캠이 필요한 teacher면 고정캠 렌더)를 따로 넣는다.
  - 에피소드마다 장면/variant, 부스(sample_booth), 시작 자세, 펜 기울기를 무작위로 바꾼다.
  - 바닥 접촉: 운동학 시뮬이라 실제 충돌은 없고, 명령한 펜 끝 높이가 0 아래로 내려가면 접촉으로 보고
    so101_seam_env.py와 같이 즉시 종료 + 감점한다(그 스텝의 펜은 MIN_TIP_Z에서 멈춘다).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ai_layer.bc_inference import fill_missing_images
from ai_layer.configs.so101_act_bc import OVERVIEW_IMAGE_KEY
from ai_layer.configs.so101_sac import CONTINUOUS_ACTION_DIM, IMAGE_KEY
from ai_layer.envs.piper_sim import (
    HOME_FLANGE_POS,
    PEN_TIP_FLANGE,
    PiperSim,
    sample_booth,
    seam_to_booth_xy,
    tilted_flange_R,
)
from ai_layer.envs.seam_ground_truth import N_VARIANTS, SCENE_NAMES, SeamGroundTruth
from ai_layer.kinematics import pose_to_state
from ai_layer.perception.seam_cv import SeamGrooveDetector
from ai_layer.perception.seam_features import seam_features_from_rgb
from ai_layer.rl.reward import WeightSchedule, total_reward

PATH_NUM_POINTS = 20
SAC_IMAGE_HW = (120, 160)
# 실물 Piper 설정(meta/piper_capture.jsonl config)의 작업공간 — 플랜지 위치 기준
WORKSPACE_MIN = np.array([0.14, -0.38, 0.04])
WORKSPACE_MAX = np.array([0.50, 0.38, 0.52])


def piper_sac_dataset_stats() -> dict[str, dict[str, torch.Tensor]]:
    """SAC 전처리 정규화 통계 — state는 실물 Piper 데이터셋 분포(평균/표준편차 근사)."""
    f = lambda *xs: torch.tensor(xs, dtype=torch.float32)  # noqa: E731
    return {
        IMAGE_KEY: {
            "mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
        },
        OBS_STATE: {
            "mean": f(0.32, 0.0, 0.26, -0.965, 0.0, -0.25, 0.0, 1.0, 0.0),
            "std": f(0.06, 0.08, 0.02, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05),
        },
        OBS_ENV_STATE: {
            "mean": f(0.0, 0.0, 0.0, 0.0, 0.5),
            "std": f(0.5, 0.5, 1.0, 1.0, 0.5),
        },
        ACTION: {"min": -torch.ones(CONTINUOUS_ACTION_DIM), "max": torch.ones(CONTINUOUS_ACTION_DIM)},
    }


@dataclass
class PiperSeamEnvCfg:
    episode_length_s: float = 20.0
    fps: int = 30
    action_scale_pos: float = 0.005  # 최대 플랜지 위치 delta [m/step] (=150mm/s; 실물 시연 90%가 67mm/s 이하)
    action_scale_rot: float = 0.02  # 최대 회전 delta [rad/step]
    target_speed_base: float = 0.0008  # 보상 목표 진행량 [m/step] (=24mm/s, 생성기 경로 속도 중앙값)
    coverage_radius: float = 0.008
    off_seam_safety_dist: float = 0.03
    floor_contact_penalty: float = 5.0
    dashed: str = "cut"
    coverage_spacing: float = 0.001
    gap_penalty: float = 0.02
    rotation_reward_weight: float = 0.3
    randomize_booth: bool = True
    tilt_deg: tuple[float, float] = (13.0, 16.0)


class PiperSeamEnv(gym.Env):
    """단일 환경. lerobot SACPolicy가 기대하는 관측 dict(배치 1)를 반환한다."""

    def __init__(self, cfg: PiperSeamEnvCfg | None = None):
        self.cfg = cfg or PiperSeamEnvCfg()
        assert self.cfg.target_speed_base < self.cfg.action_scale_pos
        self._action_scale = np.array([self.cfg.action_scale_pos] * 3 + [self.cfg.action_scale_rot] * 3)
        self._action_dim_weights = torch.tensor([1.0] * 3 + [self.cfg.rotation_reward_weight] * 3)
        self.seam_gt = SeamGroundTruth(num_points=PATH_NUM_POINTS)
        self.detector = SeamGrooveDetector()
        self._rng = np.random.default_rng()
        self.num_envs = 1
        self.dt = 1.0 / self.cfg.fps
        self.max_episode_length = int(self.cfg.episode_length_s * self.cfg.fps)
        self.sim: PiperSim | None = None
        self._weight_schedule = WeightSchedule()
        self._bc = None
        self._bc_uses_overview = False
        self._total_steps = 0
        self._total_env_steps = 200_000

    # ---- 외부 주입 ----
    def set_bc_reference(self, policy, preprocessor=None, postprocessor=None) -> None:
        self._bc = None if policy is None else (policy, preprocessor, postprocessor)
        self._bc_uses_overview = policy is not None and OVERVIEW_IMAGE_KEY in policy.config.image_features

    def set_total_env_steps(self, total_env_steps: int) -> None:
        self._total_env_steps = max(1, int(total_env_steps))

    # ---- 관측 ----
    def _observe(self) -> dict[str, torch.Tensor]:
        """SAC 관측을 만들고, BC teacher 입력(원래 해상도)도 self._bc_obs에 같이 만들어 둔다."""
        wrist = self.sim.render("wrist")
        state = torch.from_numpy(pose_to_state(self.sim.flange_pose_xyzrotvec()).astype(np.float32))[None]
        env_state = torch.from_numpy(seam_features_from_rgb(self.detector, wrist))[None]
        small = cv2.resize(wrist, SAC_IMAGE_HW[::-1], interpolation=cv2.INTER_AREA)
        obs = {
            OBS_STATE: state,
            OBS_ENV_STATE: env_state,
            IMAGE_KEY: torch.from_numpy(small).permute(2, 0, 1).float().div(255)[None],
        }
        if self._bc is not None:
            self._bc_obs = {
                OBS_STATE: state, OBS_ENV_STATE: env_state,
                IMAGE_KEY: torch.from_numpy(wrist).permute(2, 0, 1).float().div(255)[None],
            }
            if self._bc_uses_overview:
                self._bc_obs[OVERVIEW_IMAGE_KEY] = torch.from_numpy(self.sim.render("overview")).permute(2, 0, 1).float().div(255)[None]
        return obs

    def _bc_action_scaled(self) -> torch.Tensor | None:
        if self._bc is None:
            return None
        policy, pre, post = self._bc
        with torch.no_grad():
            obs = fill_missing_images(policy, dict(self._bc_obs))
            batch = pre(obs) if pre is not None else obs
            chunk = policy.predict_action_chunk(batch)
            first = post(chunk[:, 0, :]) if post is not None else chunk[:, 0, :].cpu()
        delta = first[:, :CONTINUOUS_ACTION_DIM].cpu().numpy()
        return torch.from_numpy(np.clip(delta / self._action_scale, -1.0, 1.0)).float()

    # ---- Gymnasium API ----
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self.sim is not None:
            self.sim.close()
        scene = SCENE_NAMES[int(self._rng.integers(0, len(SCENE_NAMES)))]
        variant = int(self._rng.integers(0, N_VARIANTS))
        booth = sample_booth(self._rng) if self.cfg.randomize_booth else None
        self.sim = PiperSim(scene, variant, booth)
        self._R = tilted_flange_R(float(self._rng.uniform(*self.cfg.tilt_deg)))
        pose = self.sim.move_flange(HOME_FLANGE_POS + self._rng.normal(0, [0.02, 0.01, 0.005]), self._R)
        self._pos = pose[:3].copy()
        self._R = Rotation.from_rotvec(pose[3:]).as_matrix()

        polyline, self._curv, self._thick = self.seam_gt.load(scene, variant)
        b = self.sim.booth
        self._polyline = np.concatenate([seam_to_booth_xy(polyline[:, :2], b), polyline[:, 2:]], axis=1).astype(np.float32)
        pts, paint, gap = self.seam_gt.coverage_points(scene, variant, self.cfg.coverage_spacing, self.cfg.dashed)
        pts = np.concatenate([seam_to_booth_xy(pts[:, :2], b), pts[:, 2:]], axis=1).astype(np.float32)
        self._coverage_points = torch.from_numpy(pts)[None]
        self._paint_mask = torch.from_numpy(paint)[None]
        self._gap_mask = torch.from_numpy(gap)[None]
        self._coverage_mask = torch.zeros_like(self._paint_mask)
        self._prev_progress = 0.0
        self._prev_action = np.zeros(CONTINUOUS_ACTION_DIM)
        self._episode_step = 0
        return self._observe(), {"scene": scene, "variant": variant}

    def step(self, action: torch.Tensor):
        a = action.squeeze(0).cpu().numpy() if action.dim() > 1 else action.cpu().numpy()
        continuous = np.clip(a[:CONTINUOUS_ACTION_DIM], -1.0, 1.0)
        delta = continuous * self._action_scale
        pos = np.clip(self._pos + delta[:3], WORKSPACE_MIN, WORKSPACE_MAX)
        R = Rotation.from_rotvec(delta[3:6]).as_matrix() @ self._R
        floor_touched = bool(pos[2] + (R @ PEN_TIP_FLANGE)[2] < 0.0)
        pose = self.sim.move_flange(pos, R)
        self._pos, self._R = pose[:3].copy(), Rotation.from_rotvec(pose[3:]).as_matrix()
        tip = self.sim.tip_pos()

        dist_to_seam = float(np.linalg.norm(tip[None] - self._polyline, axis=1).min())
        gripper_active = bool(a[CONTINUOUS_ACTION_DIM] > 0.5) and dist_to_seam <= self.cfg.off_seam_safety_dist
        self.sim.step_beads(gripper_active, self.dt)

        obs = self._observe()
        weights = self._weight_schedule.weights(self._total_steps / float(self._total_env_steps))
        action_bc = self._bc_action_scaled()
        reward_t, new_progress, self._coverage_mask = total_reward(
            action_rl=torch.from_numpy(continuous).float()[None],
            action_tm1=torch.from_numpy(self._prev_action).float()[None],
            eef_pos=torch.from_numpy(tip).float()[None],
            target_polyline=torch.from_numpy(self._polyline)[None],
            prev_progress=torch.tensor([self._prev_progress]).float(),
            curvature_at_progress=torch.tensor([self._curv]).float(),
            thickness_at_progress=torch.tensor([self._thick]).float(),
            weights=weights,
            coverage_mask=self._coverage_mask,
            gripper_active=torch.tensor([float(gripper_active)]),
            action_bc=action_bc,
            lambda_consistency=self._weight_schedule.lambda_consistency,
            base_speed=self.cfg.target_speed_base,
            coverage_radius=self.cfg.coverage_radius,
            dim_weights=self._action_dim_weights,
            coverage_points=self._coverage_points,
            paint_mask=self._paint_mask,
            gap_mask=self._gap_mask,
            gap_penalty=self.cfg.gap_penalty,
        )
        reward = reward_t.item() - (self.cfg.floor_contact_penalty if floor_touched else 0.0)
        self._prev_progress = new_progress.item()
        self._prev_action = continuous.copy()
        self._episode_step += 1
        self._total_steps += 1

        info = {
            "coverage": float(self._coverage_mask[self._paint_mask].float().mean()) if self._paint_mask.any() else 0.0,
            "dist_to_seam": dist_to_seam,
        }
        if action_bc is not None:
            info["bc_action_distance"] = float(torch.linalg.norm(torch.from_numpy(continuous).float() - action_bc[0]))
        truncated = self._episode_step >= self.max_episode_length
        return obs, reward, floor_touched, truncated, info

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None
