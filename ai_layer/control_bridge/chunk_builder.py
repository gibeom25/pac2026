"""모델 출력 청크 (T, 7) -> 송지수 ActionChunk.

모델 출력 규약 (configs/so101_act_bc.py, kinematics.py):
    [:, 0:3]  dx dy dz        [m]   직전 스텝 대비 증분 (스텝 0 은 anchor 대비)
    [:, 3:6]  rx ry rz        [rad] 회전벡터, R_k = Exp(w) @ R_{k-1} (월드 왼쪽 곱)
    [:, 6]    gripper         LeRobot 녹화 단위 (0~100)
제어 규약 (protocol.ActionChunk): steps[N][6] 은 위 0:6 과 정의가 같다. eef[N] 은 펌프 0/1.

7번째 채널(gripper) -> eef 매핑은 팀 회의에서 정하기 전까지 스위치(EefMode)로 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ai_layer.control_bridge.protocol import MAX_STEPS, ActionChunk, AnchorMode

DEFAULT_DT_NS = 33_333_333  # 30 Hz = configs/so101_act_bc.DT_AI_SEC


class EefMode(str, Enum):
    OFF = "off"  # 항상 0 (펌프 OFF). 회의 전 기본값.
    ON = "on"  # 항상 1
    GRIPPER_THRESHOLD = "gripper_threshold"  # gripper 채널 >= 임계값 이면 1
    FROM_CHANNEL = "from_channel"  # 7번째 채널이 이미 0/1 이라고 보고 반올림


@dataclass
class ChunkLimits:
    """제어 계층 validator 가 거부하지 않도록 AI 쪽에서 미리 자르는 한계.

    출처: control/config/chunk_policy.json policy_id=1 (seam-welding) max_step_pos_m 0.008,
    max_step_rot_rad 0.06 + robot v_max 0.15 m/s, w_max 1.5 rad/s (example_so101.json).
    validator 는 스텝 크기와 스텝/dt 속도 둘 다 검사하므로 둘 중 작은 쪽이 실제 한계다.
      dt=33.3ms: pos = min(0.008, 0.15·0.0333=0.0050) = 0.0050 m, rot = min(0.06, 1.5·0.0333=0.050) = 0.050 rad
    실로봇 capabilities 가 더 보수적이면 그 값으로 갱신할 것.
    """

    max_step_pos_m: float = 0.008
    max_step_rot_rad: float = 0.06
    v_max: float = 0.15
    w_max: float = 1.5

    def effective(self, dt_ns: int) -> tuple[float, float]:
        dt_s = dt_ns * 1e-9
        return min(self.max_step_pos_m, self.v_max * dt_s), min(self.max_step_rot_rad, self.w_max * dt_s)


@dataclass
class BuildStats:
    chunks: int = 0
    pos_clipped_steps: int = 0
    rot_clipped_steps: int = 0
    max_pos_step_seen: float = 0.0
    max_rot_step_seen: float = 0.0
    log: list = field(default_factory=list)


def map_eef(gripper_channel: np.ndarray, mode: EefMode, threshold: float = 50.0) -> np.ndarray:
    g = np.asarray(gripper_channel, dtype=float).reshape(-1)
    if mode == EefMode.OFF:
        return np.zeros(len(g), dtype=np.uint8)
    if mode == EefMode.ON:
        return np.ones(len(g), dtype=np.uint8)
    if mode == EefMode.GRIPPER_THRESHOLD:
        return (g >= threshold).astype(np.uint8)
    if mode == EefMode.FROM_CHANNEL:
        return np.clip(np.rint(g), 0, 1).astype(np.uint8)
    raise ValueError(mode)


def clip_steps(steps: np.ndarray, limits: ChunkLimits, dt_ns: int, stats: BuildStats | None = None) -> np.ndarray:
    """스텝별 위치/회전 크기를 한계 이내로 줄인다 (방향 유지, 크기만 축소)."""
    steps = np.array(steps, dtype=float, copy=True)
    pos_lim, rot_lim = limits.effective(dt_ns)
    pn = np.linalg.norm(steps[:, :3], axis=1)
    rn = np.linalg.norm(steps[:, 3:6], axis=1)
    if stats is not None:
        stats.max_pos_step_seen = max(stats.max_pos_step_seen, float(pn.max(initial=0.0)))
        stats.max_rot_step_seen = max(stats.max_rot_step_seen, float(rn.max(initial=0.0)))
    over_p = pn > pos_lim
    over_r = rn > rot_lim
    if over_p.any():
        steps[over_p, :3] *= (pos_lim / pn[over_p])[:, None]
    if over_r.any():
        steps[over_r, 3:6] *= (rot_lim / rn[over_r])[:, None]
    if stats is not None:
        stats.pos_clipped_steps += int(over_p.sum())
        stats.rot_clipped_steps += int(over_r.sum())
    return steps


def build_action_chunk(
    model_chunk: np.ndarray,
    *,
    seq_id: int,
    snap_id: int,
    t_obs_ns: int,
    anchor_mode: AnchorMode = AnchorMode.OBS_POSE,
    policy_id: int = 1,
    dt_ns: int = DEFAULT_DT_NS,
    eef_mode: EefMode = EefMode.OFF,
    eef_threshold: float = 50.0,
    limits: ChunkLimits | None = None,
    clip: bool = True,
    n_steps: int | None = None,
    stats: BuildStats | None = None,
) -> ActionChunk:
    """(T, 7) numpy -> ActionChunk. n_steps 로 앞부분만 보낼 수 있다 (<= MAX_STEPS)."""
    m = np.asarray(model_chunk, dtype=float)
    if m.ndim != 2 or m.shape[1] != 7:
        raise ValueError(f"model_chunk shape {m.shape} != (T, 7)")
    if n_steps is not None:
        m = m[:n_steps]
    if m.shape[0] > MAX_STEPS:
        m = m[:MAX_STEPS]
    if not np.all(np.isfinite(m)):
        raise ValueError("model_chunk has NaN/Inf")
    steps = m[:, :6]
    if clip:
        steps = clip_steps(steps, limits or ChunkLimits(), dt_ns, stats)
    eef = map_eef(m[:, 6], eef_mode, eef_threshold)
    if stats is not None:
        stats.chunks += 1
    return ActionChunk(
        seq_id=int(seq_id),
        snap_id=int(snap_id),
        t_obs_ns=int(t_obs_ns),
        anchor_mode=anchor_mode,
        policy_id=int(policy_id),
        dt_ns=int(dt_ns),
        steps=np.ascontiguousarray(steps, dtype=float),
        eef=eef,
    )
