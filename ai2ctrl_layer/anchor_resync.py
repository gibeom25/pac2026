"""anchor_resync — Problem.md 2/4/8번 담당 (기존 AnchorMode.COMMIT_END의 정밀 보정 단계).

AI 쪽 choose_anchor()(이미 구현됨, ai_layer/control_bridge/snapshot_adapter.py)가 제어의
CommitTrajectory를 보고 새 chunk의 시작점을 대략 예측한다(COMMIT_END). 이 모듈은 그 chunk가
실제로 도착했을 때, control(PointMass) 쪽에서 자신의 **실제 현재 위치**를 기준으로 한 번 더
정밀하게 시작 인덱스를 고른다 — AI의 예측이 빗나가도 여기서 잡는다.

핵심은 reward.py의 _point_to_polyline()을 재사용해 "위치"가 아니라 "경로 위 진행률(호
길이)"로 비교하는 것 — 단조증가 제약(뒤로 가는 매칭 금지) + 진행 방향 비교로 7번(주기성/코너
모호성)과 8번(역방향 매칭으로 인한 진동 재발)을 동시에 막는다.
"""

from __future__ import annotations

import numpy as np
import torch

from ai_layer.control_bridge.protocol import ActionChunk
from ai_layer.rl.reward import _point_to_polyline


def resync_start_index(
    chunk: ActionChunk,
    actual_pos: np.ndarray,
    reference_polyline: np.ndarray,
    last_progress: float,
    direction_weight: float = 0.3,
) -> tuple[int, float, np.ndarray]:
    """chunk.steps(N,6)의 누적 위치 후보 N개 중, actual_pos에서 가장 그럴듯한 시작점을 고른다.

    Args:
        chunk: 디코딩된 ActionChunk (steps는 "actual_pos 기준" 상대 누적이 아니라, AI가 애초에
            가정한 임의 시작점 기준 누적이므로, 여기서는 "steps의 상대적 모양(진행 벡터)"만
            쓰고 절대 위치는 actual_pos를 새 기준점으로 삼아 다시 적분한다).
        actual_pos: PointMass의 실제 현재 위치(3,) — 8번 문제의 핵심(관측 시점이 아니라 지금).
        reference_polyline: (K,3) 참조 경로.
        last_progress: 직전까지 확정된 호 길이 진행률 — 이보다 작은 후보는 제외(단조증가).
        direction_weight: 위치거리와 방향유사도를 합칠 때 방향 항 가중치.

    Returns:
        (start_index, new_progress, anchor_pos) — anchor_pos는 선택된 시작점의 실제 위치
        (actual_pos 기준으로 재적분된 값, ChunkBuffer.load()의 anchor_pos 인자로 그대로 씀).
    """
    steps = np.asarray(chunk.steps[:, :3], dtype=float)  # (N,3) 위치 델타만
    n = steps.shape[0]
    # actual_pos를 새 기준점으로 삼아 각 스텝까지의 "실제" 누적 위치 후보를 만든다.
    cum = np.cumsum(steps, axis=0)  # (N,3), cum[i] = steps[0..i] 합
    candidates = actual_pos[None, :] + np.concatenate([np.zeros((1, 3)), cum[:-1]], axis=0)  # (N,3)

    pt_t = torch.from_numpy(candidates).float().unsqueeze(1)  # (N,1,3) — _point_to_polyline은 (N,3) 기대
    pt_t = pt_t.squeeze(1)
    poly_t = torch.from_numpy(reference_polyline).float().unsqueeze(0).expand(n, -1, -1)  # (N,K,3)
    perp_dist, progress, _seg_idx = _point_to_polyline(pt_t, poly_t)
    perp_dist = perp_dist.numpy()
    progress = progress.numpy()

    valid = progress >= last_progress  # 단조증가 제약
    if not valid.any():
        # 전부 뒤로 가는 후보뿐이면(비정상 상황) 그냥 첫 스텝부터 — 퇴행 방지 최후 수단.
        return 0, float(last_progress), actual_pos.copy()

    # 방향 유사도: 각 후보 지점에서의 "진행 방향"(다음 스텝 델타의 단위벡터)과
    # 참조 경로의 국소 접선 방향을 비교 — perp_dist + (1-cos유사도)*direction_weight로 점수화(작을수록 좋음).
    nxt = np.concatenate([steps[1:], steps[-1:]], axis=0)  # (N,3) 각 후보에서의 다음 진행 방향
    nxt_norm = nxt / np.clip(np.linalg.norm(nxt, axis=1, keepdims=True), 1e-9, None)
    seg = np.diff(reference_polyline, axis=0)
    seg_norm = seg / np.clip(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9, None)
    seg_idx_np = _seg_idx.numpy().clip(0, seg_norm.shape[0] - 1)
    tangent = seg_norm[seg_idx_np]
    cos_sim = (nxt_norm * tangent).sum(axis=1)
    direction_penalty = (1.0 - cos_sim) * direction_weight

    score = np.where(valid, perp_dist + direction_penalty, np.inf)
    best = int(np.argmin(score))
    return best, float(progress[best]), candidates[best]
