"""RL(SAC) 보상 함수. 설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.

R_t = w1*R_imitation + w2*R_track + w3*R_smooth

프레임워크(IsaacLab env)와 독립적인 순수 텐서 함수로 작성 — 어떤 env에서든 재사용 가능.
모든 함수는 (num_envs, ...) 배치 텐서를 받아 (num_envs,) 보상을 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def imitation_reward(action_rl: torch.Tensor, action_bc: torch.Tensor | None) -> torch.Tensor:
    """R_imitation = -||a_RL - a_BC||^2. BC teacher가 없으면(1차 체크포인트 없음) 0 반환.

    action_bc가 None인 경우는 "아직 BC 체크포인트가 없어 순수 R_track+R_smooth로만 학습" 상황
    (train_rl.py에서 weight_schedule의 w1도 함께 0으로 둘 것).
    """
    if action_bc is None:
        return torch.zeros(action_rl.shape[0], device=action_rl.device)
    return -((action_rl - action_bc) ** 2).sum(dim=-1)


def _point_to_polyline(point: torch.Tensor, polyline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """배치 point(N,3) -> 배치 polyline(N,K,3) 최근접 세그먼트 거리/진행률/세그먼트 인덱스.

    Returns:
        perp_dist: (N,) 최근접 세그먼트까지 수직거리
        progress: (N,) 경로 시작점부터 투영점까지의 누적 arc-length
        seg_idx: (N,) 최근접 세그먼트 인덱스 (0 ~ K-2)
    """
    starts = polyline[:, :-1, :]  # (N, K-1, 3)
    ends = polyline[:, 1:, :]  # (N, K-1, 3)
    seg = ends - starts  # (N, K-1, 3)
    seg_len = seg.norm(dim=-1).clamp_min(1e-6)  # (N, K-1)

    p = point.unsqueeze(1) - starts  # (N, K-1, 3)
    t = (p * seg).sum(dim=-1) / (seg_len**2)  # (N, K-1)
    t_clamped = t.clamp(0.0, 1.0)
    closest = starts + t_clamped.unsqueeze(-1) * seg  # (N, K-1, 3)
    dist = (point.unsqueeze(1) - closest).norm(dim=-1)  # (N, K-1)

    seg_idx = dist.argmin(dim=-1)  # (N,)
    perp_dist = dist.gather(1, seg_idx.unsqueeze(1)).squeeze(1)  # (N,)
    t_at_min = t_clamped.gather(1, seg_idx.unsqueeze(1)).squeeze(1)  # (N,)

    cum_len = torch.cat(
        [torch.zeros(seg_len.shape[0], 1, device=seg_len.device), seg_len.cumsum(dim=-1)], dim=-1
    )  # (N, K)
    seg_start_len = cum_len.gather(1, seg_idx.unsqueeze(1)).squeeze(1)  # (N,)
    seg_len_at_min = seg_len.gather(1, seg_idx.unsqueeze(1)).squeeze(1)  # (N,)
    progress = seg_start_len + t_at_min * seg_len_at_min  # (N,)

    return perp_dist, progress, seg_idx


def target_speed(curvature: torch.Tensor, thickness: torch.Tensor, base_speed: float = 0.05) -> torch.Tensor:
    """곡률·선 굵기에 따른 목표 진행속도. docs 3.4: 곡률 클수록/선 얇을수록 감속.

    base_speed: 직선/표준 굵기 기준 목표 진행량(한 스텝당, m). 실측 데이터로 튜닝 필요.
    """
    curvature_factor = 1.0 / (1.0 + 5.0 * curvature)  # 곡률 클수록 0에 가까워짐
    thickness_factor = (thickness / thickness.clamp_min(1e-3).mean()).clamp(0.3, 1.5)
    return base_speed * curvature_factor * thickness_factor


def track_reward(
    eef_pos: torch.Tensor,
    target_polyline: torch.Tensor,
    prev_progress: torch.Tensor,
    curvature_at_progress: torch.Tensor,
    thickness_at_progress: torch.Tensor,
    lambda_consistency: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R_track = -|d_perp| + progress_gain - λ*|Δprogress_t - Δprogress_{t-1}(추정)|.

    Returns:
        reward: (N,)
        new_progress: (N,) 다음 스텝의 prev_progress로 사용
    """
    perp_dist, progress, _ = _point_to_polyline(eef_pos, target_polyline)
    delta_progress = (progress - prev_progress).clamp_min(0.0)  # 역행은 0 취급 (단조 진행 가정)
    target = target_speed(curvature_at_progress, thickness_at_progress)
    consistency_penalty = (delta_progress - target).abs()

    reward = -perp_dist + delta_progress - lambda_consistency * consistency_penalty
    return reward, progress


def smoothness_reward(action_t: torch.Tensor, action_tm1: torch.Tensor) -> torch.Tensor:
    """R_smooth = -||a_t - a_{t-1}||^2."""
    return -((action_t - action_tm1) ** 2).sum(dim=-1)


@dataclass
class WeightSchedule:
    """docs 3.4 권장 초기값: w1(모방) 1.0->0.3, w2(추종) 0.3->1.0, w3(부드러움) 0.2 고정."""

    w1_start: float = 1.0
    w1_end: float = 0.3
    w2_start: float = 0.3
    w2_end: float = 1.0
    w3: float = 0.2
    lambda_consistency: float = 0.5

    def weights(self, progress_fraction: float) -> tuple[float, float, float]:
        """progress_fraction: 0.0(학습 시작) ~ 1.0(학습 종료)."""
        f = max(0.0, min(1.0, progress_fraction))
        w1 = self.w1_start + f * (self.w1_end - self.w1_start)
        w2 = self.w2_start + f * (self.w2_end - self.w2_start)
        return w1, w2, self.w3


def total_reward(
    action_rl: torch.Tensor,
    action_tm1: torch.Tensor,
    eef_pos: torch.Tensor,
    target_polyline: torch.Tensor,
    prev_progress: torch.Tensor,
    curvature_at_progress: torch.Tensor,
    thickness_at_progress: torch.Tensor,
    weights: tuple[float, float, float],
    action_bc: torch.Tensor | None = None,
    lambda_consistency: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1, w2, w3 = weights
    r_imit = imitation_reward(action_rl, action_bc)
    r_track, new_progress = track_reward(
        eef_pos, target_polyline, prev_progress, curvature_at_progress, thickness_at_progress, lambda_consistency
    )
    r_smooth = smoothness_reward(action_rl, action_tm1)
    reward = w1 * r_imit + w2 * r_track + w3 * r_smooth
    return reward, new_progress
