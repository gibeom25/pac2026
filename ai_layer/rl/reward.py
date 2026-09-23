"""RL(SAC) 보상 함수. 설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.

R_t = w1*R_imitation + w2*R_track + w3*R_smooth + w4*R_coverage

프레임워크(IsaacLab env)와 독립적인 순수 텐서 함수로 작성 — 어떤 env에서든 재사용 가능.
모든 함수는 (num_envs, ...) 배치 텐서를 받아 (num_envs,) 보상을 반환한다.

R_coverage(2026-09-23 추가, 기범): "선이 얼마나 덮였는지 / 안 덮어야 하는 곳은 얼마나 덮였는지"로
계산하자는 결정. track_reward의 perp_dist+progress는 매 스텝 조밀한 shaping 신호를 주지만 "실제
과제 성공"(선 전체를 빠짐없이, 엉뚱한 데 흘리지 않고 덮었는가)을 직접 측정하진 않는다. coverage_reward는
에피소드 동안 target_polyline 샘플点들 중 실제로 방문(도포)한 비율(recall)과, 도포 신호가 켜진
상태에서 선에서 너무 멀리 떨어진 곳에 있었던 정도(precision 위반)를 직접 본다 — 둘을 합쳐서 최종
"용접선을 얼마나 깨끗하게 완주했는가"에 가장 가까운 항. 희소 신호(새로 덮은 구간에서만 양의 보상)라
기존 조밀한 track_reward를 대체하지 않고 추가 항으로 합산한다(대체하면 학습 초반 그라디언트가 너무
약해짐).
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


def target_speed(curvature: torch.Tensor, thickness: torch.Tensor, base_speed: float = 0.01) -> torch.Tensor:
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
    base_speed: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R_track = -|d_perp| + progress_gain - λ*|Δprogress_t - Δprogress_{t-1}(추정)|.

    base_speed: 목표 진행량(m/step). env의 최대 위치 delta(action_scale_pos)보다 작아야 한다.
    (예전 기본 0.05는 action_scale_pos 0.02보다 커서 로봇이 물리적으로 도달 불가 → 상시 패널티)

    Returns:
        reward: (N,)
        new_progress: (N,) 다음 스텝의 prev_progress로 사용
    """
    perp_dist, progress, _ = _point_to_polyline(eef_pos, target_polyline)
    delta_progress = (progress - prev_progress).clamp_min(0.0)  # 역행은 0 취급 (단조 진행 가정)
    target = target_speed(curvature_at_progress, thickness_at_progress, base_speed=base_speed)
    consistency_penalty = (delta_progress - target).abs()

    reward = -perp_dist + delta_progress - lambda_consistency * consistency_penalty
    return reward, progress


def coverage_reward(
    eef_pos: torch.Tensor,
    target_polyline: torch.Tensor,
    coverage_mask: torch.Tensor,
    gripper_active: torch.Tensor,
    coverage_radius: float = 0.008,
    off_target_penalty_scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R_coverage = 새로 덮은 비율(recall 증가분) - 도포 중 선 밖에 있었던 정도(precision 위반).

    target_polyline의 샘플点(K개)마다 "지금까지 한 번이라도 coverage_radius 안으로 들어온 적
    있는가"를 coverage_mask(bool)로 누적 추적한다(에피소드 시작 시 전부 False로 리셋해서 호출
    측이 관리 — reset()에서 zeros_like(target_polyline[...,0])로 초기화할 것). 매 스텝:
      1. eef_pos와 모든 샘플点 사이 거리 계산 (가장 가까운 세그먼트가 아니라 "샘플点 각각"과의
         거리 — 코너/분기처럼 폴리라인이 자기 근처를 지나는 구간에서도 "실제로 그 지점 자체를
         지나갔는가"를 보려는 것. track_reward의 최근접 세그먼트 거리와 성격이 다름).
      2. coverage_radius 안에 들어온(=방문한) 샘플点 중 이번에 처음 들어온 것만 카운트 -> 새로
         덮인 비율(K분의 몇 점)을 양의 보상으로.
      3. 도포 신호(gripper_active)가 켜져 있는데 가장 가까운 샘플点까지의 거리가 coverage_radius를
         넘으면(=선 밖에 도포 중) 그 초과분에 비례해 음의 보상 — "안 덮어야 하는 곳에 덮음" 패널티.

    Args:
        gripper_active: (N,) 0/1 (또는 bool) — 이번 스텝에 실제로 도포(비드 증착)가 일어났는지.
    Returns:
        reward: (N,)
        new_coverage_mask: (N, K) — 다음 스텝에 그대로 넘길 것 (에피소드 끝나면 새로 초기화).
    """
    dists = (eef_pos.unsqueeze(1) - target_polyline).norm(dim=-1)  # (N, K)
    visited_now = dists <= coverage_radius  # (N, K)
    newly_covered = visited_now & (~coverage_mask)
    new_mask = coverage_mask | visited_now

    k = target_polyline.shape[1]
    coverage_gain = newly_covered.float().sum(dim=-1) / max(k, 1)  # (N,)

    min_dist = dists.min(dim=-1).values  # (N,)
    off_target_excess = (min_dist - coverage_radius).clamp_min(0.0)
    off_target_penalty = gripper_active.float() * off_target_excess

    reward = coverage_gain - off_target_penalty_scale * off_target_penalty
    return reward, new_mask


def smoothness_reward(action_t: torch.Tensor, action_tm1: torch.Tensor) -> torch.Tensor:
    """R_smooth = -||a_t - a_{t-1}||^2."""
    return -((action_t - action_tm1) ** 2).sum(dim=-1)


@dataclass
class WeightSchedule:
    """docs 3.4 권장 초기값: w1(모방) 1.0->0.3, w2(추종) 0.3->1.0, w3(부드러움) 0.2 고정,
    w4(coverage) 0.5 고정(2026-09-23 추가) — 희소 신호라 스케줄링 없이 처음부터 일정 비중."""

    w1_start: float = 1.0
    w1_end: float = 0.3
    w2_start: float = 0.3
    w2_end: float = 1.0
    w3: float = 0.2
    w4: float = 0.5
    lambda_consistency: float = 0.5

    def weights(self, progress_fraction: float) -> tuple[float, float, float, float]:
        """progress_fraction: 0.0(학습 시작) ~ 1.0(학습 종료)."""
        f = max(0.0, min(1.0, progress_fraction))
        w1 = self.w1_start + f * (self.w1_end - self.w1_start)
        w2 = self.w2_start + f * (self.w2_end - self.w2_start)
        return w1, w2, self.w3, self.w4


def total_reward(
    action_rl: torch.Tensor,
    action_tm1: torch.Tensor,
    eef_pos: torch.Tensor,
    target_polyline: torch.Tensor,
    prev_progress: torch.Tensor,
    curvature_at_progress: torch.Tensor,
    thickness_at_progress: torch.Tensor,
    weights: tuple[float, float, float, float],
    coverage_mask: torch.Tensor,
    gripper_active: torch.Tensor,
    action_bc: torch.Tensor | None = None,
    lambda_consistency: float = 0.5,
    base_speed: float = 0.01,
    coverage_radius: float = 0.008,
    off_target_penalty_scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """action_rl / action_bc / action_tm1 은 모두 같은 스케일(env 정규화 [-1,1])이어야 한다.

    coverage_mask: (N, K) bool, 에피소드 시작 시 전부 False로 호출 측이 초기화 — coverage_reward
    참고. gripper_active: (N,) 이번 스텝 실제 도포 여부(0/1 또는 bool).

    Returns:
        reward, new_progress, new_coverage_mask
    """
    w1, w2, w3, w4 = weights
    r_imit = imitation_reward(action_rl, action_bc)
    r_track, new_progress = track_reward(
        eef_pos, target_polyline, prev_progress, curvature_at_progress, thickness_at_progress,
        lambda_consistency, base_speed=base_speed,
    )
    r_smooth = smoothness_reward(action_rl, action_tm1)
    r_coverage, new_coverage_mask = coverage_reward(
        eef_pos, target_polyline, coverage_mask, gripper_active,
        coverage_radius=coverage_radius, off_target_penalty_scale=off_target_penalty_scale,
    )
    reward = w1 * r_imit + w2 * r_track + w3 * r_smooth + w4 * r_coverage
    return reward, new_progress, new_coverage_mask
