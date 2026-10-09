"""ChunkBuffer — Problem.md 1번(버퍼 고갈) 담당.

제어 쪽(여기서는 PointMass 스탠드인)이 들고 있는 "지금 재생 중인 chunk" 상태 기계.
- 정상 구간(앞 safe_zone_frac, 기본 2/3)은 chunk의 nominal dt_ns 그대로 재생한다.
- 그 뒤(tail)로 넘어갔는데 새 chunk가 아직 안 왔으면, 남은 스텝들을 지수적으로 시간축을
  늘려(점점 느리게) 재생한다 — ControlMode.DRAINING. 지수 감쇠는 점근적으로만 끝점에
  가까워지므로, max_drain_ns를 넘으면 ControlMode.HOLDING으로 전환해 완전히 멈춘다
  (Problem.md에서 합의한 하한선).
- remaining_commit()은 지금 이 순간부터 예상되는 미래 궤적을 CommitTrajectory로 내보낸다 —
  DRAINING 중이면 느려진 실제 진행을 그대로 반영해서 계산한다(고정 dt_ns 샘플링 그리드는
  유지하되, 그 시점에 실제로 있을 위치를 느려진 속도로 다시 계산) — 그래야 AI 쪽
  choose_anchor()의 COMMIT_END 예측이 "정상 속도로 거기 도착할 것"이라고 잘못 가정하지 않는다
  (DRAINING과 COMMIT_END를 일관되게 잇는 지점, Experiment_Plan.md 참고).

이 프로젝트는 회전 없이 XYZ만 추종하는 벤치마크(Problem.md 참고)라, steps[:,3:6](회전)은
무시하고 위치만 적분한다 — 실제 로봇 중간계층에 옮길 땐 회전도 같은 방식으로 적분하면 된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ai_layer.control_bridge.protocol import MAX_COMMIT, ActionChunk, CommitTrajectory, ControlMode


@dataclass
class ChunkBuffer:
    safe_zone_frac: float = 2.0 / 3.0
    drain_tau_ns: float = 200_000_000.0  # 지수 감쇠 시정수 — 클수록 천천히 느려짐
    max_drain_ns: float = 2_000_000_000.0  # 하한선: tail 진입 후 이만큼 지나면 완전 정지(HOLD)

    _steps: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    _eef: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _dt_ns: int = 33_333_333
    _load_t_ns: int = 0
    _anchor_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # load() 시점의 실제 위치
    _last_progress: float = 0.0  # anchor_resync가 읽고 갱신하는 단조증가 기준값(reference_polyline 진행률)
    _held_pos: np.ndarray | None = None  # HOLDING 진입 시 고정된 위치
    _last_mode: ControlMode = ControlMode.IDLE  # next_step()이 갱신, mode 프로퍼티가 참조

    def reset(self, pos0: np.ndarray) -> None:
        """IDLE(아직 chunk 못 받음) 상태의 기본 위치를 world 원점(dataclass 기본값)이 아니라
        실제 시작 위치로 맞춘다 — 2026-10-10 joint_dynamics_bench.py로 실제 로봇(원점이 아닌
        위치에서 시작)을 검증하다가 실측 확인된 버그: point-mass 벤치마크는 우연히 시작
        위치=(0,0,0)=이 필드의 기본값이라 안 드러났지만, 실제 로봇처럼 원점이 아닌 곳에서
        시작하면 첫 chunk 도착 전(IDLE)에 _anchor_pos 기본값(0,0,0)으로 "원점으로 가라"는
        명령이 나가버린다 — 실제 로봇이면 토크 포화/위험한 급동작으로 이어지는 심각한 버그."""
        pos0 = np.asarray(pos0, dtype=float).copy()
        self._anchor_pos = pos0
        self._held_pos = pos0.copy()

    def load(self, chunk: ActionChunk, start_index: int, anchor_pos: np.ndarray, now_ns: int) -> None:
        """새 chunk를 start_index부터 적재. anchor_pos는 이 chunk의 적분을 시작할 실제 현재 위치
        (anchor_resync.resync_start_index가 고른 지점의 "실제 현재 위치" — 8번 문제의 핵심)."""
        self._steps = np.asarray(chunk.steps[start_index:], dtype=float).copy()
        self._eef = np.asarray(chunk.eef[start_index:], dtype=float).copy()
        self._dt_ns = int(chunk.dt_ns)
        self._load_t_ns = int(now_ns)
        self._anchor_pos = np.asarray(anchor_pos, dtype=float).copy()
        self._held_pos = None

    @property
    def n(self) -> int:
        return int(self._steps.shape[0])

    @property
    def _safe_zone_end(self) -> float:
        """2026-10-10: 예전엔 int()로 정수 스텝까지 반올림했는데, anchor_resync가 많이 스킵해서
        남은 스텝 n이 작아지면(예: n=2) 반올림 오차가 커진다(int(2*0.667)=1 -> 의도한 2/3 지점이
        아니라 50% 지점에서 DRAINING 시작). _integrate_to/tail_elapsed 둘 다 이미 소수 스텝
        인덱스를 지원하므로, float 그대로 둬서 반올림 오차 자체를 없앤다."""
        return max(1.0, self.n * self.safe_zone_frac)

    def _nominal_progress(self, now_ns: int) -> float:
        """DRAINING 없이 그대로 쭉 재생했다면 지금쯤 몇 번째 스텝(소수 가능)일지."""
        return (now_ns - self._load_t_ns) / max(self._dt_ns, 1)

    def _effective_progress(self, now_ns: int) -> tuple[float, ControlMode]:
        """DRAINING 지수 감쇠를 반영한 실제 유효 진행도(소수 스텝 인덱스)와 그 순간의 ControlMode."""
        if self.n == 0:
            return 0.0, ControlMode.IDLE
        nominal = self._nominal_progress(now_ns)
        safe_end = self._safe_zone_end
        if nominal < safe_end:
            return min(nominal, self.n - 1), ControlMode.TRACKING

        tail_elapsed = now_ns - (self._load_t_ns + safe_end * self._dt_ns)
        tail_len = self.n - safe_end
        if tail_elapsed >= self.max_drain_ns:
            return float(self.n - 1), ControlMode.HOLDING
        # 1 - exp(-t/tau): tail_elapsed=0 -> 0, tail_elapsed->inf -> 1 (점근적으로만 도달)
        frac = 1.0 - math.exp(-tail_elapsed / self.drain_tau_ns)
        return safe_end + frac * tail_len, ControlMode.DRAINING

    def _integrate_to(self, progress: float) -> np.ndarray:
        """anchor_pos + steps[0:progress] 누적합(소수점 인덱스는 그 지점까지 선형보간) -> 절대 위치(3,)."""
        if self.n == 0:
            return self._anchor_pos.copy()
        idx = max(0.0, min(progress, self.n - 1e-6))
        i0 = int(idx)
        frac = idx - i0
        cum = self._anchor_pos.copy()
        if i0 > 0:
            cum = cum + self._steps[:i0, :3].sum(axis=0)
        if frac > 0 and i0 < self.n:
            cum = cum + self._steps[i0, :3] * frac
        return cum

    def next_step(self, now_ns: int) -> tuple[np.ndarray, float, ControlMode]:
        """이번 틱의 (절대 위치(3,), eef_bit, 현재 ControlMode)."""
        if self.n == 0:
            pos = self._held_pos if self._held_pos is not None else self._anchor_pos
            self._last_mode = ControlMode.IDLE
            return pos.copy(), 0.0, ControlMode.IDLE
        progress, mode = self._effective_progress(now_ns)
        self._last_mode = mode
        if mode == ControlMode.HOLDING:
            if self._held_pos is None:
                self._held_pos = self._integrate_to(progress)
            return self._held_pos.copy(), 0.0, mode
        pos = self._integrate_to(progress)
        eef_idx = min(int(progress), self.n - 1)
        return pos, float(self._eef[eef_idx]), mode

    def peek_pos(self, now_ns: int) -> np.ndarray:
        """next_step()처럼 내부 상태(held_pos/last_mode)를 안 건드리고 현재 위치만 조회 —
        PointMassChunkSink가 새 chunk 도착 시 "지금 실제 위치"를 anchor_resync에 넘길 때 씀."""
        if self.n == 0:
            return (self._held_pos if self._held_pos is not None else self._anchor_pos).copy()
        progress, mode = self._effective_progress(now_ns)
        if mode == ControlMode.HOLDING and self._held_pos is not None:
            return self._held_pos.copy()
        return self._integrate_to(progress)

    @property
    def mode(self) -> ControlMode:
        """가장 최근 next_step() 호출 시점의 ControlMode (캐시) — PointMassSnapshotSource가
        StateSnapshot.mode에 그대로 쓴다."""
        return self._last_mode

    @property
    def last_progress(self) -> float:
        return self._last_progress

    @last_progress.setter
    def last_progress(self, value: float) -> None:
        self._last_progress = value

    def remaining_commit(self, now_ns: int, n: int = MAX_COMMIT, dt_ns: int = 10_000_000) -> CommitTrajectory:
        """지금부터 dt_ns 간격으로 n개, DRAINING 중이면 느려진 실제 진행을 반영해서 미래 위치 예측.

        고정된 dt_ns 샘플링 그리드(제어 계층과 맞춘 관례)는 유지하되, 각 샘플 시점의 "내용"
        (poses)은 _effective_progress()를 통해 DRAINING 여부를 반영해서 계산한다 — 그래야
        COMMIT_END 앵커링이 "정상 속도로 거기 도착" 가정을 안 하게 된다."""
        poses = np.zeros((n, 7), dtype=float)
        for k in range(n):
            t_future = now_ns + k * dt_ns
            progress, _mode = self._effective_progress(t_future)
            pos = self._integrate_to(progress)
            poses[k, :3] = pos
            poses[k, 3:] = [0.0, 0.0, 0.0, 1.0]  # 회전 미사용(identity quat, xyzw)
        return CommitTrajectory(t_start_ns=now_ns, dt_ns=dt_ns, poses=poses)
