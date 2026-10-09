"""TriggerLogic — Problem.md 3번(트리거 신호 불안정성) 담당.

chunk가 미리 계획한 트리거(eef_bit)는 미래로 갈수록 신뢰도가 떨어진다(5번, 열린 루프).
그래서 트리거만큼은 chunk의 계획을 그대로 안 믿고 매 틱 실시간으로 재평가한다 — AI 레이어
시뮬레이션 코드(so101_seam_env.py)의 off_seam_safety_dist 강제 OFF 패턴을 일반화한 것.

히스테리시스(on_tol != off_tol)로 경계에서 깜빡이는(채터링) 걸 막는다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriggerLogic:
    on_tol: float = 0.0015  # 이 거리 안이면 트리거 ON 허용
    off_tol: float = 0.003  # 이 거리 넘으면 강제 OFF (on_tol보다 커야 히스테리시스 성립)

    _active: bool = False

    def decide(self, requested_bit: float, dist_to_line: float) -> bool:
        """chunk가 요청한 bit(0/1 근방 float)와 실측 선까지 거리로 최종 on/off 결정.

        로직: chunk가 OFF를 요청하면 무조건 OFF(= 안전 쪽으로). chunk가 ON을 요청해도
        dist_to_line이 off_tol을 넘으면 강제 OFF. 이미 ON 상태였다면 off_tol을 넘기 전까진
        유지(히스테리시스) — 반대로 OFF 상태였다면 on_tol 안에 들어와야만 새로 ON.
        """
        requested = requested_bit > 0.5
        if not requested:
            self._active = False
            return False

        if self._active:
            self._active = dist_to_line <= self.off_tol
        else:
            self._active = dist_to_line <= self.on_tol
        return self._active
