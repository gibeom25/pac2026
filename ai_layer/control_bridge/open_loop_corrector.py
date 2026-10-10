"""OpenLoopCorrector — Problem.md 5번(청크 내부 열린 루프) 담당, "아이디어" 단계였던 걸 구현.

Feedforward(AI chunk) + Feedback(이 모듈) 구조: chunk는 "큰 그림 계획"으로만 쓰고, 매 틱
실측 위치가 들어올 때마다 그 오차를 반영해 위치/속도 추정을 보정한다. constant-velocity(등속)
운동 모델 기반의 표준 칼만 필터로 구현한다(공분산 P를 매 틱 전파/갱신) — 상태(위치, 속도)
6차원, 관측(위치) 3차원. 설명은 docs/Experiment_Plan.md "핵심 기법 설명" 절 참고.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConstantVelocityKF:
    """상태 x=[pos(3), vel(3)]. predict()는 chunk 실행 중 매 틱(관측 없이) 외삽,
    update()는 실제 관측(측정 위치)이 들어왔을 때 보정."""

    process_var: float = 1e-6  # 가속도(모델화 안 된 거동) 분산 — 클수록 예측을 덜 신뢰
    measurement_var: float = 1e-8  # 관측 노이즈 분산 — 클수록 관측을 덜 신뢰

    _pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _P: np.ndarray = field(default_factory=lambda: np.eye(6) * 1e-3)  # 공분산(6x6)
    _initialized: bool = False

    def reset(self, pos0: np.ndarray) -> None:
        self._pos = np.asarray(pos0, dtype=float).copy()
        self._vel = np.zeros(3)
        self._P = np.eye(6) * 1e-3
        self._initialized = True

    def predict(self, dt_ns: int) -> np.ndarray:
        """dt_ns만큼 등속 외삽. 열린 루프 구간(AI chunk 실행 중, 새 관측 없음)에서 호출.

        표준 선형 KF 전이를 그대로 쓴다: F=[[I, dt*I],[0,I]], P <- F P F^T + Q.
        (처음엔 블록별로 손으로 근사했다가, 위치-속도 교차공분산 항이 빠져서 속도 추정이
        전혀 안 되는 버그가 났다 — 실측으로 확인 후 제대로 된 행렬 전파로 교체.)
        """
        if not self._initialized:
            return self._pos.copy()
        dt = dt_ns * 1e-9
        self._pos = self._pos + self._vel * dt

        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.process_var * dt
        Q = np.eye(6) * q
        self._P = F @ self._P @ F.T + Q
        return self._pos.copy()

    def update(self, measured_pos: np.ndarray) -> np.ndarray:
        """실측 위치 관측 — Kalman gain으로 위치/속도 보정."""
        z = np.asarray(measured_pos, dtype=float)
        if not self._initialized:
            self.reset(z)
            return self._pos.copy()
        H = np.hstack([np.eye(3), np.zeros((3, 3))])  # 관측은 위치만
        R = np.eye(3) * self.measurement_var
        y = z - self._pos  # innovation
        S = self._P[:3, :3] + R
        K = self._P[:, :3] @ np.linalg.inv(S)  # (6,3) Kalman gain
        dx = K @ y  # (6,)
        self._pos = self._pos + dx[:3]
        self._vel = self._vel + dx[3:]
        self._P = self._P - K @ H @ self._P
        return self._pos.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._vel.copy()
