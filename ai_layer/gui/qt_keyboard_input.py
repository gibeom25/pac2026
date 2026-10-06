"""키보드(Qt 네이티브 키 이벤트) -> EE 속도 명령. PyQt GUI 전용 — ai_layer/tools/keyboard_input.py
(터미널 raw 모드, CLI 도구용)와는 별개다.

이 GUI는 애초에 Qt 이벤트 루프 안에서 돈다 — 그러면 `keyPressEvent`/`keyReleaseEvent`가
**진짜 press/release 쌍**을 그대로 준다(OS가 자동반복인지도 `event.isAutoRepeat()`로 구분해줌).
터미널(raw 모드, release 통지 없음)이나 MuJoCo 뷰어 GLFW 콜백(release 통지 없음)과 달리 "뗐다"를
타임아웃으로 흉내낼 필요가 전혀 없다 — 눌림 집합(set)에 넣고 빼기만 하면 된다. GUI 메인 스레드
안에서만 이벤트가 오고 시뮬레이션 틱도 같은 스레드(QTimer)에서 돌 것이므로 락도 필요 없다.

`QApplication.installEventFilter(ctl)`로 등록해서 쓴다 — 특정 위젯에 포커스가 있어야 하는 게
아니라 앱 전체의 키 이벤트를 받는다(필터일 뿐 소비하지 않음 — 다른 위젯의 단축키/타이핑은
그대로 동작).
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import Qt

_KEY_HOLD_MAP: dict[int, str] = {
    Qt.Key.Key_W: "x+", Qt.Key.Key_S: "x-",
    Qt.Key.Key_D: "y+", Qt.Key.Key_A: "y-",
    Qt.Key.Key_R: "z+", Qt.Key.Key_F: "z-",
    Qt.Key.Key_Q: "roll-", Qt.Key.Key_E: "roll+",
    Qt.Key.Key_Z: "pitch-", Qt.Key.Key_X: "pitch+",
    Qt.Key.Key_C: "yaw-", Qt.Key.Key_V: "yaw+",
    Qt.Key.Key_Space: "trigger",
}
_KEY_END_EPISODE = (Qt.Key.Key_Return, Qt.Key.Key_Enter)
_KEY_DISCARD = (Qt.Key.Key_Backspace, Qt.Key.Key_Delete)


class QtKeyboardEEController(QObject):
    """JoystickEEController/KeyboardEEController와 동일한 공개 인터페이스 — 드롭인 대체용."""

    def __init__(self):
        super().__init__()
        self._held: set[str] = set()
        self._pending_end = False
        self._pending_discard = False

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 — Qt override 시그니처
        if isinstance(event, QKeyEvent) and not event.isAutoRepeat():
            if event.type() == QEvent.Type.KeyPress:
                self._on_press(event.key())
            elif event.type() == QEvent.Type.KeyRelease:
                self._on_release(event.key())
        return False  # 소비하지 않음 — 다른 위젯(텍스트 입력 등)도 정상 동작

    def _on_press(self, key: int) -> None:
        name = _KEY_HOLD_MAP.get(key)
        if name is not None:
            self._held.add(name)
        elif key in _KEY_END_EPISODE:
            self._pending_end = True
        elif key in _KEY_DISCARD:
            self._pending_discard = True

    def _on_release(self, key: int) -> None:
        name = _KEY_HOLD_MAP.get(key)
        if name is not None:
            self._held.discard(name)

    def _axis(self, neg: str, pos: str) -> float:
        return (1.0 if pos in self._held else 0.0) - (1.0 if neg in self._held else 0.0)

    def poll(self) -> None:
        """Qt 이벤트가 바로바로 들어오므로 폴링이 필요 없다 — 인터페이스 호환용 no-op."""

    def ee_velocity(
        self, max_linear: float = 0.05, invert_x: bool = False, invert_y: bool = False, invert_z: bool = False
    ) -> tuple[float, float, float]:
        vx = self._axis("x-", "x+") * max_linear
        # 2026-10-06: D(+y 키)를 누르면 손목 카메라 기준 실제로는 왼쪽으로 가는 걸로 실측
        # 확인됨(ai_layer/tools/keyboard_input.py와 같은 이유) — D/A를 서로 바꿨다.
        vy = self._axis("y+", "y-") * max_linear
        vz = self._axis("z-", "z+") * max_linear
        if invert_x:
            vx = -vx
        if invert_y:
            vy = -vy
        if invert_z:
            vz = -vz
        return vx, vy, vz

    def gripper_bit(self) -> float:
        return 1.0 if "trigger" in self._held else 0.0

    def rotation_rate(
        self, max_angular: float = 1.0, invert_x: bool = False, invert_y: bool = False, invert_z: bool = False
    ) -> tuple[float, float, float]:
        wx = self._axis("roll-", "roll+") * max_angular
        wy = self._axis("pitch-", "pitch+") * max_angular
        wz = self._axis("yaw-", "yaw+") * max_angular
        if invert_x:
            wx = -wx
        if invert_y:
            wy = -wy
        if invert_z:
            wz = -wz
        return wx, wy, wz

    def episode_end_requested(self) -> bool:
        v, self._pending_end = self._pending_end, False
        return v

    def discard_requested(self) -> bool:
        v, self._pending_discard = self._pending_discard, False
        return v

    def close(self) -> None:
        """이벤트 필터만 뗴면 됨 — 호출 측이 QApplication.removeEventFilter(ctl)로 처리."""
