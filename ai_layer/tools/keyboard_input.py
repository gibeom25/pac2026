#!/usr/bin/env python
"""키보드(evdev) -> EE 속도 명령. 조이스틱(joystick_input.py) 없을 때 쓰는 대체 입력.

2026-10-06: "조이스틱 없을 때를 대비해서" 요청 — `JoystickEEController`와 **똑같은 공개
인터페이스**(poll/ee_velocity/gripper_bit/rotation_rate/episode_end_requested/
discard_requested/close)를 그대로 구현해서, record_mujoco.py/check_ee.py/record_gui.py
어디서든 교체만 하면 바로 쓸 수 있게 했다(실제 선택 로직은 teleop_input.build_controller()).
joystick_input.py와 같은 evdev 기반이라(새 의존성 없음) 조이스틱과 같은 polling/SYN_DROPPED
복구 패턴을 그대로 재사용한다.

키 배치 (조이스틱의 "누르는 동안 레이트" 관례를 그대로 따름 — 아날로그가 없으니 전부 on/off):
  W/S              +x/-x
  D/A              +y/-y
  R/F              +z/-z
  Q/E              roll -/+
  Z/X              pitch -/+
  C/V              yaw -/+
  SPACE(누르는 동안) 그리퍼/도구 신호 = 1
  ENTER            에피소드 저장+종료 (BTN_THUMB과 동일)
  BACKSPACE        에피소드 폐기+재시도 (BTN_THUMB2와 동일)

단독 실행하면 라이브 진단 모드: 눌린 키/속도를 실시간으로 출력한다.
  PYTHONPATH=. python ai_layer/tools/keyboard_input.py [--list]
"""

from __future__ import annotations

import argparse
import select
import time
from dataclasses import dataclass, field

import evdev
from evdev import ecodes

# evdev 키보드 판별: EV_KEY를 가진 장치는 마우스/조이스틱도 있으므로, 알파벳 키(KEY_A)를
# 가진 것만 "진짜 키보드"로 본다.
_KEYBOARD_PROBE_CODE = ecodes.KEY_A

_KEY_X_POS = ecodes.KEY_W
_KEY_X_NEG = ecodes.KEY_S
_KEY_Y_POS = ecodes.KEY_D
_KEY_Y_NEG = ecodes.KEY_A
_KEY_Z_POS = ecodes.KEY_R
_KEY_Z_NEG = ecodes.KEY_F
_KEY_ROLL_NEG = ecodes.KEY_Q
_KEY_ROLL_POS = ecodes.KEY_E
_KEY_PITCH_NEG = ecodes.KEY_Z
_KEY_PITCH_POS = ecodes.KEY_X
_KEY_YAW_NEG = ecodes.KEY_C
_KEY_YAW_POS = ecodes.KEY_V
_KEY_TRIGGER = ecodes.KEY_SPACE
_KEY_END_EPISODE = ecodes.KEY_ENTER
_KEY_DISCARD = ecodes.KEY_BACKSPACE

_TRACKED_KEYS = (
    _KEY_X_POS, _KEY_X_NEG, _KEY_Y_POS, _KEY_Y_NEG, _KEY_Z_POS, _KEY_Z_NEG,
    _KEY_ROLL_NEG, _KEY_ROLL_POS, _KEY_PITCH_NEG, _KEY_PITCH_POS, _KEY_YAW_NEG, _KEY_YAW_POS,
    _KEY_TRIGGER, _KEY_END_EPISODE, _KEY_DISCARD,
)


def list_keyboards() -> None:
    devs = [evdev.InputDevice(p) for p in evdev.list_devices()]
    if not devs:
        print("[keyboard] /dev/input에 인식된 장치가 없습니다.")
        return
    for d in devs:
        caps = d.capabilities().get(ecodes.EV_KEY, [])
        tag = " (키보드로 보임)" if _KEYBOARD_PROBE_CODE in caps else ""
        print(f"  {d.path}  {d.name}{tag}")


def find_keyboard() -> str:
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if _KEYBOARD_PROBE_CODE in dev.capabilities().get(ecodes.EV_KEY, []):
            return path
    raise RuntimeError(
        "키보드로 보이는 입력 장치를 못 찾았습니다(KEY_A를 가진 장치 없음). "
        "--list로 연결된 장치를 확인하세요. evdev로 읽으려면 보통 'input' 그룹 권한이 필요합니다."
    )


@dataclass
class KeyboardState:
    held: set = field(default_factory=set)


class KeyboardEEController:
    """JoystickEEController와 동일한 공개 인터페이스 — 드롭인 대체용."""

    def __init__(self, device_path: str | None = None):
        path = device_path or find_keyboard()
        self.dev = evdev.InputDevice(path)
        print(f"[keyboard] connected: {self.dev.path} ({self.dev.name})")
        print(
            "[keyboard] 조이스틱 없을 때 대체 입력 — WASD(xy) R/F(z) Q/E(roll) Z/X(pitch) "
            "C/V(yaw) SPACE(트리거) ENTER(저장종료) BACKSPACE(폐기재시도)"
        )
        self.state = KeyboardState()
        self._edge_prev: dict[int, bool] = {}

    def _resync_from_hardware(self) -> None:
        """SYN_DROPPED(커널 이벤트 버퍼 오버플로) 이후 실제 키 상태로 강제 재동기화 —
        joystick_input.py의 같은 패턴(에피소드 사이 input()으로 오래 블로킹하는 동안 이벤트가
        쌓이면 버퍼가 넘칠 수 있음)."""
        active = set(self.dev.active_keys())
        self.state.held = {code for code in _TRACKED_KEYS if code in active}

    def poll(self) -> None:
        while True:
            r, _, _ = select.select([self.dev.fd], [], [], 0)
            if not r:
                return
            try:
                events = list(self.dev.read())
            except BlockingIOError:
                return
            if not events:
                return
            for e in events:
                if e.type == ecodes.EV_SYN and e.code == ecodes.SYN_DROPPED:
                    self._resync_from_hardware()
                    continue
                if e.type == ecodes.EV_KEY and e.code in _TRACKED_KEYS:
                    if e.value == 0:
                        self.state.held.discard(e.code)
                    else:  # 1=press, 2=repeat 전부 "눌림"으로 취급
                        self.state.held.add(e.code)

    def _axis(self, neg_code: int, pos_code: int) -> float:
        held = self.state.held
        return (1.0 if pos_code in held else 0.0) - (1.0 if neg_code in held else 0.0)

    def ee_velocity(
        self,
        max_linear: float = 0.05,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
    ) -> tuple[float, float, float]:
        vx = self._axis(_KEY_X_NEG, _KEY_X_POS) * max_linear
        vy = self._axis(_KEY_Y_NEG, _KEY_Y_POS) * max_linear
        vz = self._axis(_KEY_Z_NEG, _KEY_Z_POS) * max_linear
        if invert_x:
            vx = -vx
        if invert_y:
            vy = -vy
        if invert_z:
            vz = -vz
        return vx, vy, vz

    def gripper_bit(self) -> float:
        return 1.0 if _KEY_TRIGGER in self.state.held else 0.0

    def rotation_rate(
        self, max_angular: float = 1.0, invert_x: bool = False, invert_y: bool = False, invert_z: bool = False
    ) -> tuple[float, float, float]:
        wx = self._axis(_KEY_ROLL_NEG, _KEY_ROLL_POS) * max_angular
        wy = self._axis(_KEY_PITCH_NEG, _KEY_PITCH_POS) * max_angular
        wz = self._axis(_KEY_YAW_NEG, _KEY_YAW_POS) * max_angular
        if invert_x:
            wx = -wx
        if invert_y:
            wy = -wy
        if invert_z:
            wz = -wz
        return wx, wy, wz

    def _rising_edge(self, code: int) -> bool:
        now = code in self.state.held
        was = self._edge_prev.get(code, False)
        self._edge_prev[code] = now
        return now and not was

    def episode_end_requested(self) -> bool:
        return self._rising_edge(_KEY_END_EPISODE)

    def discard_requested(self) -> bool:
        return self._rising_edge(_KEY_DISCARD)

    def close(self) -> None:
        self.dev.close()


def _live_diagnostic() -> None:
    ctl = KeyboardEEController()
    print("[keyboard] 키를 눌러보세요. Ctrl+C로 종료.\n")
    try:
        while True:
            ctl.poll()
            vx, vy, vz = ctl.ee_velocity()
            wx, wy, wz = ctl.rotation_rate()
            held_names = [ecodes.KEY.get(c, str(c)) for c in ctl.state.held]
            print(
                f"\rv=({vx:+.3f},{vy:+.3f},{vz:+.3f}) w=({wx:+.2f},{wy:+.2f},{wz:+.2f}) "
                f"trigger={int(ctl.gripper_bit())}  held={held_names}" + " " * 20,
                end="",
                flush=True,
            )
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("\n[keyboard] 종료.")
    finally:
        ctl.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="키보드 연결/매핑 진단 도구 (조이스틱 대체 입력).")
    p.add_argument("--list", action="store_true", help="연결된 입력 장치 목록만 출력하고 종료")
    args = p.parse_args()
    if args.list:
        list_keyboards()
    else:
        _live_diagnostic()
