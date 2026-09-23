#!/usr/bin/env python
"""Logitech Extreme 3D Pro(evdev) -> EE 속도 명령.

로봇 몸통(5관절 체인) 없이 EE만 시뮬레이션하기로 하면서(2026-09-23), 리더암 대신 조이스틱으로
mocap_target(`assets/so101/ee_rig.xml`)을 직접 움직인다. 리더 FK가 필요 없어지고(조이스틱 자체가
EE-space 입력), 기구부 백래시/떨림 없이 더 매끄러운 시연 궤적을 얻을 수 있다는 게 의도.

실제 장치에서 확인된 매핑 (2026-09-23, `/dev/input/event21` "Logitech Logitech Extreme 3D",
evdev.InputDevice.capabilities()로 직접 읽음 — 추측 아님):
  ABS_X (0~1023, center 508)       좌우 스틱 -> EE y 속도
  ABS_Y (0~1023, center 508)       전후 스틱 -> EE x 속도 (스틱 전방 = +x)
  ABS_THROTTLE (0~255, 절대위치)    슬라이더 -> EE z 속도 (자체 복원 없음 — 중앙 근처서 손 떼면 정지)
  ABS_RZ (0~255, center ~128)      트위스트 -> 예약(회전), v1은 미사용 (ZERO_YAW 결정과 일치)
  ABS_HAT0X/ABS_HAT0Y (-1/0/1)     POV 햇스위치, v1은 미사용
  BTN_TRIGGER(288)                 누르는 동안 그리퍼/도구 신호 = 1 (임계값 없이 즉시 반영)

방향(좌우/전후 부호)은 실제로 스틱을 움직여보고 뷰어에서 확인해야 한다 — 반대면 부호만 뒤집을 것
(ee_velocity()의 vx/vy 계산 부분).

단독 실행하면 라이브 진단 모드: 축/버튼 값을 실시간으로 출력한다.
  PYTHONPATH=. python ai_layer/tools/joystick_input.py [--list]
"""

from __future__ import annotations

import argparse
import select
import time
from dataclasses import dataclass, field

import evdev
from evdev import ecodes


def list_joysticks() -> None:
    devs = [evdev.InputDevice(p) for p in evdev.list_devices()]
    if not devs:
        print("[joystick] /dev/input에 인식된 장치가 없습니다.")
        return
    for d in devs:
        print(f"  {d.path}  {d.name}")


def find_joystick(name_substring: str = "Extreme 3D") -> str:
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if name_substring.lower() in dev.name.lower():
            return path
    raise RuntimeError(
        f"'{name_substring}' 이름을 가진 조이스틱을 못 찾았습니다. "
        f"--list로 연결된 장치를 확인하세요."
    )


@dataclass
class JoystickState:
    x: float = 0.0  # -1..1, 좌우 스틱
    y: float = 0.0  # -1..1, 전후 스틱
    throttle: float = 0.0  # -1..1, 슬라이더 (중앙 0 근처가 정지)
    twist: float = 0.0  # -1..1, 트위스트 (v1 미사용)
    hat_x: int = 0
    hat_y: int = 0
    trigger: bool = False
    buttons: dict = field(default_factory=dict)


_AXIS_CODES = (ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_RZ, ecodes.ABS_THROTTLE)


class JoystickEEController:
    def __init__(self, device_path: str | None = None):
        path = device_path or find_joystick()
        self.dev = evdev.InputDevice(path)
        print(f"[joystick] connected: {self.dev.path} ({self.dev.name})")
        self._axis_info = {code: self.dev.absinfo(code) for code in _AXIS_CODES}
        # ABS_THROTTLE은 스프링 복원이 없는 슬라이더 — 실제 장치에서 확인해보니 가만히 둔 상태에서
        # value=0(범위 끝, 중앙이 아님)이었다. 고정 중앙(127.5)을 0으로 잡으면 슬라이더를 안 만져도
        # 계속 최대 속도로 z가 움직여버리므로, 연결 시점의 실제 위치를 "정지" 기준으로 삼는다 —
        # 사용자가 시작 전에 슬라이더를 원하는 중립 위치에 둔 채로 스크립트를 켜면 된다.
        self._throttle_zero_raw = self.dev.absinfo(ecodes.ABS_THROTTLE).value
        t_info = self._axis_info[ecodes.ABS_THROTTLE]
        print(
            f"[joystick] throttle 슬라이더 현재 위치(raw={self._throttle_zero_raw}, "
            f"범위 {t_info.min}~{t_info.max})를 정지(z속도=0) 기준으로 잡습니다."
        )
        margin = (t_info.max - t_info.min) * 0.1
        if self._throttle_zero_raw <= t_info.min + margin or self._throttle_zero_raw >= t_info.max - margin:
            print(
                "[joystick] ⚠️  슬라이더가 끝(min/max) 근처에 있습니다 — 이 상태로 시작하면 z가 "
                "한쪽 방향으로만 움직입니다. Ctrl+C로 종료하고 슬라이더를 중간쯤으로 옮긴 뒤 "
                "다시 실행하는 걸 권장합니다 (지금 이대로 진행하면 반대 방향은 쓸 수 없음)."
            )
        self.state = JoystickState()
        self._sync_from_device()

    def _norm(self, code: int, raw: int) -> float:
        info = self._axis_info[code]
        if code == ecodes.ABS_THROTTLE:
            # 자체 복원 없는 슬라이더라 연결 시점 위치를 "정지" 기준(zero)으로 삼는데, 그 기준을
            # 축 전체 half-span(고정값)으로 나누면 zero가 한쪽 끝 근처일 때 반대 방향은 거의
            # 못 쓰고(값이 -1 근처로 바로 포화) 다른 방향만 넓게 쓰이는 비대칭 버그가 생긴다.
            # zero 기준 "남은 이동 범위"를 방향별로 따로 잡아 양쪽 다 -1..1 전체를 쓸 수 있게 한다.
            zero = self._throttle_zero_raw
            span = (info.max - zero) if raw >= zero else (zero - info.min)
            center = zero
        else:
            span = (info.max - info.min) / 2.0
            center = (info.max + info.min) / 2.0
        if span <= 0:
            return 0.0
        val = (raw - center) / span
        if abs(raw - center) < info.flat:
            val = 0.0
        return float(max(-1.0, min(1.0, val)))

    def _sync_from_device(self) -> None:
        """장치를 처음 열었을 때 현재 값으로 state를 채운다 (움직이기 전 스냅샷)."""
        for code in _AXIS_CODES:
            raw = self.dev.absinfo(code).value
            val = self._norm(code, raw)
            if code == ecodes.ABS_X:
                self.state.x = val
            elif code == ecodes.ABS_Y:
                self.state.y = val
            elif code == ecodes.ABS_RZ:
                self.state.twist = val
            elif code == ecodes.ABS_THROTTLE:
                self.state.throttle = val

    def poll(self) -> None:
        """대기 중인 이벤트를 전부(non-blocking) 읽어 state를 갱신한다."""
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
                if e.type == ecodes.EV_ABS:
                    if e.code == ecodes.ABS_X:
                        self.state.x = self._norm(ecodes.ABS_X, e.value)
                    elif e.code == ecodes.ABS_Y:
                        self.state.y = self._norm(ecodes.ABS_Y, e.value)
                    elif e.code == ecodes.ABS_RZ:
                        self.state.twist = self._norm(ecodes.ABS_RZ, e.value)
                    elif e.code == ecodes.ABS_THROTTLE:
                        self.state.throttle = self._norm(ecodes.ABS_THROTTLE, e.value)
                    elif e.code == ecodes.ABS_HAT0X:
                        self.state.hat_x = e.value
                    elif e.code == ecodes.ABS_HAT0Y:
                        self.state.hat_y = e.value
                elif e.type == ecodes.EV_KEY:
                    self.state.buttons[e.code] = bool(e.value)
                    if e.code == ecodes.BTN_TRIGGER:
                        self.state.trigger = bool(e.value)

    def ee_velocity(
        self,
        max_linear: float = 0.05,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
    ) -> tuple[float, float, float]:
        """(vx, vy, vz) m/s.

        2026-09-23: 실사용 확인 결과 x/z가 뒤집혀 있어 기본 부호를 반전 (z: 슬라이더 밀면
        내려가야 할 방향이 반대, x: 스틱 전방이 -x가 되어야 함). y는 여전히 애매할 수 있어
        --invert-y로 바로 뒤집어 테스트할 수 있게 열어둠 — check_ee.py로 `target=(x,y,z)`가
        스틱 방향과 맞는지 보고 필요하면 켤 것.
        """
        vx = -self.state.y * max_linear  # 스틱 전방 = -x로 반전
        vy = -self.state.x * max_linear  # 스틱 오른쪽 = -y
        vz = -self.state.throttle * max_linear  # 슬라이더 밀면(+) 아래로(-z) 가도록 반전
        if invert_x:
            vx = -vx
        if invert_y:
            vy = -vy
        if invert_z:
            vz = -vz
        return vx, vy, vz

    def gripper_bit(self) -> float:
        return 1.0 if self.state.trigger else 0.0

    def close(self) -> None:
        self.dev.close()


def _live_diagnostic() -> None:
    ctl = JoystickEEController()
    print("[joystick] 스틱/슬라이더/버튼을 움직여보세요. Ctrl+C로 종료.\n")
    try:
        while True:
            ctl.poll()
            s = ctl.state
            vx, vy, vz = ctl.ee_velocity()
            print(
                f"\rx={s.x:+.2f} y={s.y:+.2f} throttle={s.throttle:+.2f} twist={s.twist:+.2f} "
                f"hat=({s.hat_x:+d},{s.hat_y:+d}) trigger={int(s.trigger)}  |  "
                f"v=({vx:+.3f},{vy:+.3f},{vz:+.3f}) m/s",
                end="",
                flush=True,
            )
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("\n[joystick] 종료.")
    finally:
        ctl.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="조이스틱 연결/매핑 진단 도구.")
    p.add_argument("--list", action="store_true", help="연결된 입력 장치 목록만 출력하고 종료")
    args = p.parse_args()
    if args.list:
        list_joysticks()
    else:
        _live_diagnostic()
