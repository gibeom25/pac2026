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
  ABS_RZ (0~255, center ~128)      트위스트 -> yaw 각속도
  ABS_HAT0X/ABS_HAT0Y (-1/0/1)     POV 햇스위치(스틱 위쪽 미니 조이스틱) -> roll/pitch 각속도
  BTN_TRIGGER(288)                 누르는 동안 그리퍼/도구 신호 = 1 (임계값 없이 즉시 반영)
  BTN_BASE/BTN_BASE2(294/295)      더 이상 안 씀(2026-10-06 이전엔 roll -/+였음)
  BTN_BASE3/BTN_BASE4(296/297)     더 이상 안 씀(2026-10-06 이전엔 pitch -/+였음)

2026-09-23(2차): roll/pitch는 베이스 버튼(레이트 컨트롤 — 누르고 있는 동안만 회전, throttle과
같은 방식)으로, yaw는 트위스트 축(연속값 — 손목을 실제로 돌리는 축이라 버튼 두 개보다 자연스러움,
5차 변경)으로 조절한다(rotation_rate()). mocap_target이 위치+전체 회전을 같이 명령하고, ee_body는
weld+접촉 반발력으로 따라간다.

2026-10-06(6차): roll/pitch를 베이스 버튼 대신 햇스위치로 바꿨다 — 엄지로 미니 스틱 하나
누르는 게 버튼 두 쌍을 누르는 것보다 자연스럽다는 요청. 레이트 컨트롤인 건 동일(기울인 동안만
회전).

방향(좌우/전후 부호)은 실제로 스틱을 움직여보고 뷰어에서 확인해야 한다 — 반대면 부호만 뒤집을 것
(ee_velocity()의 vx/vy 계산 부분).

단독 실행하면 라이브 진단 모드: 축/버튼 값을 실시간으로 출력한다.
  PYTHONPATH=. python ai_layer/tools/joystick_input.py [--list] [--calibrate]

2026-10-06: throttle 보정값을 파일로 저장/재사용하게 바꿨다 — "매번 켤 때마다 영점이 다르고,
특히 z축이 한쪽 끝에서 영점으로 잡혀서 한 방향으로만 움직인다"는 문제(기범) 원인은, 이전엔
스크립트를 켤 때마다 "그 순간 슬라이더가 어디 있든" 그 값을 그대로 영점으로 썼기 때문이다
(자체 복원이 없는 슬라이더라 이전 세션에서 쓰던 자리에 그대로 멈춰 있는데, 그게 끝 쪽이면
그대로 끝이 영점이 됨). 이제 처음 한 번 `_calibrate_throttle_zero()`로 "원하는 중립 위치"를
직접 지정해서 `~/.config/pac2026/joystick_calibration.json`에 저장해두고, 그다음부터는 슬라이더가
실제로 어디 있는지와 무관하게 항상 그 저장된 값을 영점 기준으로 쓴다(재보정은 --recalibrate).
"""

from __future__ import annotations

import argparse
import json
import select
import time
from dataclasses import dataclass, field
from pathlib import Path

import evdev
from evdev import ecodes

CALIBRATION_PATH = Path.home() / ".config" / "pac2026" / "joystick_calibration.json"


def load_calibration() -> dict | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_calibration(data: dict) -> None:
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2))


def list_joysticks() -> None:
    devs = [evdev.InputDevice(p) for p in evdev.list_devices()]
    if not devs:
        from ai_layer.tools.teleop_input import diagnose_no_devices

        msg = diagnose_no_devices()
        print(f"[joystick] {msg}" if msg else "[joystick] /dev/input에 인식된 장치가 없습니다.")
        return
    for d in devs:
        print(f"  {d.path}  {d.name}")


def find_joystick(name_substring: str = "Extreme 3D") -> str:
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        if name_substring.lower() in dev.name.lower():
            return path

    from ai_layer.tools.teleop_input import diagnose_no_devices

    perm_msg = diagnose_no_devices()
    if perm_msg:
        raise RuntimeError(f"'{name_substring}' 이름을 가진 조이스틱을 못 찾았습니다.\n{perm_msg}")
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


def _calibrate_throttle_zero(dev: evdev.InputDevice) -> int:
    """스로틀 슬라이더의 "정지(z속도=0)" 위치를 대화형으로 지정받아 raw 값을 반환.

    자체 복원이 없는 슬라이더라 "지금 어디 있는지"가 아니라 "원하는 중립 위치가 어디인지"를
    물어봐야 한다 — 이걸 한 번 저장해두면 다음부터는 슬라이더가 실제로 어디 있든 이 값을
    기준으로 삼는다(이전 방식: 켤 때마다 그 순간 위치를 영점으로 썼다가, 하필 끝에 놓여 있으면
    그대로 끝이 영점이 돼버리는 문제가 있었음).
    """
    t_info = dev.absinfo(ecodes.ABS_THROTTLE)
    print(
        f"[joystick] 스로틀 보정 — 슬라이더를 z속도=0(정지)으로 쓸 중립 위치에 놓고 Enter "
        f"(범위 {t_info.min}~{t_info.max}, 보통 중간쯤을 추천)"
    )
    input()
    raw = dev.absinfo(ecodes.ABS_THROTTLE).value
    margin = (t_info.max - t_info.min) * 0.1
    if raw <= t_info.min + margin or raw >= t_info.max - margin:
        print(
            f"[joystick] ⚠️  지금 위치(raw={raw})가 끝(min/max) 근처입니다 — 이대로 저장하면 z가 "
            f"한쪽 방향으로만 움직입니다. 괜찮으면 Enter, 다시 하려면 Ctrl+C."
        )
        input()
    print(f"[joystick] 스로틀 영점 raw={raw}로 보정 완료.")
    return raw


class JoystickEEController:
    def __init__(self, device_path: str | None = None, recalibrate: bool = False):
        path = device_path or find_joystick()
        self.dev = evdev.InputDevice(path)
        print(f"[joystick] connected: {self.dev.path} ({self.dev.name})")
        self._axis_info = {code: self.dev.absinfo(code) for code in _AXIS_CODES}

        calib = None if recalibrate else load_calibration()
        if calib is not None and "throttle_zero" in calib:
            self._throttle_zero_raw = calib["throttle_zero"]
            print(
                f"[joystick] 저장된 스로틀 보정값 사용: raw={self._throttle_zero_raw} "
                f"({CALIBRATION_PATH}, 재보정하려면 --recalibrate-joystick)"
            )
        else:
            if calib is None and not recalibrate:
                print("[joystick] 저장된 보정값이 없습니다 — 처음 한 번만 하면 다음부턴 자동으로 재사용됩니다.")
            self._throttle_zero_raw = _calibrate_throttle_zero(self.dev)
            save_calibration({"throttle_zero": self._throttle_zero_raw, "device_name": self.dev.name})
            print(f"[joystick] 저장됨: {CALIBRATION_PATH}")

        self.state = JoystickState()
        self._sync_from_device()
        self._edge_prev: dict[int, bool] = {}  # episode_end_requested()/discard_requested() 엣지 검출용

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

    def _resync_from_hardware(self) -> None:
        """SYN_DROPPED(커널 이벤트 버퍼 오버플로) 이후 실제 하드웨어 현재 상태로 강제 재동기화.

        record_mujoco.py에서 에피소드 사이 input()으로 오래 블로킹하는 동안(poll()을 안 부름)
        조이스틱이 계속 이벤트를 만들어내면 커널 버퍼가 넘칠 수 있다 — 그러면 release 이벤트가
        유실돼 버튼이 실제로는 떼졌는데도 state.buttons에는 계속 눌린 걸로 남을 수 있다
        (에피소드 폐기가 계속 반복되는 버그로 관측됨, 2026-09-23). active_keys()/absinfo는
        이벤트 스트림과 무관하게 지금 이 순간의 진짜 상태를 직접 읽어오므로 이걸로 덮어쓴다.
        """
        self._sync_from_device()
        active = set(self.dev.active_keys())
        for code in list(self.state.buttons.keys()) + [ecodes.BTN_TRIGGER, ecodes.BTN_THUMB, ecodes.BTN_THUMB2]:
            self.state.buttons[code] = code in active
        self.state.trigger = ecodes.BTN_TRIGGER in active

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
                if e.type == ecodes.EV_SYN and e.code == ecodes.SYN_DROPPED:
                    self._resync_from_hardware()
                    continue
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
        내려가야 할 방향이 반대, x: 스틱 전방이 -x가 되어야 함).

        2026-10-06(1차): y도 반대로 느껴진다는 피드백 — 조이스틱 "스틱 오른쪽"과 키보드 D가
        서로 다른 부호를 쓰고 있어서(장치 간 불일치 버그) 키보드 쪽("스틱 오른쪽 = +y")에
        맞춰 통일했었다.

        2026-10-06(2차): 그런데 손목 카메라로 직접 확인해보니 그 "키보드 D=+y" 관례 자체가
        틀렸었다 — 손목 카메라가 블록 왼쪽에 달려 있어서(ee_rig.xml) +y로 가면 카메라 화면
        기준 실제로는 왼쪽으로 간다(렌더 비교로 실측). keyboard_input.py/qt_keyboard_input.py의
        D/A를 서로 바꿔서 "D = 손목 카메라 기준 오른쪽"이 되게 고쳤고, 조이스틱도 그 새
        관례("스틱 오른쪽 = -y")에 다시 맞췄다. 그래도 반대로 느껴지면 --invert-y로 뒤집을 것.
        """
        vx = -self.state.y * max_linear  # 스틱 전방 = -x로 반전
        vy = -self.state.x * max_linear  # 스틱 오른쪽 = -y (키보드 D=-y와 통일, 2026-10-06 2차)
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

    def _rising_edge(self, code: int) -> bool:
        """지금 눌려 있고 직전 확인 시점엔 안 눌려 있었을 때만 True (1회성, 뗐다 다시 눌러야 재발동).

        2026-09-23: level(그냥 "지금 눌려 있나")로 체크했더니, 에피소드 사이 input()으로 오래
        블로킹하는 동안 버튼이 눌린 채로 남아(또는 SYN_DROPPED로 release 유실) 다음 에피소드가
        시작하자마자 계속 폐기되는 버그가 났다 — "한번 눌리고 계속 눌린 상태로 유지" 리포트.
        엣지 검출로 바꾸면 설령 state가 눌림으로 stuck돼도 "직전에도 True"였을 테니 다시는
        안 걸리고, 실제로 떼었다 다시 누르는 새 press만 감지한다.
        """
        now = bool(self.state.buttons.get(code, False))
        was = self._edge_prev.get(code, False)
        self._edge_prev[code] = now
        return now and not was

    def episode_end_requested(self) -> bool:
        """BTN_THUMB(엄지 버튼) 눌림(엣지) -> "이 에피소드 지금 끝내고 저장" 신호.

        2026-09-23: --episode-seconds를 고정 길이가 아니라 상한(최대 시간)으로 바꾸면서 추가 —
        다 그렸으면 시간 다 찰 때까지 기다릴 필요 없이 버튼으로 바로 다음 에피소드로 넘어간다.
        """
        return self._rising_edge(ecodes.BTN_THUMB)

    def discard_requested(self) -> bool:
        """BTN_THUMB2 눌림(엣지) -> "이 에피소드는 실패, 저장하지 말고 버려라" 신호.

        2026-09-23 추가. record_mujoco.py가 이 신호를 보면 dataset.clear_episode_buffer()로
        지금까지 쌓인 프레임(이미지 포함)을 버리고, 같은 에피소드 번호를 다시 시도한다.
        """
        return self._rising_edge(ecodes.BTN_THUMB2)

    def rotation_rate(
        self, max_angular: float = 1.0, invert_x: bool = False, invert_y: bool = False, invert_z: bool = False
    ) -> tuple[float, float, float]:
        """roll/pitch(햇스위치) + yaw(트위스트 축) -> world-frame 각속도 (wx, wy, wz) [rad/s].

        2026-10-06: roll/pitch를 베이스 버튼(BTN_BASE~4) 대신 **햇스위치**(스틱 위쪽에 달린
        작은 8방향 POV 미니 조이스틱, ABS_HAT0X/ABS_HAT0Y)로 바꿨다 — 버튼 두 개씩 눌러야
        하는 것보다 엄지로 미니 스틱 하나를 꾹 누르는 게 더 자연스럽다는 요청. 햇스위치는
        디지털(-1/0/1)이라 버튼과 마찬가지로 레이트 컨트롤(누르는 동안만)로 쓴다.
        yaw는 2026-09-23(5차)부터 BASE5/BASE6 대신 트위스트(ABS_RZ, 연속값)로 바꿨다 —
        "yaw는 조이스틱 회전으로 해도 될듯"(스틱 손목을 실제로 돌리는 축이라 연속 제어가
        버튼 두 개보다 자연스러움). kinematics.apply_pose_delta와 같은 world-frame 왼쪽곱
        합성 규약을 쓴다 — record_mujoco.py가 Rotation.from_rotvec(w*dt) @ R_cmd 로 적분한다.

        2026-09-23: 실사용 확인 결과 yaw(트위스트) 기본 부호가 반대라 반전. roll/pitch는 아직
        애매할 수 있어 --invert-x/--invert-y로 열어둠(ee_velocity()의 invert_x/y와는 별개 인자) —
        햇스위치로 바꾸면서 부호를 다시 확인 못 했으니(실측 전) 반대로 느껴지면 그걸로 뒤집을 것.
        """
        wx = float(self.state.hat_x) * max_angular  # roll -/+ (햇스위치 좌우)
        wy = float(self.state.hat_y) * max_angular  # pitch -/+ (햇스위치 상하)
        wz = -self.state.twist * max_angular  # 기본 부호 반전 확인됨
        if invert_x:
            wx = -wx
        if invert_y:
            wy = -wy
        if invert_z:
            wz = -wz
        return wx, wy, wz

    def close(self) -> None:
        self.dev.close()


def _live_diagnostic(recalibrate: bool = False) -> None:
    ctl = JoystickEEController(recalibrate=recalibrate)
    print("[joystick] 스틱/슬라이더/버튼을 움직여보세요. Ctrl+C로 종료.\n")
    try:
        while True:
            ctl.poll()
            s = ctl.state
            vx, vy, vz = ctl.ee_velocity()
            pressed = [ecodes.BTN.get(code, str(code)) for code, down in s.buttons.items() if down]
            print(
                f"\rx={s.x:+.2f} y={s.y:+.2f} throttle={s.throttle:+.2f} twist={s.twist:+.2f} "
                f"hat=({s.hat_x:+d},{s.hat_y:+d}) trigger={int(s.trigger)}  |  "
                f"v=({vx:+.3f},{vy:+.3f},{vz:+.3f}) m/s  |  buttons={pressed}" + " " * 20,
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
    p.add_argument(
        "--calibrate", action="store_true",
        help="스로틀 영점만 다시 잡고 저장한 뒤 종료 (라이브 진단 없이)",
    )
    p.add_argument(
        "--recalibrate", action="store_true",
        help="저장된 보정값을 무시하고 다시 물어본 뒤(라이브 진단 모드로) 저장",
    )
    args = p.parse_args()
    if args.list:
        list_joysticks()
    elif args.calibrate:
        ctl = JoystickEEController(recalibrate=True)
        ctl.close()
    else:
        _live_diagnostic(recalibrate=args.recalibrate)
