#!/usr/bin/env python
"""키보드(터미널 raw 모드) -> EE 속도 명령. 조이스틱 없을 때 쓰는 대체 입력.

2026-10-06(4차, 기범 지적 — "기존 teleop 프로그램들처럼 터미널에서 받으면 되는거 아니야?"):
맞다. 1차(evdev 직접 읽기)와 2차(evdev+grab+ESC비상탈출)는 'input' 그룹 권한이 필요했고
엉뚱한 장치를 잡을 위험이 있었다. 3차는 그걸 피하려고 MuJoCo 뷰어의 GLFW key_callback으로
바꿨는데, 그 결과 뷰어 창에 포커스를 매번 클릭해줘야 하고 `--headless`와는 아예 못 쓰게 됐다
— 불필요하게 복잡했다. teleop_twist_keyboard 같은 표준 CLI teleop 도구들이 쓰는 방식 그대로,
**터미널을 raw 모드로 바꿔서 이 프로세스의 stdin을 직접 읽는다** — 이게 맞는 답이었다:
  - 이 프로세스 본인의 tty를 만지는 것이므로 별도 권한이 전혀 필요 없다(evdev처럼 커널 전역
    장치를 읽는 게 아님).
  - 엉뚱한 장치를 잡을 위험이 없다(장치 탐색 자체가 없음 — 그냥 stdin).
  - 뷰어 창이 없어도(= `--headless`) 그대로 동작한다.
  - 포커스는 "이 명령을 실행한 터미널 창"에 있으면 된다 — 원래 명령어 출력도 그 터미널에서
    보고 있었을 테니 추가로 뭘 더 클릭할 필요가 없다.

터미널 raw 모드는 ICANON(줄 단위 입력 대기)과 ECHO(입력한 글자가 화면에 그대로 찍히는 것)만
끈다 — ISIG는 그대로 둬서 Ctrl+C(SIGINT)는 평소처럼 작동한다("Ctrl+C로 종료" 안내와 호환).
tty.setraw()를 안 쓰고 직접 termios 플래그를 만지는 이유도 이것 — setraw는 ISIG까지 꺼버려서
Ctrl+C가 안 먹히게 된다.

**제약**: 일반 키보드 입력과 마찬가지로 release 통지가 없다(터미널은 "눌림/뗌"이 아니라
"문자가 들어왔다"만 안다). 그래서 GLFW 버전과 똑같은 방식으로 "누르고 있음"을 흉내낸다 — 문자가
들어온 시각을 기록해두고 `_HOLD_TIMEOUT_S` 안에 또 안 들어오면 "뗐다"고 본다(실제 키를 누르고
있으면 OS 키 반복이 그 안에 계속 다시 보내준다). 그리고 실제 tty가 있어야 한다 — 파이프/리다이렉트로
stdin을 돌리면(`echo | python ...`) 생성 시점에 바로 에러를 낸다.

키 배치는 joystick_input.py의 "누르는 동안 레이트" 관례 그대로:
  w/s              +x/-x
  d/a              +y/-y
  r/f              +z/-z
  q/e              roll -/+
  z/x              pitch -/+
  c/v              yaw -/+
  SPACE(누르는 동안) 그리퍼/도구 신호 = 1
  ENTER            에피소드 저장+종료 (BTN_THUMB과 동일)
  BACKSPACE/DEL    에피소드 폐기+재시도 (BTN_THUMB2와 동일)
(대소문자 구분 안 함 — Shift/Caps Lock으로 눌려도 그대로 인식)
"""

from __future__ import annotations

import atexit
import select
import sys
import termios
import threading
import time

# 문자가 이 시간 안에 또 안 들어오면 "뗐다"고 본다. OS 키 반복의 초기 지연(데스크톱 기본값
# 보통 400~600ms)을 덮을 만큼 넉넉하게 잡음 — 너무 짧으면 누르고 있어도 반복 시작 전 짧은
# 공백 동안 "뗐다"로 깜빡여서 움직임이 끊겨 보인다.
_HOLD_TIMEOUT_S = 0.6

_KEY_X_POS = "w"
_KEY_X_NEG = "s"
_KEY_Y_POS = "d"
_KEY_Y_NEG = "a"
_KEY_Z_POS = "r"
_KEY_Z_NEG = "f"
_KEY_ROLL_NEG = "q"
_KEY_ROLL_POS = "e"
_KEY_PITCH_NEG = "z"
_KEY_PITCH_POS = "x"
_KEY_YAW_NEG = "c"
_KEY_YAW_POS = "v"
_KEY_TRIGGER = " "
_KEY_END_EPISODE = ("\r", "\n")  # 터미널/에뮬레이터에 따라 둘 중 하나로 옴
_KEY_DISCARD = ("\x7f", "\x08")  # DEL, BACKSPACE — 터미널에 따라 다름

_HOLD_KEYS = (
    _KEY_X_POS, _KEY_X_NEG, _KEY_Y_POS, _KEY_Y_NEG, _KEY_Z_POS, _KEY_Z_NEG,
    _KEY_ROLL_NEG, _KEY_ROLL_POS, _KEY_PITCH_NEG, _KEY_PITCH_POS, _KEY_YAW_NEG, _KEY_YAW_POS,
    _KEY_TRIGGER,
)


class KeyboardEEController:
    """JoystickEEController와 동일한 공개 인터페이스 — 드롭인 대체용.

    생성 시 이 프로세스의 stdin(tty)을 raw 모드로 바꾸고 백그라운드 스레드로 읽는다 — 별도
    장치나 권한이 필요 없다. `close()`에서 반드시 원래 터미널 설정으로 복원한다(안 하면 터미널이
    "줄바꿈도 에코도 안 되는" 상태로 남는다) — 혹시 close()를 못 부르고 죽는 경우를 대비해
    atexit에도 복원을 걸어둔다.
    """

    def __init__(self):
        if not sys.stdin.isatty():
            raise RuntimeError(
                "키보드 입력은 실제 터미널(tty)에서만 됩니다 — 파이프나 리다이렉트로 stdin을 "
                "바꾼 상태에서는 못 씁니다. 조이스틱을 쓰거나(--input joystick) 일반 터미널에서 "
                "직접 실행하세요."
            )

        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        new_settings = termios.tcgetattr(self._fd)
        new_settings[3] = new_settings[3] & ~(termios.ICANON | termios.ECHO)  # ISIG는 유지(Ctrl+C 보존)
        termios.tcsetattr(self._fd, termios.TCSANOW, new_settings)
        self._restored = False
        atexit.register(self._restore_terminal)

        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}
        self._pending_end = False
        self._pending_discard = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        print(
            "[keyboard] 조이스틱 없을 때 대체 입력 — 이 터미널에 포커스가 있는 동안 키 입력을 "
            "그대로 받는다(별도 권한/장치 불필요). WASD(xy) R/F(z) Q/E(roll) Z/X(pitch) "
            "C/V(yaw) SPACE(트리거) ENTER(저장종료) BACKSPACE(폐기재시도)"
        )

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([self._fd], [], [], 0.1)
            if not ready:
                continue
            try:
                ch = sys.stdin.read(1)
            except Exception:
                continue
            if ch:
                self._handle_char(ch)

    def _handle_char(self, ch: str) -> None:
        ch = ch.lower()
        with self._lock:
            if ch in _HOLD_KEYS:
                self._last_seen[ch] = time.perf_counter()
            elif ch in _KEY_END_EPISODE:
                self._pending_end = True
            elif ch in _KEY_DISCARD:
                self._pending_discard = True

    def _held(self, ch: str) -> bool:
        with self._lock:
            t = self._last_seen.get(ch)
        return t is not None and (time.perf_counter() - t) < _HOLD_TIMEOUT_S

    def _axis(self, neg_ch: str, pos_ch: str) -> float:
        return (1.0 if self._held(pos_ch) else 0.0) - (1.0 if self._held(neg_ch) else 0.0)

    def poll(self) -> None:
        """백그라운드 스레드가 비동기로 바로 읽으므로 폴링이 필요 없다 — 인터페이스 호환용 no-op."""

    def ee_velocity(
        self,
        max_linear: float = 0.05,
        invert_x: bool = False,
        invert_y: bool = False,
        invert_z: bool = False,
    ) -> tuple[float, float, float]:
        vx = self._axis(_KEY_X_NEG, _KEY_X_POS) * max_linear
        # 2026-10-06: D(+y 키)를 누르면 손목 카메라 기준 실제로는 왼쪽으로 가는 걸로 실측
        # 확인됨(카메라가 블록 왼쪽에 달려 있어 "오른쪽" 쪽 부호가 직관과 반대였음) — D/A를
        # 서로 바꿔서 D가 손목 카메라 기준 오른쪽으로 가게 했다.
        vy = self._axis(_KEY_Y_POS, _KEY_Y_NEG) * max_linear
        vz = self._axis(_KEY_Z_NEG, _KEY_Z_POS) * max_linear
        if invert_x:
            vx = -vx
        if invert_y:
            vy = -vy
        if invert_z:
            vz = -vz
        return vx, vy, vz

    def gripper_bit(self) -> float:
        return 1.0 if self._held(_KEY_TRIGGER) else 0.0

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

    def episode_end_requested(self) -> bool:
        with self._lock:
            v, self._pending_end = self._pending_end, False
        return v

    def wait_for_enter(self, prompt: str = "") -> None:
        """record_mujoco.py/check_ee.py의 "준비되면 Enter" 대기에 builtin `input()` 대신 쓴다.

        이 컨트롤러가 생성되는 순간부터 백그라운드 스레드가 같은 stdin fd를 raw 모드로 읽고
        있어서, 그 상태에서 `input()`을 같이 부르면 둘이 같은 바이트를 두고 경쟁해 입력이
        한쪽으로만 가거나 영영 안 돌아오는(행) 문제가 생긴다(raw 모드라 ECHO/캐노니컬 줄 버퍼링도
        없어서 input()이 기대하는 전제 자체가 깨짐). ENTER 감지를 이 컨트롤러 자신의 메커니즘
        (episode_end_requested와 동일한 pending 플래그)으로 통일해서 경쟁을 없앤다 — "준비 확인"과
        "에피소드 종료"는 같은 ENTER 키를 쓰지만 시점이 겹치지 않아(에피소드 진행 루프가 매 프레임
        episode_end_requested()를 이미 소비하므로, 다음 "준비되면 Enter"로 돌아올 때 남아있는
        pending이 없다) 플래그를 공유해도 안전하다. **여기서 미리 한 번 비우면 안 된다** — 이
        호출 전에(= 프롬프트가 화면에 뜨기도 전에) 사용자가 성급하게 누른 진짜 Enter까지 같이
        버려져서 이후에 또 눌러야 하는 것처럼 보이는 버그가 된다(실측으로 확인).
        """
        print(prompt, end="", flush=True)
        while not self.episode_end_requested():
            time.sleep(0.03)
        print()

    def discard_requested(self) -> bool:
        with self._lock:
            v, self._pending_discard = self._pending_discard, False
        return v

    def _restore_terminal(self) -> None:
        if self._restored:
            return
        self._restored = True
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._restore_terminal()


def _live_diagnostic() -> None:
    """단독 진단 모드 — 뷰어 없이 터미널에서 바로 키 상태를 실시간 출력한다.
    PYTHONPATH=. python ai_layer/tools/keyboard_input.py
    """
    ctl = KeyboardEEController()
    print("[keyboard] 키를 눌러보세요. Ctrl+C로 종료.\n")
    try:
        while True:
            vx, vy, vz = ctl.ee_velocity()
            wx, wy, wz = ctl.rotation_rate()
            print(
                f"\rv=({vx:+.3f},{vy:+.3f},{vz:+.3f}) w=({wx:+.2f},{wy:+.2f},{wz:+.2f}) "
                f"trigger={int(ctl.gripper_bit())}  end={ctl.episode_end_requested()} "
                f"discard={ctl.discard_requested()}" + " " * 10,
                end="",
                flush=True,
            )
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        ctl.close()
    print("\n[keyboard] 종료.")


if __name__ == "__main__":
    _live_diagnostic()
