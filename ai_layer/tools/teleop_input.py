"""조이스틱/키보드 중 뭘 쓸지 고르는 공통 진입점 — record_mujoco.py/check_ee.py/record_gui.py가
공유한다. `JoystickEEController`/`KeyboardEEController`는 같은 공개 인터페이스(poll/ee_velocity/
gripper_bit/rotation_rate/episode_end_requested/discard_requested/close)를 구현하므로 호출 측은
어느 쪽이 돌아왔는지 신경 쓸 필요 없다.

2026-10-06: "조이스틱 없을 때를 대비해서" 추가 — `--input auto`(기본)면 조이스틱을 먼저 찾아보고
없으면 키보드로 자동 전환, `--input keyboard`/`--input joystick`으로 강제 지정도 가능.

2026-10-06(3차→4차): evdev로 raw 키보드 장치를 grab하는 설계, 그 다음 MuJoCo 뷰어의 GLFW
`key_callback`을 쓰는 설계를 거쳐, 지금은 **터미널 raw 모드로 stdin을 직접 읽는 방식**으로
정착했다(자세한 이유는 keyboard_input.py 모듈 docstring 참고 — teleop_twist_keyboard 같은
표준 CLI teleop 도구들과 같은 방식). 덕분에 별도 권한도, 뷰어 창도 필요 없고 `--headless`와도
그냥 같이 쓸 수 있다 — "뷰어 창에 포커스를 줘야 한다"는 제약은 더 이상 없다(터미널에 포커스가
있으면 됨, 즉 명령어를 실행한 그 터미널).

`resolve_input_mode()`를 따로 둔 이유: 조이스틱 보정은 터미널 입력을 기다리는 블로킹 프롬프트가
있을 수 있어서 호출 측이 "창을 먼저 띄우고 그 안에서 보정하라"는 안내를 주는 기존 흐름이 있는데,
키보드 쪽은 생성이 즉시 끝나(블로킹 프롬프트 없음) 그 춤이 필요 없다. 호출 측이 실제 장치를
만들기 전에 "joystick"/"keyboard" 중 뭐가 될지 미리 알아야 그 분기를 탈 수 있어서 분리했다.
"""

from __future__ import annotations

import argparse
import glob


def diagnose_no_devices() -> str | None:
    """evdev로 열리는 입력 장치가 하나도 없을 때(find_joystick이 겪는 증상) 원인이 "장치가
    아예 없음"인지 "권한 문제"인지 구분해서 메시지를 만든다. 문제가 명확한 권한 케이스가
    아니면 None.

    2026-10-06: 실제로 이 증상을 겪었다 — evdev.list_devices()가 빈 리스트를 돌려줬는데,
    `/dev/input/event*` 자체는 존재했다(ls로 확인됨). evdev가 각 장치를 열어보고 실패하면
    조용히 건너뛰므로(예외를 안 띄움), list_devices()만 봐서는 "장치가 없다"와 "권한이 없어서
    하나도 못 열었다"를 구분할 수 없다 — 글로 직접 비교해서 알려준다.

    (키보드는 더 이상 evdev를 안 쓰므로 이 함수는 조이스틱 전용이 됐다.)
    """
    import evdev

    raw_paths = glob.glob("/dev/input/event*")
    if not raw_paths:
        return None  # 장치 파일 자체가 없음 — 권한 문제 아님(진짜로 아무것도 안 꽂혀 있음)
    if evdev.list_devices():
        return None  # 최소 하나는 열림 — 권한 문제 아님
    return (
        f"/dev/input에 장치 파일이 {len(raw_paths)}개 있지만(ls /dev/input/event*) evdev로는 "
        "하나도 못 열었습니다 — 거의 확실히 'input' 그룹 권한 문제입니다. 고치는 법:\n"
        "    sudo usermod -aG input $USER\n"
        "    그 다음 로그아웃 후 다시 로그인(또는 재부팅) — 그룹 변경은 새 로그인 세션부터 "
        "적용되고, 같은 터미널에서 su/newgrp로도 당장 적용 가능(newgrp input).\n"
        "    확인: groups 에 input이 보이는지, 그 다음 --list를 다시 실행."
    )


def add_input_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--input", choices=["auto", "joystick", "keyboard"], default="auto",
        help="조작 입력 장치. auto(기본)는 조이스틱을 찾아보고 없으면 키보드로 자동 전환. "
             "키보드는 이 명령을 실행한 터미널에 포커스가 있어야 동작(별도 권한 불필요).",
    )


def resolve_input_mode(input_mode: str = "auto") -> str:
    """"auto"를 실제로 "joystick"/"keyboard" 중 뭐가 될지로 미리 정한다 — 장치를 만들지는 않고
    find_joystick()만 시도해본다. 호출 측이 (조이스틱이면 보정 프롬프트가 있을 수 있으니 창을
    먼저 띄워야 한다는) 분기를 장치 생성 전에 타야 할 때 쓴다."""
    if input_mode in ("joystick", "keyboard"):
        return input_mode

    from ai_layer.tools.joystick_input import find_joystick

    try:
        find_joystick()
        return "joystick"
    except RuntimeError:
        print("[teleop] 조이스틱을 못 찾아서 키보드 입력으로 대체합니다 (--input keyboard로 강제 가능).")
        return "keyboard"


def build_ee_controller(input_mode: str = "auto", recalibrate: bool = False):
    from ai_layer.tools.joystick_input import JoystickEEController
    from ai_layer.tools.keyboard_input import KeyboardEEController

    resolved = resolve_input_mode(input_mode)
    if resolved == "keyboard":
        return KeyboardEEController()
    return JoystickEEController(recalibrate=recalibrate)
