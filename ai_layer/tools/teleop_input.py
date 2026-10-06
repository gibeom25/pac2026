"""조이스틱/키보드 중 뭘 쓸지 고르는 공통 진입점 — record_mujoco.py/check_ee.py/record_gui.py가
공유한다. `JoystickEEController`/`KeyboardEEController`는 같은 공개 인터페이스(poll/ee_velocity/
gripper_bit/rotation_rate/episode_end_requested/discard_requested/close)를 구현하므로 호출 측은
어느 쪽이 돌아왔는지 신경 쓸 필요 없다.

2026-10-06: "조이스틱 없을 때를 대비해서" 추가 — `--input auto`(기본)면 조이스틱을 먼저 찾아보고
없으면 키보드로 자동 전환, `--input keyboard`/`--input joystick`으로 강제 지정도 가능.

2026-10-06(2차, 기범 지적 — "노트북 키보드인데"): 외장 키보드를 조종용으로 따로 둘 수 없는
노트북에서는, grab 안 한 키보드 입력(WASD/ENTER 등)이 지금 포커스된 다른 창(터미널 등)에도
그대로 들어간다 — ENTER가 터미널에 반쯤 쳐둔 명령을 실행시킬 수도 있어 위험. `--grab-keyboard`를
주면 `KeyboardEEController`가 그 키보드를 커널 레벨로 독점해서 막는다(기본은 안전 우선으로 off).
독점 중 먹통되면 ESC로 즉시 풀고 빠져나올 수 있다(keyboard_input.KeyboardGrabReleased).
"""

from __future__ import annotations

import argparse
import glob


def diagnose_no_devices() -> str | None:
    """evdev로 열리는 입력 장치가 하나도 없을 때(find_keyboard/find_joystick 둘 다 겪는 증상)
    원인이 "장치가 아예 없음"인지 "권한 문제"인지 구분해서 메시지를 만든다. 문제가 명확한
    권한 케이스가 아니면 None.

    2026-10-06: 실제로 이 증상을 두 번 겪었다 — 조이스틱/키보드 둘 다 evdev.list_devices()가
    빈 리스트를 돌려줬는데, `/dev/input/event*` 자체는 존재했다(ls로 확인됨). evdev가 각
    장치를 열어보고 실패하면 조용히 건너뛰므로(예외를 안 띄움), list_devices()만 봐서는
    "장치가 없다"와 "권한이 없어서 하나도 못 열었다"를 구분할 수 없다 — 글로 직접 비교해서
    알려준다.
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
        help="조작 입력 장치. auto(기본)는 조이스틱을 찾아보고 없으면 키보드로 자동 전환.",
    )
    p.add_argument(
        "--grab-keyboard", action="store_true",
        help="키보드 입력일 때, 이 키보드를 독점(다른 창엔 입력 안 감) — 노트북처럼 조종용을 "
             "따로 못 둘 때 특히 권장. 먹통되면 ESC로 즉시 해제됨.",
    )


def build_ee_controller(input_mode: str = "auto", recalibrate: bool = False, grab_keyboard: bool = False):
    from ai_layer.tools.joystick_input import JoystickEEController, find_joystick
    from ai_layer.tools.keyboard_input import KeyboardEEController

    if input_mode == "keyboard":
        return KeyboardEEController(grab=grab_keyboard)
    if input_mode == "joystick":
        return JoystickEEController(recalibrate=recalibrate)

    # auto
    try:
        find_joystick()
    except RuntimeError:
        print("[teleop] 조이스틱을 못 찾아서 키보드 입력으로 대체합니다 (--input keyboard로 강제 가능).")
        return KeyboardEEController(grab=grab_keyboard)
    return JoystickEEController(recalibrate=recalibrate)
