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
