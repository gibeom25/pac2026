"""조이스틱/키보드 중 뭘 쓸지 고르는 공통 진입점 — record_mujoco.py/check_ee.py/record_gui.py가
공유한다. `JoystickEEController`/`KeyboardEEController`는 같은 공개 인터페이스(poll/ee_velocity/
gripper_bit/rotation_rate/episode_end_requested/discard_requested/close)를 구현하므로 호출 측은
어느 쪽이 돌아왔는지 신경 쓸 필요 없다.

2026-10-06: "조이스틱 없을 때를 대비해서" 추가 — `--input auto`(기본)면 조이스틱을 먼저 찾아보고
없으면 키보드로 자동 전환, `--input keyboard`/`--input joystick`으로 강제 지정도 가능.
"""

from __future__ import annotations

import argparse


def add_input_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--input", choices=["auto", "joystick", "keyboard"], default="auto",
        help="조작 입력 장치. auto(기본)는 조이스틱을 찾아보고 없으면 키보드로 자동 전환.",
    )


def build_ee_controller(input_mode: str = "auto", recalibrate: bool = False):
    from ai_layer.tools.joystick_input import JoystickEEController, find_joystick
    from ai_layer.tools.keyboard_input import KeyboardEEController

    if input_mode == "keyboard":
        return KeyboardEEController()
    if input_mode == "joystick":
        return JoystickEEController(recalibrate=recalibrate)

    # auto
    try:
        find_joystick()
    except RuntimeError:
        print("[teleop] 조이스틱을 못 찾아서 키보드 입력으로 대체합니다 (--input keyboard로 강제 가능).")
        return KeyboardEEController()
    return JoystickEEController(recalibrate=recalibrate)
