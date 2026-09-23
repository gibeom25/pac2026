#!/usr/bin/env python
"""SO-101 leader -> MuJoCo 실시간 뷰어. 그리퍼는 이진(0/1) 신호 + LED/비드 테스트용.

record_mujoco.py로 전체 녹화를 돌리지 않고, 팔 5관절은 리더를 따라 실시간으로 움직이는 걸
뷰어 창으로 보고, 그리퍼는 더 이상 구동하지 않는 대신(설계문서 2절 gripper_signal[0/1]) 리더
그리퍼 raw(0~100)를 임계값으로 이진화해 MuJoCo에 붙인 LED(tool_led, so101_new_calib_camera.xml)를
켜고 끄고, 신호 on인 동안 그리스/실리콘 비드처럼 자국도 남긴다(record_mujoco.py의 _draw_bead_trail).
기본적으로 뷰어 창을 띄운다 (다른 시뮬레이션 도구와 동일한 기본값).

실행 (pac2026 conda 환경):
  conda activate pac2026
  PYTHONPATH=. python ai_layer/tools/check_gripper.py --leader-port /dev/ttyACM0 \
      --leader-id my_awesome_leader_arm --scene dashed

리더 그리퍼를 반 이상 닫으면 LED가 켜지고(bit=1) 비드가 쌓이기 시작, 열면 꺼진다(bit=0) — 뷰어 속
LED 색/비드 자국과 콘솔 출력을 같이 본다. 반대로 켜지길 원하면 --gripper-invert.
Ctrl+C 또는 뷰어 창 닫기로 종료.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.kinematics import JOINT_NAMES  # noqa: E402
from ai_layer.tools.record_mujoco import (  # noqa: E402
    ARM_JOINT_NAMES,
    LED_GEOM_NAME,
    LED_OFF_RGBA,
    LED_ON_RGBA,
    SCENE_VARIANTS,
    _draw_bead_trail,
    _mjcf_path,
    _remap_deg,
    build_joint_remap,
    gripper_bit,
)

from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402
from lerobot.teleoperators.so_leader.so_leader import SOLeader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="리더 -> MuJoCo 실시간 뷰어 (그리퍼는 이진 LED 테스트).")
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument("--leader-id", default="my_awesome_leader_arm")
    p.add_argument("--gripper-invert", action="store_true", help="record_mujoco.py와 동일한 플래그")
    p.add_argument("--scene", choices=SCENE_VARIANTS, default="curve", help="A4 용접선 형태")
    p.add_argument("--hz", type=float, default=30.0, help="제어/출력 주기")
    p.add_argument("--headless", action="store_true", help="뷰어 창 없이 콘솔 출력만 (기본: 창 띄움)")
    return p.parse_args()


def _run(leader, model, data, viewer, remap: dict, gripper_invert: bool, hz: float) -> None:
    dt = 1.0 / hz
    substeps = max(1, int(round(dt / model.opt.timestep)))
    gripper_idx = JOINT_NAMES.index("gripper")
    gripper_fixed_rad = float(model.actuator_ctrlrange[gripper_idx][0])  # ctrlrange 하한 = 닫힘
    led_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LED_GEOM_NAME)
    print_every = max(1, int(hz // 5))
    bead_points: list[np.ndarray] = []

    step = 0
    while True:
        loop_t0 = time.perf_counter()

        action = leader.get_action()
        leader_deg = {name: action[f"{name}.pos"] for name in ARM_JOINT_NAMES}
        joint_deg = _remap_deg(leader_deg, remap)

        raw = action["gripper.pos"]
        bit = gripper_bit(raw, invert=gripper_invert)
        joint_deg["gripper"] = float(np.rad2deg(gripper_fixed_rad))

        data.ctrl[:] = np.deg2rad([joint_deg[n] for n in JOINT_NAMES])
        model.geom_rgba[led_gid] = LED_ON_RGBA if bit else LED_OFF_RGBA
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        if bit and step % 2 == 0:
            bead_points.append(data.geom_xpos[led_gid].copy())

        if viewer is not None:
            viewer.user_scn.ngeom = 0
            _draw_bead_trail(viewer.user_scn, bead_points)
            viewer.sync()
            if not viewer.is_running():
                print("\n[check_gripper] 뷰어 창이 닫혀서 종료합니다.")
                return

        if step % print_every == 0:
            led = "ON " if bit else "OFF"
            print(f"\rraw(0~100)={raw:6.1f}  bit={bit:.0f}  LED={led}", end="", flush=True)
        step += 1

        elapsed = time.perf_counter() - loop_t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


def main() -> None:
    args = parse_args()

    leader_cfg = SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id)
    leader = SOLeader(leader_cfg)
    leader.connect(calibrate=True)
    print(f"[check_gripper] leader connected on {args.leader_port} (id={args.leader_id})")

    mjcf_path = _mjcf_path(args.scene)
    print(f"[check_gripper] scene: {args.scene} ({mjcf_path.name})")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    remap = build_joint_remap(leader, model)
    print("[check_gripper] 그리퍼: 조 구동 안 함, raw<50(닫힘) -> LED 켜짐(bit=1)" + (" [반전]" if args.gripper_invert else ""))
    print("[check_gripper] 리더를 움직여보세요 (그리퍼는 반 이상 닫았다/열었다 반복). Ctrl+C로 종료.\n")

    try:
        if args.headless:
            _run(leader, model, data, viewer=None, remap=remap, gripper_invert=args.gripper_invert, hz=args.hz)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                _run(leader, model, data, viewer, remap, args.gripper_invert, args.hz)
    except KeyboardInterrupt:
        print("\n[check_gripper] 종료.")
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
