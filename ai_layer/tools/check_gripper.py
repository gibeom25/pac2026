#!/usr/bin/env python
"""SO-101 leader -> MuJoCo 그리퍼 리매핑 방향/범위 확인용 실시간 모니터.

record_mujoco.py로 전체 녹화를 돌리지 않고, 리더 그리퍼를 손으로 움직이면서
raw(leader.get_action()["gripper.pos"], 0~100) 값과 build_joint_remap()으로 계산된
MuJoCo 목표 각도가 기대한 방향으로 움직이는지만 빠르게 확인한다.

실행 (pac2026 conda 환경):
  conda activate pac2026
  PYTHONPATH=. python ai_layer/tools/check_gripper.py --leader-port /dev/ttyACM0 \
      --leader-id my_awesome_leader_arm

리더 그리퍼를 완전히 닫았다 열었다 반복하면서 화면 출력을 본다:
  - "닫힘"에서 mujoco deg가 ctrlrange 하한(-10 근처)에 붙어야 하고
  - "열림"에서 ctrlrange 상한(100 근처)에 붙어야 한다.
반대로 움직이면 --gripper-invert를 붙여서 다시 실행 (record_mujoco.py도 동일 플래그 사용).
Ctrl+C로 종료.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.tools.record_mujoco import MJCF_PATH, build_joint_remap  # noqa: E402

from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402
from lerobot.teleoperators.so_leader.so_leader import SOLeader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="리더 그리퍼 -> MuJoCo 리매핑 방향 확인용 실시간 모니터.")
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument("--leader-id", default="my_awesome_leader_arm")
    p.add_argument("--gripper-invert", action="store_true", help="record_mujoco.py와 동일한 플래그")
    p.add_argument("--hz", type=float, default=10.0, help="출력 갱신 주기")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    leader_cfg = SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id)
    leader = SOLeader(leader_cfg)
    leader.connect(calibrate=True)
    print(f"[check_gripper] leader connected on {args.leader_port} (id={args.leader_id})")

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    remap = build_joint_remap(leader, model, gripper_invert=args.gripper_invert)
    l_lo, l_hi, m_lo, m_hi = remap["gripper"]
    print(f"[check_gripper] gripper remap: leader[{l_lo:.1f},{l_hi:.1f}] -> mujoco[{m_lo:.1f},{m_hi:.1f}] deg")
    print(f"[check_gripper] gripper_invert={args.gripper_invert}")
    print("[check_gripper] 리더 그리퍼를 완전히 닫았다/열었다 반복해보세요. Ctrl+C로 종료.\n")

    dt = 1.0 / args.hz
    try:
        while True:
            action = leader.get_action()
            raw = action["gripper.pos"]  # 0~100 (RANGE_0_100), 각도 아님
            t = (raw - l_lo) / (l_hi - l_lo)
            mujoco_deg = float(np.clip(m_lo + t * (m_hi - m_lo), min(m_lo, m_hi), max(m_lo, m_hi)))

            if raw < 15:
                state = "닫힘 근처"
            elif raw > 85:
                state = "열림 근처"
            else:
                state = "중간"

            print(f"\rraw(0~100)={raw:6.1f}  mujoco_deg={mujoco_deg:7.2f}  [{state:8s}]", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[check_gripper] 종료.")
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
