#!/usr/bin/env python
"""SO-101 leader -> MuJoCo 그리퍼 리매핑 방향/범위 확인용 실시간 뷰어.

record_mujoco.py로 전체 녹화를 돌리지 않고, 리더를 손으로 움직이면서 MuJoCo 팔로워가
실시간으로 따라 움직이는 걸 뷰어 창으로 직접 보고, 그리퍼 raw(0~100)/리매핑된 각도를
콘솔에 같이 출력한다. 기본적으로 뷰어 창을 띄운다 (다른 시뮬레이션 도구와 동일한 기본값).

실행 (pac2026 conda 환경):
  conda activate pac2026
  PYTHONPATH=. python ai_layer/tools/check_gripper.py --leader-port /dev/ttyACM0 \
      --leader-id my_awesome_leader_arm

리더 그리퍼를 완전히 닫았다 열었다 반복하면서 뷰어 속 그리퍼 + 콘솔 출력을 본다:
  - "닫힘"에서 MuJoCo 그리퍼가 실제로 완전히 다물어지고 mujoco_deg가 ctrlrange 하한(-10 근처)에 붙어야 하고
  - "열림"에서 ctrlrange 상한(100 근처)에 붙어야 한다.
반대로 움직이면 --gripper-invert를 붙여서 다시 실행 (record_mujoco.py도 동일 플래그 사용).
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
from ai_layer.tools.record_mujoco import MJCF_PATH, _remap_deg, build_joint_remap  # noqa: E402

from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402
from lerobot.teleoperators.so_leader.so_leader import SOLeader  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="리더 -> MuJoCo 리매핑 확인용 실시간 뷰어 (그리퍼 방향 위주).")
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument("--leader-id", default="my_awesome_leader_arm")
    p.add_argument("--gripper-invert", action="store_true", help="record_mujoco.py와 동일한 플래그")
    p.add_argument("--hz", type=float, default=30.0, help="제어/출력 주기")
    p.add_argument("--headless", action="store_true", help="뷰어 창 없이 콘솔 출력만 (기본: 창 띄움)")
    return p.parse_args()


def _run(leader, model, data, viewer, remap: dict, hz: float) -> None:
    dt = 1.0 / hz
    substeps = max(1, int(round(dt / model.opt.timestep)))
    l_lo, l_hi, m_lo, m_hi = remap["gripper"]
    print_every = max(1, int(hz // 5))  # 콘솔은 초당 약 5회만 찍는다 (뷰어는 매 스텝 갱신)

    step = 0
    while True:
        loop_t0 = time.perf_counter()

        action = leader.get_action()
        leader_deg = {name: action[f"{name}.pos"] for name in JOINT_NAMES}
        joint_deg = _remap_deg(leader_deg, remap)

        data.ctrl[:] = np.deg2rad([joint_deg[n] for n in JOINT_NAMES])
        for _ in range(substeps):
            mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                print("\n[check_gripper] 뷰어 창이 닫혀서 종료합니다.")
                return

        if step % print_every == 0:
            raw = leader_deg["gripper"]
            state = "닫힘 근처" if raw < 15 else "열림 근처" if raw > 85 else "중간"
            print(
                f"\rraw(0~100)={raw:6.1f}  mujoco_target={joint_deg['gripper']:7.2f}  "
                f"range[{m_lo:.1f},{m_hi:.1f}]  [{state:8s}]",
                end="",
                flush=True,
            )
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

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    remap = build_joint_remap(leader, model, gripper_invert=args.gripper_invert)
    l_lo, l_hi, m_lo, m_hi = remap["gripper"]
    print(f"[check_gripper] gripper remap: leader[{l_lo:.1f},{l_hi:.1f}] -> mujoco[{m_lo:.1f},{m_hi:.1f}] deg")
    print(f"[check_gripper] gripper_invert={args.gripper_invert}")
    print("[check_gripper] 리더를 움직여보세요 (그리퍼는 완전히 닫았다/열었다 반복). Ctrl+C로 종료.\n")

    try:
        if args.headless:
            _run(leader, model, data, viewer=None, remap=remap, hz=args.hz)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                _run(leader, model, data, viewer, remap, args.hz)
    except KeyboardInterrupt:
        print("\n[check_gripper] 종료.")
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()
