"""SO-101 leader 실물 티칭 -> IsaacLab 시뮬레이션 미러링 (데이터 생성용).

원래 설계(ai_layer_welding_trajectory_design.md 메모리 / docs 3.2절)의 첫 단계:
"SOArm-leader로 EEF 궤적 데이터를 모읍니다 ... IsaacLab에서 볼 수 있게 하여 궤적을 생성합니다"를
구현한 것. 실물 리더 암의 관절각을 매 프레임 읽어 시뮬레이션 SO-101에 그대로 반영(미러링)하고,
선택적으로 EEF pose 궤적을 파일로 기록한다.

실행 (pac2026_isaaclab 환경 — lerobot도 이미 설치돼 있음, train_rl.py 참고):
  ./isaaclab.sh -p /home/robot/pac2026/ai_layer/sim/teleop_leader_to_isaac.py --leader-port /dev/ttyACM0
  (헤드리스로 돌리려면 --headless 추가, GUI로 실시간으로 보려면 빼고 실행)

⚠️ 이 세션(Bash 도구 샌드박스)에서는 실행 검증 불가 — 사용자 터미널에서 실행할 것.
⚠️ 하드웨어 연결 자체가 아직 안 되는 상태(scan_so101_leader.py로 확인)라면 이 스크립트도
   당연히 동작하지 않는다 — 먼저 하드웨어 통신부터 해결할 것.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 leader -> IsaacLab teleoperation mirror.")
parser.add_argument("--leader-port", default="/dev/ttyACM0")
parser.add_argument("--num-steps", type=int, default=100_000)
parser.add_argument("--record-out", default=None, help="지정하면 매 프레임 (t, joint_deg[6], eef_pose[6])를 .jsonl로 기록")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""이 아래부터는 시뮬레이터 앱이 뜬 뒤에만 import 가능."""

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets" / "so101_isaac"))
from so101_cfg import SO101_CFG  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_layer.kinematics import (  # noqa: E402
    ISAAC_JOINT_NAMES,
    JOINT_NAMES,
    build_kinematics,
    pose_to_xyzrotvec,
)

from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402
from lerobot.teleoperators.so_leader.so_leader import SOLeader  # noqa: E402


def design_scene() -> Articulation:
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )
    robot_cfg = SO101_CFG.copy()
    robot_cfg.prim_path = "/World/SO101"
    return Articulation(robot_cfg)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.6, 0.4], [0.0, 0.0, 0.15])

    robot = design_scene()
    sim.reset()

    leader_cfg = SOLeaderTeleopConfig(port=args_cli.leader_port, id="leader")
    leader = SOLeader(leader_cfg)
    leader.connect(calibrate=False)
    print(f"[teleop] leader connected on {args_cli.leader_port}")

    kin = build_kinematics()

    joint_idx = [robot.joint_names.index(n) for n in ISAAC_JOINT_NAMES]

    record_file = open(args_cli.record_out, "w") if args_cli.record_out else None

    sim_dt = sim.get_physics_dt()
    try:
        for step in range(args_cli.num_steps):
            if not simulation_app.is_running():
                break

            action = leader.get_action()  # {"shoulder_pan.pos": deg, ...} — JOINT_NAMES 순서
            joint_deg = np.array([action[f"{n}.pos"] for n in JOINT_NAMES], dtype=float)

            target = torch.zeros(1, robot.num_joints)
            target[0, joint_idx] = torch.deg2rad(torch.tensor(joint_deg, dtype=torch.float32))
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()

            sim.step()
            robot.update(sim_dt)

            if record_file is not None:
                eef_pose = pose_to_xyzrotvec(kin.forward_kinematics(joint_deg))
                record_file.write(
                    json.dumps({"t": time.time(), "joint_deg": joint_deg.tolist(), "eef_pose": eef_pose.tolist()})
                    + "\n"
                )

            if step % 60 == 0:
                print(f"[teleop] step={step} joint_deg={joint_deg.round(1).tolist()}")
    finally:
        leader.disconnect()
        if record_file is not None:
            record_file.close()
            print(f"[teleop] recorded to {args_cli.record_out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
