"""IsaacLab에 SO-101 USD가 정상적으로 로드되고 물리 스텝이 도는지 확인하는 헤드리스 스모크 테스트.

실행 (IsaacLab 디렉토리에서, pac2026_isaaclab 환경):
  ./isaaclab.sh -p /home/robot/pac2026/ai_layer/sim/smoke_test_so101.py

목적: 새 conda 환경 + Isaac Sim 4.5.0.0 + IsaacLab v2.1.0 + NVIDIA 공식 SO-101 USD 자산
조합이 실제로 동작하는지 최소 확인. BC/RL 학습 환경 자체는 아직 아님.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 USD load smoke test.")
parser.add_argument("--num-steps", type=int, default=60)
parser.add_argument("--sim-device", type=str, default="cuda:0", help="'cpu' to bypass GPU PhysX pipeline")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""이 아래부터는 시뮬레이터 앱이 뜬 뒤에만 import 가능."""

import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# 프로젝트의 SO-101 IsaacLab 자산 설정을 import
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "assets" / "so101_isaac"))
from so101_cfg import SO101_CFG  # noqa: E402


def design_scene() -> Articulation:
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = SO101_CFG.copy()
    robot_cfg.prim_path = "/World/SO101"
    robot = Articulation(cfg=robot_cfg)
    return robot


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.sim_device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.6, 0.6, 0.4], [0.0, 0.0, 0.15])

    robot = design_scene()
    sim.reset()

    print(f"[smoke_test] joint names: {robot.joint_names}")
    print(f"[smoke_test] num bodies: {robot.num_bodies}, num joints: {robot.num_joints}")

    sim_dt = sim.get_physics_dt()
    for step in range(args_cli.num_steps):
        sim.step()
        robot.update(sim_dt)
        if step % 20 == 0:
            q = robot.data.joint_pos[0].tolist()
            print(f"[smoke_test] step={step} joint_pos(rad)={['%.3f' % v for v in q]}")

    print("[smoke_test] SUCCESS: SO-101 USD loaded and stepped without error.")


if __name__ == "__main__":
    main()
    simulation_app.close()
