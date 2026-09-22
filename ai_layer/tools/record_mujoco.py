#!/usr/bin/env python
"""SO-101 leader 실물 티칭 -> MuJoCo 팔로워 시뮬레이션 미러링 -> LeRobotDataset 기록.

실물 팔로워/카메라 없이 리더 암 하나만으로 BC 학습용 데이터셋을 만들기 위한 스크립트.
관절공간(joint-space) 그대로 기록한다 — `train_bc.py`/`SO101BCDataset`이 기대하는 것과 동일한
형식(observation.state/action = JOINT_NAMES 순서, degree)이라 실물 `lerobot-record` 결과물과
호환된다. 이미지는 MuJoCo 손목 카메라(`so101_new_calib_camera.xml`, `assets/so101/README` 참고)로
렌더링한다.

실행 (pac2026 conda 환경, lerobot 설치되어 있음 — leader 통신용):
  conda activate pac2026
  PYTHONPATH=. python ai_layer/tools/record_mujoco.py \
      --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm \
      --repo-id <hf-user>/so101-mujoco-demo --root ./datasets/so101-mujoco-demo \
      --num-episodes 5 --episode-seconds 15

에피소드 사이에 Enter를 누르면 다음 에피소드 녹화를 시작한다 (리더를 시작 자세로 되돌릴 시간을 준다).
Ctrl+C로 중단하면 그때까지 저장된 에피소드는 유지된다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.kinematics import JOINT_NAMES  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # noqa: E402
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402
from lerobot.teleoperators.so_leader.so_leader import SOLeader  # noqa: E402

MJCF_PATH = Path(__file__).resolve().parents[2] / "assets" / "so101" / "so101_new_calib_camera.xml"
CAMERA_NAME = "wrist"
CAMERA_HW = (240, 320, 3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 leader -> MuJoCo follower 미러링 데이터 수집.")
    p.add_argument("--leader-port", default="/dev/ttyACM0")
    p.add_argument("--leader-id", default="my_awesome_leader_arm")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="로컬 저장 경로 (없으면 HF_LEROBOT_HOME/<repo-id>)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--episode-seconds", type=float, default=15.0)
    p.add_argument("--task", default="weld seam following demo (mujoco mirror)")
    return p.parse_args()


def build_dataset(args: argparse.Namespace) -> LeRobotDataset:
    hw_features = {name: float for name in JOINT_NAMES}
    hw_features_with_cam = {**hw_features, "wrist": CAMERA_HW}

    obs_features = hw_to_dataset_features(hw_features_with_cam, "observation", use_video=False)
    action_features = hw_to_dataset_features(hw_features, "action", use_video=False)
    features = {**obs_features, **action_features}

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type="so101_follower_mujoco",
        use_videos=False,
    )


def main() -> None:
    args = parse_args()

    leader_cfg = SOLeaderTeleopConfig(port=args.leader_port, id=args.leader_id)
    leader = SOLeader(leader_cfg)
    leader.connect(calibrate=True)
    print(f"[record] leader connected on {args.leader_port} (id={args.leader_id})")

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=CAMERA_HW[0], width=CAMERA_HW[1])
    mujoco.mj_forward(model, data)

    dataset = build_dataset(args)
    dt = 1.0 / args.fps
    substeps = max(1, int(round(dt / model.opt.timestep)))

    try:
        for ep in range(args.num_episodes):
            input(f"\n[record] 에피소드 {ep + 1}/{args.num_episodes} — 준비되면 Enter (Ctrl+C 종료) ")
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

            t0 = time.perf_counter()
            n_steps = int(args.episode_seconds * args.fps)
            for step in range(n_steps):
                loop_t0 = time.perf_counter()

                action = leader.get_action()  # {"shoulder_pan.pos": deg, ...}
                joint_deg = {name: action[f"{name}.pos"] for name in JOINT_NAMES}

                data.ctrl[:] = np.deg2rad([joint_deg[n] for n in JOINT_NAMES])
                for _ in range(substeps):
                    mujoco.mj_step(model, data)

                state_deg = {name: float(np.rad2deg(data.qpos[i])) for i, name in enumerate(JOINT_NAMES)}

                renderer.update_scene(data, camera=CAMERA_NAME)
                img = renderer.render()

                obs_values = {**state_deg, "wrist": img}
                obs_frame = build_dataset_frame(dataset.features, obs_values, prefix="observation")
                action_frame = build_dataset_frame(dataset.features, joint_deg, prefix="action")

                dataset.add_frame({**obs_frame, **action_frame, "task": args.task})

                if step % args.fps == 0:
                    print(f"  t={step / args.fps:.1f}s state={[round(state_deg[n], 1) for n in JOINT_NAMES]}")

                elapsed = time.perf_counter() - loop_t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            dataset.save_episode()
            print(f"[record] 에피소드 {ep + 1} 저장 완료 ({time.perf_counter() - t0:.1f}s, {n_steps} 프레임)")
    except KeyboardInterrupt:
        print("\n[record] 중단됨 — 이미 저장된 에피소드는 유지됩니다.")
    finally:
        leader.disconnect()

    print(f"[record] 데이터셋: {dataset.root}")


if __name__ == "__main__":
    main()
