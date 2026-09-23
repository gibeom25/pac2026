#!/usr/bin/env python
"""SO-101 leader 실물 티칭 -> MuJoCo 팔로워 시뮬레이션 미러링 -> LeRobotDataset 기록.

실물 팔로워/카메라 없이 리더 암 하나만으로 BC 학습용 데이터셋을 만들기 위한 스크립트.
관절공간(joint-space) 그대로 기록한다 — `train_bc.py`/`SO101BCDataset`이 기대하는 것과 동일한
형식(observation.state/action = JOINT_NAMES 순서, degree)이라 실물 `lerobot-record` 결과물과
호환된다. 단, 그리퍼 채널은 예외로 항상 0.0/1.0 이진값이다 (설계문서 2절 gripper_signal[0/1] —
그리퍼 조는 구동하지 않고 고정, MuJoCo에 붙인 LED(tool_led)로 뷰어에서 확인 가능).
이미지는 MuJoCo 손목 카메라(`so101_new_calib_camera.xml`, `assets/so101/README` 참고)로
렌더링한다. 씬은 `scene_a4.xml`(위 로봇 파일 + 바닥 + 용접선 그려진 A4 용지, textures/a4_weld_seam.png)을 쓴다.

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
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.kinematics import JOINT_NAMES  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # noqa: E402
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402
from lerobot.teleoperators.so_leader.so_leader import SOLeader  # noqa: E402

MJCF_PATH = Path(__file__).resolve().parents[2] / "assets" / "so101" / "scene_a4.xml"
CAMERA_NAME = "wrist"
CAMERA_HW = (240, 320, 3)

# 2026-09-23: 그리퍼(움직이는 조)는 더 이상 구동하지 않는다 — 설계문서 2절의 gripper_signal[0/1]을
# 그대로 이진 트리거로 기록한다 (실제로는 그리퍼가 아니라 펌프/도구 on-off일 가능성이 큼, docs 3.4절
# "그리퍼/펌프 신호" 참고). MJCF에 5cm 고정 막대(tool_rod) + LED(tool_led)를 붙여 뷰어에서 0/1을
# 눈으로 확인할 수 있게 했다 (so101_new_calib_camera.xml).
GRIPPER_RAW_THRESHOLD = 50.0  # leader RANGE_0_100 값 기준 이진화 임계값
LED_GEOM_NAME = "tool_led"
LED_ON_RGBA = (0.15, 0.95, 0.15, 1.0)
LED_OFF_RGBA = (0.3, 0.05, 0.05, 1.0)


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
    p.add_argument("--headless", action="store_true", help="라이브 뷰어 창 없이 실행 (기본: 창 띄움)")
    p.add_argument(
        "--gripper-invert",
        action="store_true",
        help="leader raw>=50을 0으로, <50을 1로 뒤집는다 (기본: raw>=50 -> 1/켜짐, <50 -> 0/꺼짐)",
    )
    return p.parse_args()


def _dataset_root(args: argparse.Namespace) -> Path:
    if args.root is not None:
        return Path(args.root)
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / args.repo_id


def _existing_episode_count(root: Path) -> int:
    """meta/info.json만 로컬에서 직접 읽어 에피소드 수를 확인 (LeRobotDataset을 열지 않음 —
    에피소드 0개인 데이터셋을 열면 tasks.parquet가 없어 lerobot이 HF Hub 조회를 시도하다 실패한다)."""
    import json

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return 0
    return json.loads(info_path.read_text()).get("total_episodes", 0)


def build_dataset(args: argparse.Namespace) -> LeRobotDataset:
    hw_features = {name: float for name in JOINT_NAMES}
    hw_features_with_cam = {**hw_features, "wrist": CAMERA_HW}

    obs_features = hw_to_dataset_features(hw_features_with_cam, "observation", use_video=False)
    action_features = hw_to_dataset_features(hw_features, "action", use_video=False)
    features = {**obs_features, **action_features}

    root = _dataset_root(args)
    if root.exists():
        n = _existing_episode_count(root)
        if n == 0:
            print(f"[record] 기존 데이터셋이 비어있습니다({root}, 에피소드 0개) — 지우고 새로 시작합니다.")
            import shutil

            shutil.rmtree(root)
        else:
            while True:
                reply = (
                    input(
                        f"[record] 기존 데이터셋 발견: {root} (에피소드 {n}개)\n"
                        f"         [o]verwrite 덮어쓰기 / [r]esume 이어서 기록 / [c]ancel 취소 ? "
                    )
                    .strip()
                    .lower()
                )
                if reply in ("o", "overwrite"):
                    import shutil

                    shutil.rmtree(root)
                    break
                elif reply in ("r", "resume"):
                    print(f"[record] 이어서 기록합니다 (기존 {n}개 에피소드 뒤에 추가).")
                    return LeRobotDataset(repo_id=args.repo_id, root=args.root)
                elif reply in ("c", "cancel"):
                    print("[record] 취소했습니다.")
                    sys.exit(1)
                else:
                    print("  o/r/c 중 하나를 입력해주세요.")

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type="so101_follower_mujoco",
        use_videos=False,
    )


ARM_JOINT_NAMES = [n for n in JOINT_NAMES if n != "gripper"]


def build_joint_remap(leader, model) -> dict[str, tuple[float, float, float, float]]:
    """leader 각도 범위 -> MuJoCo 관절(ctrlrange) 각도 범위 선형 리매핑 테이블 (팔 5관절만).

    lerobot의 DEGREES 정규화 공식(``degrees = (raw - mid) * 360 / max_res``,
    FeetechMotorsBus._normalize)으로 leader의 실제 각도 범위를 구해 MuJoCo 쪽 범위와 짝짓는다.

    그리퍼는 여기 포함하지 않는다 — lerobot `SOLeader`(so_leader.py)에서 ``use_degrees``와
    무관하게 항상 ``MotorNormMode.RANGE_0_100``으로 고정돼 있어 애초에 각도가 아니고
    (leader.get_action()["gripper.pos"]는 0~100 값), 게다가 이제는 그리퍼 조를 구동하지도
    않으므로(위 GRIPPER_RAW_THRESHOLD 참고) 각도 리매핑 자체가 필요 없다.
    """
    remap: dict[str, tuple[float, float, float, float]] = {}
    for i, name in enumerate(JOINT_NAMES):
        if name == "gripper":
            continue
        m_lo, m_hi = (float(np.rad2deg(v)) for v in model.actuator_ctrlrange[i])
        cal = leader.bus.calibration[name]
        model_name = leader.bus.motors[name].model
        max_res = leader.bus.model_resolution_table[model_name] - 1
        mid = (cal.range_min + cal.range_max) / 2
        l_lo = (cal.range_min - mid) * 360 / max_res
        l_hi = (cal.range_max - mid) * 360 / max_res
        remap[name] = (l_lo, l_hi, m_lo, m_hi)
    return remap


def _remap_deg(leader_deg: dict[str, float], remap: dict[str, tuple[float, float, float, float]]) -> dict[str, float]:
    """leader 각도(deg) -> MuJoCo 관절 각도(deg) 선형 변환 (범위 밖은 clip). 팔 5관절 전용."""
    out = {}
    for name, val in leader_deg.items():
        l_lo, l_hi, m_lo, m_hi = remap[name]
        t = (val - l_lo) / (l_hi - l_lo)
        out[name] = float(np.clip(m_lo + t * (m_hi - m_lo), m_lo, m_hi))
    return out


def gripper_bit(raw: float, invert: bool = False) -> float:
    """leader raw gripper.pos(0~100, RANGE_0_100) -> 이진 신호(0.0/1.0). raw>=임계값 -> 1(켜짐)."""
    on = raw >= GRIPPER_RAW_THRESHOLD
    if invert:
        on = not on
    return 1.0 if on else 0.0


def _run(
    args: argparse.Namespace,
    leader,
    model,
    data,
    renderer,
    dataset,
    viewer,
    joint_remap: dict[str, tuple[float, float, float, float]],
) -> None:
    dt = 1.0 / args.fps
    substeps = max(1, int(round(dt / model.opt.timestep)))

    gripper_idx = JOINT_NAMES.index("gripper")
    gripper_fixed_rad = float(np.mean(model.actuator_ctrlrange[gripper_idx]))  # 조는 항상 이 값 고정, 구동 안 함
    led_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, LED_GEOM_NAME)
    print(f"[record] gripper: 조 구동 안 함 (고정 {np.rad2deg(gripper_fixed_rad):.1f}deg), 신호는 0/1 이진 (LED로 표시)")

    for ep in range(args.num_episodes):
        input(f"\n[record] 에피소드 {ep + 1}/{args.num_episodes} — 준비되면 Enter (Ctrl+C 종료) ")
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        t0 = time.perf_counter()
        n_steps = int(args.episode_seconds * args.fps)
        for step in range(n_steps):
            loop_t0 = time.perf_counter()

            action = leader.get_action()  # {"shoulder_pan.pos": deg, ..., "gripper.pos": 0~100}
            leader_deg = {name: action[f"{name}.pos"] for name in ARM_JOINT_NAMES}
            joint_deg = _remap_deg(leader_deg, joint_remap)  # -> MuJoCo 관절 범위로 리매핑 (팔 5관절만)

            gripper_raw = action["gripper.pos"]
            bit = gripper_bit(gripper_raw, invert=args.gripper_invert)
            joint_deg["gripper"] = float(np.rad2deg(gripper_fixed_rad))

            data.ctrl[:] = np.deg2rad([joint_deg[n] for n in JOINT_NAMES])
            model.geom_rgba[led_gid] = LED_ON_RGBA if bit else LED_OFF_RGBA
            for _ in range(substeps):
                mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    print("[record] 뷰어 창이 닫혀서 중단합니다.")
                    return

            state_deg = {name: float(np.rad2deg(data.qpos[i])) for i, name in enumerate(JOINT_NAMES)}

            renderer.update_scene(data, camera=CAMERA_NAME)
            img = renderer.render()

            # 그리퍼 채널은 관절각 대신 이진 신호(0.0/1.0)를 기록 — 조는 고정이라 qpos는 의미 없음.
            action_values = {**joint_deg, "gripper": bit}
            obs_values = {**state_deg, "gripper": bit, "wrist": img}
            obs_frame = build_dataset_frame(dataset.features, obs_values, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")

            dataset.add_frame({**obs_frame, **action_frame, "task": args.task})

            if step % args.fps == 0:
                led = "ON " if bit else "OFF"
                print(
                    f"  t={step / args.fps:.1f}s arm={[round(joint_deg[n], 1) for n in ARM_JOINT_NAMES]}"
                    f"  | gripper raw={gripper_raw:5.1f} -> bit={bit:.0f} LED={led}"
                )

            elapsed = time.perf_counter() - loop_t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

        dataset.save_episode()
        print(f"[record] 에피소드 {ep + 1} 저장 완료 ({time.perf_counter() - t0:.1f}s, {n_steps} 프레임)")


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

    joint_remap = build_joint_remap(leader, model)
    for name in ARM_JOINT_NAMES:
        l_lo, l_hi, m_lo, m_hi = joint_remap[name]
        print(f"[record] remap {name}: leader[{l_lo:.1f},{l_hi:.1f}] -> mujoco[{m_lo:.1f},{m_hi:.1f}] deg")

    dataset = build_dataset(args)

    try:
        if args.headless:
            _run(args, leader, model, data, renderer, dataset, viewer=None, joint_remap=joint_remap)
        else:
            print("[record] 뷰어 창을 띄웁니다 (--headless로 끌 수 있음).")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                _run(args, leader, model, data, renderer, dataset, viewer, joint_remap=joint_remap)
    except KeyboardInterrupt:
        print("\n[record] 중단됨 — 이미 저장된 에피소드는 유지됩니다.")
    finally:
        leader.disconnect()
        # 필수: 안 부르면 parquet footer 메타데이터가 안 써져서 방금 녹화한 에피소드까지 전부
        # 다음에 못 여는 깨진 데이터셋이 된다 (LeRobotDataset.finalize 문서 참고).
        dataset.finalize()

    print(f"[record] 데이터셋: {dataset.root}")


if __name__ == "__main__":
    main()
