#!/usr/bin/env python
"""조이스틱(Logitech Extreme 3D Pro) -> MuJoCo EE 전용 리그(ee_rig.xml) -> LeRobotDataset 기록.

2026-09-23: 로봇 몸통(5관절 체인) 없이 EE만 시뮬레이션한다 — 시뮬레이션에서는 리더의 EE만
참고하면 되고 관절 구속/캘리브레이션은 신경 쓸 필요 없다는 결정. 리더암 대신 조이스틱으로
`ee_rig.xml`의 mocap_target을 직접 움직인다(joystick_input.py) — 리더 FK가 필요 없어지고,
기구부 백래시/떨림 없이 더 매끄러운 시연 궤적을 얻을 수 있다는 게 의도. mocap_target은
kinematic(물리 반응 없음)이고, 실제 물리 바디(ee_body, freejoint)가 weld로 그 뒤를 따라가며
바닥에 닿으면 진짜 반발력이 생긴다(ee_rig.xml 참고, 안정성 검증 완료).

데이터셋은 관절공간이 아니라 EE-native 포맷으로 직접 기록한다 (더 이상 관절이 없으므로):
  observation.state (9,) = [x, y, z, rot6d(6)]       -- kinematics.pose_to_state와 동일 표현
  observation.images.wrist                            -- 렌더링된 손목 카메라
  action (7,) = [dx, dy, dz, drx, dry, drz, gripper]  -- 이전 프레임 측정 pose -> 이번 프레임 증분
이미 ai_layer/configs/so101_act_bc.py의 ACTConfig 입출력 스펙과 형태가 같다.
⚠️ train_bc.py가 쓰는 SO101BCDataset은 아직 관절공간 -> EEF 변환을 전제로 하므로 이 포맷을
바로 못 읽는다 — 이 도구로 모은 데이터를 학습에 쓰려면 SO101BCDataset 쪽에 "이미 EEF 포맷인
데이터셋은 변환 없이 그대로 통과" 경로를 추가하는 후속 작업이 필요하다.

실행 (pac2026 conda 환경):
  conda activate pac2026
  PYTHONPATH=. python ai_layer/tools/record_mujoco.py \
      --repo-id <hf-user>/so101-mujoco-demo --root ./datasets/so101-mujoco-demo \
      --scene dashed --num-episodes 5 --episode-seconds 15

BTN_TRIGGER를 누르고 있는 동안 그리퍼/도구 신호=1, 막대가 실제로 바닥/용지에 닿아 있을 때만
그 접촉점에 비드가 찍힌다(신호만 켜져 있고 떠 있으면 안 찍힘). 에피소드 사이에 Enter를 누르면
다음 녹화를 시작한다. Ctrl+C로 중단하면 그때까지 저장된 에피소드는 유지된다.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.kinematics import pose_delta, pose_to_state, pose_to_xyzrotvec  # noqa: E402
from ai_layer.tools.joystick_input import JoystickEEController  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "so101"
SCENE_VARIANTS = ["curve", "straight", "sharp_curve", "corner", "branch", "dashed"]
# textures/gen_seam_textures.py --variants 기본값과 맞춤. variant 0 = 노이즈 없는 기준
# (scene_a4_<scene>.xml), 1~N_VARIANTS-1 = 가우시안 노이즈 준 사전 생성본(scene_a4_<scene>_<i>.xml) —
# 곡선 진폭/주기/위상, 코너 각도/위치, 분기 각도, 점선 간격, 선 굵기 등을 형태별로 흔들어둔 것.
# 런타임에 텍스처를 바꾸려면 mjr_uploadTexture로 GPU 재업로드 + 뷰어와의 동기화가 필요해 번거로워서,
# 오프라인에 미리 구워두고 --variant로 고르는 쪽을 택함 (2026-09-23).
N_VARIANTS = 5
CAMERA_NAME = "wrist"
CAMERA_HW = (240, 320, 3)

MOCAP_BODY_NAME = "mocap_target"
EE_BODY_NAME = "ee_body"
ROD_GEOM_NAME = "tool_rod"
FLOOR_GEOM_NAME = "floor"

# EE-native observation/action 벡터의 개별 스칼라 이름 (hw_to_dataset_features가 자동으로
# observation.state(9,)/action(7,) 벡터 feature로 묶어준다 — 관절 이름 대신 이 이름들을 쓴다).
STATE_KEYS = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5"]
ACTION_KEYS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]

# mocap_target 위치 clamp — 관절 리치 제약이 없어져서(자유 EE) 안 씌우면 슬라이더를 오래 누르고
# 있을 때 작업공간 밖으로 계속 날아간다. A4 용지(x=0.25 중심, ±0.1485)보다 약간 넉넉하게 잡음.
WORKSPACE_X = (0.05, 0.45)
WORKSPACE_Y = (-0.20, 0.20)
WORKSPACE_Z = (-0.02, 0.35)

MOCAP_HOME = np.array([0.25, 0.0, 0.15])

# 2026-09-23: 신호 on(1) + 막대-바닥 접촉 동안 접촉점에 그리스/실리콘 비드처럼 자국을 남긴다.
# 렌더된 손목 카메라 이미지에도 남아서(뷰어 전용이 아님) BC가 이미 도포된 구간을 시각적으로
# 구분할 수 있다. mjv_initGeom으로 씬에 시각 전용 geom을 얹는 방식이라 이 자국 자체는 물리에
# 영향 없다 (mujoco.Renderer.scene / viewer.user_scn 둘 다 지원).
# 실리콘 느낌: 살짝 반투명한 미색 + 무광에 가까운 낮은 광택(진짜 광택 플라스틱처럼 반짝이지 않게
# specular/shininess를 낮게, reflectance는 거의 0으로 — mjvGeom은 material 없이도 이 필드들을
# geom 단위로 직접 지원한다(mjv_initGeom이 채우는 기본 필드 외에 아래서 수동으로 덮어씀).
BEAD_RGBA = np.array([0.85, 0.83, 0.78, 0.95], dtype=np.float32)  # 살짝 반투명한 미색(실리콘)
BEAD_SPECULAR = 0.25
BEAD_SHININESS = 0.15
BEAD_REFLECTANCE = 0.05
BEAD_RADIUS = 0.0018  # 1.8mm — 촘촘한 간격과 겹쳐 매끈하게 이어진 비드처럼 보이게 살짝 키움
BEAD_STRIDE = 1  # 매 스텝 찍음 — 점 간격이 구슬 반지름보다 촘촘해져 거의 이어진 선처럼 보임


def _mjcf_path(scene: str, variant: int = 0) -> Path:
    if variant == 0:
        name = "scene_a4.xml" if scene == "curve" else f"scene_a4_{scene}.xml"
    else:
        name = f"scene_a4_{scene}_{variant}.xml"
    return ASSETS_DIR / name


def _draw_bead_trail(scene, points: list[np.ndarray]) -> None:
    """축적된 비드 자국(world xyz 리스트)을 시각 전용 geom으로 씬에 얹는다.

    scene.ngeom을 리셋하지 않고 이어서 채운다 — renderer.scene은 update_scene() 직후(이미 모델
    geom들로 ngeom이 채워진 상태) 호출하고, viewer.user_scn은 매 프레임 호출 전에 ngeom=0으로
    직접 리셋해야 한다(그래야 매번 전체 궤적을 다시 그리지, 프레임마다 누적 중복되지 않는다).
    """
    size = np.array([BEAD_RADIUS, 0.0, 0.0])
    mat = np.eye(3).flatten()
    for p in points:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=size, pos=p, mat=mat, rgba=BEAD_RGBA)
        g.specular = BEAD_SPECULAR
        g.shininess = BEAD_SHININESS
        g.reflectance = BEAD_REFLECTANCE
        scene.ngeom += 1


def _contact_pos(data, gid_a: int, gid_b: int) -> np.ndarray | None:
    """gid_a<->gid_b 접촉이 있으면 첫 접촉점 world 좌표, 없으면 None."""
    for i in range(data.ncon):
        c = data.contact[i]
        if {c.geom1, c.geom2} == {gid_a, gid_b}:
            return c.pos.copy()
    return None


def _yaw_quat(yaw: float) -> np.ndarray:
    """world z축 기준 yaw 회전 -> MuJoCo quat [w,x,y,z]. mocap_target은 pitch/roll을 안 가짐 —
    ee_body가 weld+접촉 반발력으로 알아서 기울고(_ee_pose_xyzrotvec으로 실측 기록), mocap은
    조이스틱 트위스트로 명령한 yaw만 목표로 준다."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _ee_pose_xyzrotvec(data, ee_bid: int) -> np.ndarray:
    """ee_body의 실측 world pose(회전은 data.xmat, 별도 FK 없음) -> (6,) [xyz, rotvec]."""
    T = np.eye(4)
    T[:3, :3] = data.xmat[ee_bid].reshape(3, 3)
    T[:3, 3] = data.xpos[ee_bid]
    return pose_to_xyzrotvec(T)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="조이스틱 -> MuJoCo EE 리그 미러링 데이터 수집.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="로컬 저장 경로 (없으면 HF_LEROBOT_HOME/<repo-id>)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--episode-seconds", type=float, default=15.0)
    p.add_argument("--task", default="weld seam following demo (mujoco ee-only, joystick)")
    p.add_argument("--scene", choices=SCENE_VARIANTS, default="curve", help="A4 용접선 형태")
    p.add_argument(
        "--variant", type=int, default=-1,
        help=f"용접선 변형(0={{노이즈 없음}}, 1~{N_VARIANTS - 1}=가우시안 노이즈 사전생성본). "
             f"기본(-1)은 매번 무작위로 고름 — textures/gen_seam_textures.py 참고",
    )
    p.add_argument("--headless", action="store_true", help="라이브 뷰어 창 없이 실행 (기본: 창 띄움)")
    p.add_argument("--max-linear-speed", type=float, default=0.05, help="조이스틱 최대 EE 속도 [m/s]")
    p.add_argument("--max-angular-speed", type=float, default=1.0, help="조이스틱 트위스트 최대 yaw 각속도 [rad/s]")
    p.add_argument("--invert-x", action="store_true", help="EE x축 방향 반전")
    p.add_argument("--invert-y", action="store_true", help="EE y축 방향 반전")
    p.add_argument("--invert-z", action="store_true", help="EE z축 방향 반전")
    p.add_argument("--invert-yaw", action="store_true", help="트위스트 yaw 방향 반전")
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
    hw_obs = {name: float for name in STATE_KEYS}
    hw_obs["wrist"] = CAMERA_HW
    hw_action = {name: float for name in ACTION_KEYS}

    obs_features = hw_to_dataset_features(hw_obs, "observation", use_video=False)
    action_features = hw_to_dataset_features(hw_action, "action", use_video=False)
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
        robot_type="so101_ee_mujoco_joystick",
        use_videos=False,
    )


def _run(args: argparse.Namespace, ctl: JoystickEEController, model, data, renderer, dataset, viewer) -> None:
    dt = 1.0 / args.fps
    substeps = max(1, int(round(dt / model.opt.timestep)))

    mocap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODY_NAME)
    mocap_idx = model.body_mocapid[mocap_bid]
    ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
    rod_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ROD_GEOM_NAME)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
    print(f"[record] EE 최대속도={args.max_linear_speed} m/s, 작업공간 x={WORKSPACE_X} y={WORKSPACE_Y} z={WORKSPACE_Z}")

    for ep in range(args.num_episodes):
        input(f"\n[record] 에피소드 {ep + 1}/{args.num_episodes} — 준비되면 Enter (Ctrl+C 종료) ")
        mujoco.mj_resetData(model, data)
        target_pos = MOCAP_HOME.copy()
        yaw = 0.0
        data.mocap_pos[mocap_idx] = target_pos
        data.mocap_quat[mocap_idx] = _yaw_quat(yaw)
        mujoco.mj_forward(model, data)
        bead_points: list[np.ndarray] = []
        prev_pose: np.ndarray | None = None

        t0 = time.perf_counter()
        n_steps = int(args.episode_seconds * args.fps)
        for step in range(n_steps):
            loop_t0 = time.perf_counter()

            ctl.poll()
            vx, vy, vz = ctl.ee_velocity(
                max_linear=args.max_linear_speed,
                invert_x=args.invert_x,
                invert_y=args.invert_y,
                invert_z=args.invert_z,
            )
            target_pos = target_pos + np.array([vx, vy, vz]) * dt
            target_pos[0] = float(np.clip(target_pos[0], *WORKSPACE_X))
            target_pos[1] = float(np.clip(target_pos[1], *WORKSPACE_Y))
            target_pos[2] = float(np.clip(target_pos[2], *WORKSPACE_Z))
            data.mocap_pos[mocap_idx] = target_pos
            yaw += ctl.yaw_rate(max_angular=args.max_angular_speed, invert=args.invert_yaw) * dt
            data.mocap_quat[mocap_idx] = _yaw_quat(yaw)
            bit = ctl.gripper_bit()

            for _ in range(substeps):
                mujoco.mj_step(model, data)

            contact_pos = _contact_pos(data, rod_gid, floor_gid)
            if bit and contact_pos is not None and step % BEAD_STRIDE == 0:
                bead_points.append(contact_pos)

            if viewer is not None:
                viewer.user_scn.ngeom = 0
                _draw_bead_trail(viewer.user_scn, bead_points)
                viewer.sync()
                if not viewer.is_running():
                    print("[record] 뷰어 창이 닫혀서 중단합니다.")
                    return

            pose = _ee_pose_xyzrotvec(data, ee_bid)  # (6,) [xyz, rotvec], 실측
            if prev_pose is None:
                delta6 = np.zeros(6)
            else:
                delta6 = pose_delta(prev_pose, pose)
            prev_pose = pose

            renderer.update_scene(data, camera=CAMERA_NAME)
            _draw_bead_trail(renderer.scene, bead_points)
            img = renderer.render()

            state9 = pose_to_state(pose)
            obs_values = {**dict(zip(STATE_KEYS, state9.tolist())), "wrist": img}
            action_values = {**dict(zip(ACTION_KEYS[:6], delta6.tolist())), "gripper": bit}
            obs_frame = build_dataset_frame(dataset.features, obs_values, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")

            dataset.add_frame({**obs_frame, **action_frame, "task": args.task})

            if step % args.fps == 0:
                touching = "접촉" if contact_pos is not None else "떠있음"
                print(
                    f"  t={step / args.fps:.1f}s pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})"
                    f"  | trigger={bit:.0f} 막대={touching} 비드={len(bead_points)}점"
                )

            elapsed = time.perf_counter() - loop_t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

        dataset.save_episode()
        print(f"[record] 에피소드 {ep + 1} 저장 완료 ({time.perf_counter() - t0:.1f}s, {n_steps} 프레임)")


def main() -> None:
    args = parse_args()

    ctl = JoystickEEController()

    variant = args.variant if args.variant >= 0 else random.randint(0, N_VARIANTS - 1)
    mjcf_path = _mjcf_path(args.scene, variant)
    print(f"[record] scene: {args.scene} variant={variant} ({mjcf_path.name})")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=CAMERA_HW[0], width=CAMERA_HW[1])
    mujoco.mj_forward(model, data)

    dataset = build_dataset(args)

    try:
        if args.headless:
            _run(args, ctl, model, data, renderer, dataset, viewer=None)
        else:
            print("[record] 뷰어 창을 띄웁니다 (--headless로 끌 수 있음).")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                _run(args, ctl, model, data, renderer, dataset, viewer)
    except KeyboardInterrupt:
        print("\n[record] 중단됨 — 이미 저장된 에피소드는 유지됩니다.")
    finally:
        ctl.close()
        # 필수: 안 부르면 parquet footer 메타데이터가 안 써져서 방금 녹화한 에피소드까지 전부
        # 다음에 못 여는 깨진 데이터셋이 된다 (LeRobotDataset.finalize 문서 참고).
        dataset.finalize()

    print(f"[record] 데이터셋: {dataset.root}")


if __name__ == "__main__":
    main()
