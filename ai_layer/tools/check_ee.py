#!/usr/bin/env python
"""조이스틱 -> MuJoCo EE 전용 리그(ee_rig.xml) 실시간 뷰어. 방향/접촉/실패 확인용.

record_mujoco.py로 전체 녹화를 돌리지 않고, 조이스틱으로 mocap_target을 움직이면서 ee_body가
잘 따라가는지, 비드가 신호(트리거)만으로 찍히는지, 막대가 바닥에 닿으면 실패로 뜨는지, 베이스
버튼으로 roll/pitch/yaw가 도는지 빠르게 확인한다. 기본적으로 뷰어 창을 띄운다.

2026-09-23(3차): roll/pitch/yaw는 베이스 6개 버튼(rotation_rate())으로 레이트 컨트롤한다.
막대가 바닥/용지에 물리적으로 닿으면 "실패"로 표시된다(record_mujoco.py에서는 자동 폐기) —
도구는 표면에 닿지 않고 일정 간격을 띄운 채로 작업해야 한다. 비드는 접촉/높이와 무관하게
트리거만 켜져 있으면 찍히고, 중력 때문에 도구 끝이 아니라 바로 아래 바닥면에 찍힌다.

실행 (pac2026 conda 환경):
  conda activate pac2026
  PYTHONPATH=. python ai_layer/tools/check_ee.py --scene dashed --variant 2

스틱을 움직여서 EE가 기대한 방향(앞/뒤=x, 좌/우=y)으로 가는지, 슬라이더로 z가 오르내리는지,
베이스 버튼으로 회전이 도는지, 트리거로 비드가 찍히는지, 막대가 바닥에 닿으면 "실패"가 뜨는지
확인한다. 방향이 반대면 joystick_input.py의 ee_velocity()에서 부호만 뒤집을 것.
Ctrl+C 또는 뷰어 창 닫기로 종료.
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

from ai_layer.tools.joystick_input import JoystickEEController  # noqa: E402
from ai_layer.tools.record_mujoco import (  # noqa: E402
    BEAD_STRIDE,
    FLOOR_GEOM_NAME,
    IDENTITY_QUAT,
    MAX_ANGULAR_SPEED_DEFAULT,
    MOCAP_BODY_NAME,
    MOCAP_HOME,
    N_VARIANTS,
    ROD_GEOM_NAME,
    SCENE_VARIANTS,
    WORKSPACE_X,
    WORKSPACE_Y,
    WORKSPACE_Z,
    BeadDrop,
    Rotation,
    _contact_pos,
    _draw_bead_trail,
    _mjcf_path,
    _rod_tip_world,
    _rotmat_to_mujoco_quat,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="조이스틱 -> MuJoCo EE 리그 실시간 뷰어 (방향/접촉/실패 확인).")
    p.add_argument("--scene", choices=SCENE_VARIANTS, default="curve", help="A4 용접선 형태")
    p.add_argument("--variant", type=int, default=-1, help=f"0~{N_VARIANTS - 1}, 기본(-1)은 무작위")
    p.add_argument("--max-linear-speed", type=float, default=0.05, help="조이스틱 최대 EE 속도 [m/s]")
    p.add_argument("--max-angular-speed", type=float, default=MAX_ANGULAR_SPEED_DEFAULT, help="베이스 버튼 최대 각속도 [rad/s]")
    p.add_argument("--invert-x", action="store_true", help="EE x축 방향 반전")
    p.add_argument("--invert-y", action="store_true", help="EE y축 방향 반전")
    p.add_argument("--invert-z", action="store_true", help="EE z축 방향 반전")
    p.add_argument("--hz", type=float, default=30.0, help="제어/출력 주기")
    p.add_argument("--headless", action="store_true", help="뷰어 창 없이 콘솔 출력만 (기본: 창 띄움)")
    return p.parse_args()


def _run(
    ctl: JoystickEEController,
    model,
    data,
    viewer,
    max_linear_speed: float,
    max_angular_speed: float,
    hz: float,
    invert_x: bool = False,
    invert_y: bool = False,
    invert_z: bool = False,
) -> None:
    dt = 1.0 / hz
    substeps = max(1, int(round(dt / model.opt.timestep)))

    mocap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODY_NAME)
    mocap_idx = model.body_mocapid[mocap_bid]
    rod_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ROD_GEOM_NAME)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)

    target_pos = MOCAP_HOME.copy()
    R_cmd = np.eye(3)
    data.mocap_pos[mocap_idx] = target_pos
    data.mocap_quat[mocap_idx] = IDENTITY_QUAT
    print_every = max(1, int(hz // 5))
    bead_points: list[BeadDrop] = []

    step = 0
    while True:
        loop_t0 = time.perf_counter()

        ctl.poll()
        vx, vy, vz = ctl.ee_velocity(
            max_linear=max_linear_speed, invert_x=invert_x, invert_y=invert_y, invert_z=invert_z
        )
        target_pos = target_pos + np.array([vx, vy, vz]) * dt
        target_pos[0] = float(np.clip(target_pos[0], *WORKSPACE_X))
        target_pos[1] = float(np.clip(target_pos[1], *WORKSPACE_Y))
        target_pos[2] = float(np.clip(target_pos[2], *WORKSPACE_Z))
        data.mocap_pos[mocap_idx] = target_pos

        wx, wy, wz = ctl.rotation_rate(max_angular=max_angular_speed)
        if wx or wy or wz:
            R_cmd = Rotation.from_rotvec(np.array([wx, wy, wz]) * dt).as_matrix() @ R_cmd
        data.mocap_quat[mocap_idx] = _rotmat_to_mujoco_quat(R_cmd)
        bit = ctl.gripper_bit()

        for _ in range(substeps):
            mujoco.mj_step(model, data)

        contact_pos = _contact_pos(data, rod_gid, floor_gid)
        failed = contact_pos is not None
        if bit and step % BEAD_STRIDE == 0:
            bead_points.append(BeadDrop(_rod_tip_world(data, rod_gid)))
        for b in bead_points:
            b.step(dt)

        if viewer is not None:
            viewer.user_scn.ngeom = 0
            _draw_bead_trail(viewer.user_scn, bead_points)
            viewer.sync()
            if not viewer.is_running():
                print("\n[check_ee] 뷰어 창이 닫혀서 종료합니다.")
                return

        if step % print_every == 0:
            status = "실패(접촉!)" if failed else "정상(안 닿음)"
            tip = _rod_tip_world(data, rod_gid)  # record_mujoco.py가 기록하는 것과 같은 기준점(도구 끝)
            print(
                f"\rtarget=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  "
                f"tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f})  "
                f"trigger={int(bit)}  막대={status}  비드={len(bead_points)}점",
                end="",
                flush=True,
            )
        step += 1

        elapsed = time.perf_counter() - loop_t0
        if elapsed < dt:
            time.sleep(dt - elapsed)


def main() -> None:
    args = parse_args()

    ctl = JoystickEEController()

    variant = args.variant if args.variant >= 0 else random.randint(0, N_VARIANTS - 1)
    mjcf_path = _mjcf_path(args.scene, variant)
    print(f"[check_ee] scene: {args.scene} variant={variant} ({mjcf_path.name})")
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("[check_ee] 스틱/슬라이더/베이스버튼으로 EE를 움직여보고, 트리거로 비드/접촉 시 실패를 확인하세요. Ctrl+C로 종료.\n")

    invert_kwargs = dict(invert_x=args.invert_x, invert_y=args.invert_y, invert_z=args.invert_z)
    try:
        if args.headless:
            _run(
                ctl, model, data, viewer=None,
                max_linear_speed=args.max_linear_speed, max_angular_speed=args.max_angular_speed,
                hz=args.hz, **invert_kwargs,
            )
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                _run(
                    ctl, model, data, viewer,
                    args.max_linear_speed, args.max_angular_speed, args.hz, **invert_kwargs,
                )
    except KeyboardInterrupt:
        print("\n[check_ee] 종료.")
    finally:
        ctl.close()


if __name__ == "__main__":
    main()
