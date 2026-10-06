#!/usr/bin/env python
"""데이터 수집 GUI (Dear PyGui) — 손목 카메라 + 오버뷰 카메라 실시간 화면, 버튼, 수집 현황 차트.

2026-10-06(기범) 요청 4가지를 하나로 묶었다:
  1. MuJoCo 자유시점 뷰어(mujoco.viewer, 마우스로 돌려보는 디버그 카메라) 대신, "설치된 카메라"
     (손목 카메라, 데이터셋에 실제로 저장되는 바로 그 화면)와 "전체 카메라"(assets/so101/
     scene_common.xml의 고정 overview 카메라)를 나란히 보여준다 — 나중에 실로봇으로 녹화할 때도
     "손목 카메라 화면 + 작업대를 보는 고정 카메라 화면"이라는 같은 레이아웃을 쓸 수 있게,
     카메라 소스를 MujocoDualCamera 클래스 하나로 묶어뒀다(실로봇 전환 시 이 클래스만
     cv2.VideoCapture 등으로 교체하면 됨 — GUI/에피소드 로직은 그대로).
  2. 녹화 버튼들(에피소드 시작/저장/폐기)을 화면 버튼으로도 조작 가능(조이스틱 BTN_THUMB/
     BTN_THUMB2와 동일 기능 — 조이스틱이 메인 입력이고 버튼은 보조/대체 수단, 실로봇에서
     조이스틱 없이 화면만으로 조작해야 하는 상황도 고려).
  3. (형태, variant)별로 지금까지 몇 개 모였는지 막대그래프로 보여준다(BalancedSceneSampler의
     실시간 카운트 — 세션 중 즉시 갱신, meta/scene_balance.json과 동일 소스).

record_mujoco.py의 물리/조이스틱/데이터셋/balanced 샘플링 로직을 그대로 재사용한다(새로 안
만듦) — 여기서 새로 만든 건 "한 번에 쭉 도는 while 루프"를 "GUI 프레임마다 한 스텝씩" 도는
형태로 바꾼 _EpisodeTicker 뿐이다 (record_mujoco.py의 터미널 while 루프는 안 건드림 — 이미
검증된 터미널 플로우를 그대로 쓰고 싶은 사람은 계속 그걸 쓰면 됨).

실행 (pac2026 conda 환경, pip install dearpygui 필요):
  PYTHONPATH=. python ai_layer/tools/record_gui.py --repo-id <hf-user>/so101-weld-demo --num-episodes 30
  PYTHONPATH=. python ai_layer/tools/record_gui.py --dry-run --num-episodes 3   # 저장 없이 연습
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import dearpygui.dearpygui as dpg
import mujoco
import numpy as np

from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.rotation import Rotation

from ai_layer.kinematics import pose_delta, pose_to_state
from ai_layer.tools.joystick_input import JoystickEEController
from ai_layer.tools.keyboard_input import KeyboardEEController
from ai_layer.tools.teleop_input import add_input_arg, build_ee_controller
from ai_layer.tools.record_mujoco import (  # noqa: E402 — record_mujoco.py의 검증된 로직 재사용
    ACTION_KEYS,
    BEAD_STRIDE,
    CAMERA_HW,
    CAMERA_NAME,
    EE_BODY_NAME,
    FLOOR_GEOM_NAME,
    IDENTITY_QUAT,
    MIN_TIP_Z,
    MOCAP_BODY_NAME,
    MOCAP_HOME,
    N_VARIANTS,
    ROD_GEOM_NAME,
    SCENE_VARIANTS,
    STATE_KEYS,
    WORKSPACE_X,
    WORKSPACE_Y,
    WORKSPACE_Z,
    BalancedSceneSampler,
    BeadDrop,
    _build_combos,
    _contact_pos,
    _draw_bead_trail,
    _ee_pose_xyzrotvec,
    _mjcf_path,
    _rod_tip_world,
    _rotmat_to_mujoco_quat,
    build_dataset,
)

OVERVIEW_CAMERA_NAME = "overview"


class MujocoDualCamera:
    """손목/오버뷰 두 카메라를 같은 mujoco.Renderer로 렌더링.

    실로봇 전환 시 이 클래스를 real-camera 버전으로 바꿔 끼우면 된다 — GUI/에피소드 로직은
    get_wrist_frame()/get_overview_frame()이 (H,W,3) uint8 RGB를 돌려준다는 계약만 보고 동작한다.
    """

    def __init__(self, model):
        self._renderer = mujoco.Renderer(model, height=CAMERA_HW[0], width=CAMERA_HW[1])

    def get_wrist_frame(self, data, bead_points: list[BeadDrop]) -> np.ndarray:
        self._renderer.update_scene(data, camera=CAMERA_NAME)
        _draw_bead_trail(self._renderer.scene, bead_points)
        return self._renderer.render()

    def get_overview_frame(self, data, bead_points: list[BeadDrop]) -> np.ndarray:
        self._renderer.update_scene(data, camera=OVERVIEW_CAMERA_NAME)
        _draw_bead_trail(self._renderer.scene, bead_points)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


def _rgb_to_dpg_texture(frame_rgb_uint8: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB -> Dear PyGui raw 텍스처용 flat float32 RGBA [0,1]."""
    h, w, _ = frame_rgb_uint8.shape
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = frame_rgb_uint8.astype(np.float32) / 255.0
    rgba[:, :, 3] = 1.0
    return rgba.flatten()


class _EpisodeTicker:
    """record_mujoco.py `_run_one_episode()`의 while 루프 한 바퀴를 tick() 한 번으로 쪼갠 버전.

    GUI는 블로킹 while을 돌릴 수 없으므로(프레임마다 그리고 돌아와야 함) 상태를 인스턴스에
    들고 tick()을 GUI 루프에서 매 프레임 호출한다. 물리/비드/접촉/높이제한 로직은
    record_mujoco.py와 완전히 동일 — 바깥 루프 구조만 다르다.
    """

    def __init__(self, args: argparse.Namespace, model, data, dual_cam: MujocoDualCamera, scene: str, variant: int):
        self.args = args
        self.model = model
        self.data = data
        self.dual_cam = dual_cam
        self.scene = scene
        self.variant = variant

        self.dt = 1.0 / args.fps
        self.substeps = max(1, int(round(self.dt / model.opt.timestep)))
        self.mocap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODY_NAME)
        self.mocap_idx = model.body_mocapid[self.mocap_bid]
        self.ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        self.rod_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ROD_GEOM_NAME)
        self.floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)

        mujoco.mj_resetData(model, data)
        self.target_pos = MOCAP_HOME.copy()
        self.R_cmd = np.eye(3)
        data.mocap_pos[self.mocap_idx] = self.target_pos
        data.mocap_quat[self.mocap_idx] = IDENTITY_QUAT
        mujoco.mj_forward(model, data)

        self.bead_points: list[BeadDrop] = []
        self.prev_pose: np.ndarray | None = None
        self.t0 = time.perf_counter()
        self.max_steps = int(args.episode_seconds * args.fps) if args.episode_seconds else None
        self.step = 0
        self.last_wrist_frame: np.ndarray | None = None
        self.last_overview_frame: np.ndarray | None = None
        self.last_contact = False
        self.last_trigger = 0.0

    def tick(self, ctl: JoystickEEController | KeyboardEEController, dataset, force_end: bool, force_discard: bool) -> str | None:
        """한 프레임 진행. 끝났으면 "saved"/"discarded", 아니면 None."""
        if self.max_steps is not None and self.step >= self.max_steps:
            return self._finish(dataset, discard=False)

        ctl.poll()
        vx, vy, vz = ctl.ee_velocity(
            max_linear=self.args.max_linear_speed,
            invert_x=self.args.invert_x, invert_y=self.args.invert_y, invert_z=self.args.invert_z,
        )
        self.target_pos = self.target_pos + np.array([vx, vy, vz]) * self.dt
        self.target_pos[0] = float(np.clip(self.target_pos[0], *WORKSPACE_X))
        self.target_pos[1] = float(np.clip(self.target_pos[1], *WORKSPACE_Y))
        self.target_pos[2] = float(np.clip(self.target_pos[2], *WORKSPACE_Z))
        min_target_z = MIN_TIP_Z + 0.05 * self.R_cmd[2, 2]  # record_mujoco.py와 동일 근사(ROD_HALF_LENGTH*2)
        self.target_pos[2] = max(self.target_pos[2], min_target_z)
        self.data.mocap_pos[self.mocap_idx] = self.target_pos

        wx, wy, wz = ctl.rotation_rate(
            max_angular=self.args.max_angular_speed, invert_x=self.args.invert_roll, invert_y=self.args.invert_pitch
        )
        if wx or wy or wz:
            self.R_cmd = Rotation.from_rotvec(np.array([wx, wy, wz]) * self.dt).as_matrix() @ self.R_cmd
        self.data.mocap_quat[self.mocap_idx] = _rotmat_to_mujoco_quat(self.R_cmd)
        bit = ctl.gripper_bit()
        self.last_trigger = bit

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        tip = _rod_tip_world(self.data, self.rod_gid)
        contact_pos = _contact_pos(self.data, self.rod_gid, self.floor_gid)
        self.last_contact = contact_pos is not None
        if tip[2] < MIN_TIP_Z:
            self.target_pos[2] += MIN_TIP_Z - tip[2]
        if bit and self.step % BEAD_STRIDE == 0:
            self.bead_points.append(BeadDrop(tip.copy()))
        for b in self.bead_points:
            b.step(self.dt)

        self.last_wrist_frame = self.dual_cam.get_wrist_frame(self.data, self.bead_points)
        self.last_overview_frame = self.dual_cam.get_overview_frame(self.data, self.bead_points)

        pose = _ee_pose_xyzrotvec(self.data, self.ee_bid, self.rod_gid)
        delta6 = np.zeros(6) if self.prev_pose is None else pose_delta(self.prev_pose, pose)
        self.prev_pose = pose

        if dataset is not None:
            state9 = pose_to_state(pose)
            obs_values = {**dict(zip(STATE_KEYS, state9.tolist())), "wrist": self.last_wrist_frame}
            action_values = {**dict(zip(ACTION_KEYS[:6], delta6.tolist())), "gripper": bit}
            obs_frame = build_dataset_frame(dataset.features, obs_values, prefix="observation")
            action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")
            dataset.add_frame({**obs_frame, **action_frame, "task": self.args.task})

        self.step += 1
        if force_discard or ctl.discard_requested():
            return self._finish(dataset, discard=True)
        if force_end or ctl.episode_end_requested():
            return self._finish(dataset, discard=False)
        return None

    def _finish(self, dataset, discard: bool) -> str:
        if discard:
            if dataset is not None:
                dataset.clear_episode_buffer()
            return "discarded"
        if dataset is not None:
            dataset.save_episode()
        return "saved"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="데이터 수집 GUI (Dear PyGui) — 카메라 화면 + 버튼 + 수집 현황.")
    p.add_argument("--repo-id", default=None, help="--dry-run이 아니면 필수")
    p.add_argument("--root", default=None, help="로컬 저장 경로 (없으면 datasets/<repo-id>)")
    p.add_argument("--dry-run", action="store_true", help="저장 없이 GUI/조작만 연습")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--episode-seconds", type=float, default=None)
    p.add_argument("--task", default="weld seam following demo (mujoco ee-only, joystick, gui)")
    p.add_argument("--scene", choices=["balanced", *SCENE_VARIANTS], default="balanced")
    p.add_argument("--variant", type=int, default=-1)
    p.add_argument("--max-linear-speed", type=float, default=0.05)
    p.add_argument("--max-angular-speed", type=float, default=1.0)
    p.add_argument("--invert-x", action="store_true")
    p.add_argument("--invert-y", action="store_true")
    p.add_argument("--invert-z", action="store_true")
    p.add_argument("--invert-roll", action="store_true")
    p.add_argument("--invert-pitch", action="store_true")
    p.add_argument("--recalibrate-joystick", action="store_true")
    add_input_arg(p)
    args = p.parse_args()
    if args.repo_id is None and not args.dry_run:
        p.error("--repo-id는 --dry-run이 아니면 필수다.")
    return args


def main() -> None:
    args = parse_args()
    combos = _build_combos(args.scene, args.variant)

    # 2026-10-06: 창을 조이스틱 연결/보정, 데이터셋 덮어쓰기 확인 등 터미널 input()이 필요할 수
    # 있는 모든 단계보다 먼저 띄운다 — record_mujoco.py/check_ee.py와 같은 이유(창이 하나도 없이
    # 터미널만 기다리면 "헤드리스로 도는 거 아니냐"는 오해가 생김). GUI 도구인 만큼 특히 더
    # 창부터 보여야 한다.
    dpg.create_context()
    wrist_tex = np.zeros(CAMERA_HW[0] * CAMERA_HW[1] * 4, dtype=np.float32)
    overview_tex = np.zeros(CAMERA_HW[0] * CAMERA_HW[1] * 4, dtype=np.float32)
    with dpg.texture_registry():
        dpg.add_raw_texture(CAMERA_HW[1], CAMERA_HW[0], wrist_tex, tag="wrist_tex", format=dpg.mvFormat_Float_rgba)
        dpg.add_raw_texture(CAMERA_HW[1], CAMERA_HW[0], overview_tex, tag="overview_tex", format=dpg.mvFormat_Float_rgba)

    scene_names = sorted({s for s, _ in combos})
    ui = {"saved_count": 0, "ticker": None, "dual_cam": None, "force_end": False, "force_discard": False}

    def _scene_counts() -> dict[str, int]:
        agg: dict[str, int] = {s: 0 for s in scene_names}
        for key, n in sampler.counts.items():
            s = key.split(":")[0]
            agg[s] = agg.get(s, 0) + n
        return agg

    def _refresh_chart() -> None:
        agg = _scene_counts()
        names = list(agg.keys())
        counts = [agg[n] for n in names]
        dpg.set_value("chart_series", [list(range(len(names))), counts])
        dpg.set_axis_ticks("chart_x", tuple((n, i) for i, n in enumerate(names)))

    def _start_episode(sender=None, app_data=None, user_data=None) -> None:
        if ui["saved_count"] >= args.num_episodes:
            dpg.set_value("status_text", "모든 에피소드 완료.")
            return
        scene, variant = sampler.pick()
        mjcf_path = _mjcf_path(scene, variant)
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dual_cam = MujocoDualCamera(model)
        ui["dual_cam"] = dual_cam
        ui["ticker"] = _EpisodeTicker(args, model, data, dual_cam, scene, variant)
        ui["force_end"] = False
        ui["force_discard"] = False
        dpg.configure_item("start_btn", enabled=False)
        dpg.configure_item("save_btn", enabled=True)
        dpg.configure_item("discard_btn", enabled=True)
        dpg.set_value(
            "status_text",
            f"녹화 중 — 에피소드 {ui['saved_count'] + 1}/{args.num_episodes} scene={scene} variant={variant}",
        )

    def _request_save(sender=None, app_data=None, user_data=None) -> None:
        ui["force_end"] = True

    def _request_discard(sender=None, app_data=None, user_data=None) -> None:
        ui["force_discard"] = True

    with dpg.window(tag="main", label="PAC2026 데이터 수집"):
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("손목 카메라 (데이터셋 저장 화면)")
                dpg.add_image("wrist_tex", width=CAMERA_HW[1], height=CAMERA_HW[0])
            with dpg.group():
                dpg.add_text("전체(오버뷰) 카메라")
                dpg.add_image("overview_tex", width=CAMERA_HW[1], height=CAMERA_HW[0])

        dpg.add_separator()
        dpg.add_text("대기 중 — Start Episode를 누르세요.", tag="status_text")
        dpg.add_text("", tag="frame_status_text")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start Episode", tag="start_btn", callback=_start_episode)
            dpg.add_button(label="Save && End (BTN_THUMB)", tag="save_btn", callback=_request_save, enabled=False)
            dpg.add_button(label="Discard && Retry (BTN_THUMB2)", tag="discard_btn", callback=_request_discard, enabled=False)

        dpg.add_separator()
        dpg.add_text("형태별 수집 현황 (저장된 에피소드 수)")
        with dpg.plot(height=220, width=600):
            dpg.add_plot_axis(dpg.mvXAxis, tag="chart_x")
            with dpg.plot_axis(dpg.mvYAxis, tag="chart_y"):
                dpg.add_bar_series([], [], tag="chart_series", weight=0.6)

    dpg.create_viewport(title="PAC2026 데이터 수집", width=760, height=760)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    dpg.render_dearpygui_frame()  # 아래 데이터셋/조이스틱 준비 단계 전에 창을 실제로 한 번 그려둔다

    # 창이 뜬 다음에야 터미널 input()이 필요할 수 있는 단계를 진행한다(데이터셋 덮어쓰기 확인,
    # 조이스틱 스로틀 보정) — 위 주석 참고.
    if args.dry_run:
        dataset = None
        sampler = BalancedSceneSampler(combos, counts_path=None)
    else:
        dataset = build_dataset(args)
        counts_path = Path(dataset.root) / "meta" / "scene_balance.json"
        sampler = BalancedSceneSampler(combos, counts_path)
    _refresh_chart()

    ctl = build_ee_controller(args.input, recalibrate=args.recalibrate_joystick)

    try:
        while dpg.is_dearpygui_running():
            ticker: _EpisodeTicker | None = ui["ticker"]
            if ticker is not None:
                result = ticker.tick(ctl, dataset, ui["force_end"], ui["force_discard"])
                dpg.set_value("wrist_tex", _rgb_to_dpg_texture(ticker.last_wrist_frame))
                dpg.set_value("overview_tex", _rgb_to_dpg_texture(ticker.last_overview_frame))
                contact_txt = "접촉" if ticker.last_contact else "정상"
                dpg.set_value(
                    "frame_status_text",
                    f"t={ticker.step / args.fps:.1f}s trigger={ticker.last_trigger:.0f} "
                    f"막대={contact_txt} 비드={len(ticker.bead_points)}점",
                )
                if result is not None:
                    ticker.dual_cam.close()
                    if result == "saved":
                        sampler.commit(ticker.scene, ticker.variant)
                        ui["saved_count"] += 1
                    ui["ticker"] = None
                    dpg.configure_item("start_btn", enabled=True)
                    dpg.configure_item("save_btn", enabled=False)
                    dpg.configure_item("discard_btn", enabled=False)
                    label = "저장됨" if result == "saved" else "폐기됨"
                    dpg.set_value(
                        "status_text",
                        f"에피소드 {label} ({ui['saved_count']}/{args.num_episodes}) — "
                        "Start Episode를 누르면 다음 씬으로 이어서.",
                    )
                    _refresh_chart()
            dpg.render_dearpygui_frame()
    finally:
        if ui["ticker"] is not None and ui["dual_cam"] is not None:
            ui["dual_cam"].close()
        dpg.destroy_context()
        ctl.close()
        if dataset is not None:
            dataset.finalize()
        print(f"[record_gui] {'데이터셋: ' + str(dataset.root) if dataset is not None else 'dry-run 종료 (저장 안 함)'}")


if __name__ == "__main__":
    main()
