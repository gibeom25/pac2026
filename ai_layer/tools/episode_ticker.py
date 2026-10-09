"""record_mujoco.py `_run_one_episode()`의 물리/비드/접촉/높이제한 while 루프를, GUI 프레임마다
한 스텝씩 부를 수 있는 `tick()` 형태로 쪼갠 버전 — record_gui.py(Dear PyGui)에서 처음 만들었고,
2026-10-06에 PyQt GUI(ai_layer/gui/)도 같은 걸 쓰도록 프레임워크 중립 모듈로 뽑아냈다.

GUI 프레임워크(Dear PyGui/PyQt/...)에 전혀 의존하지 않는다 — numpy/mujoco만 쓴다. 이 파일을
바꾸면 record_gui.py와 PyQt GUI 둘 다 그대로 영향을 받는다(단일 소스).
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from lerobot.datasets.utils import build_dataset_frame
from lerobot.utils.rotation import Rotation

from ai_layer.kinematics import pose_delta, pose_to_state
from ai_layer.tools.joystick_input import JoystickEEController
from ai_layer.tools.keyboard_input import KeyboardEEController
from ai_layer.tools.record_mujoco import (  # noqa: E402 — record_mujoco.py의 검증된 로직 재사용
    ACTION_KEYS,
    BEAD_RGBA,
    BEAD_STRIDE,
    CAMERA_HW,
    CAMERA_NAME,
    EE_BODY_NAME,
    FLOOR_GEOM_NAME,
    IDENTITY_QUAT,
    MIN_TIP_Z,
    MOCAP_BODY_NAME,
    MOCAP_HOME,
    ROD_GEOM_NAME,
    STATE_KEYS,
    WORKSPACE_X,
    WORKSPACE_Y,
    WORKSPACE_Z,
    BeadDrop,
    _contact_pos,
    _draw_bead_trail,
    _ee_pose_xyzrotvec,
    _rod_tip_world,
    _rotmat_to_mujoco_quat,
)

OVERVIEW_CAMERA_NAME = "overview"


class MujocoDualCamera:
    """손목/오버뷰 두 카메라를 같은 mujoco.Renderer로 렌더링.

    실로봇 전환 시 이 클래스를 real-camera 버전으로 바꿔 끼우면 된다 — GUI/에피소드 로직은
    get_wrist_frame()/get_overview_frame()이 (H,W,3) uint8 RGB를 돌려준다는 계약만 보고 동작한다.
    """

    def __init__(self, model):
        self._renderer = mujoco.Renderer(model, height=CAMERA_HW[0], width=CAMERA_HW[1])
        self.bead_rgba = BEAD_RGBA  # generate_demos.py가 에피소드마다 필라멘트 색으로 바꾼다

    def get_wrist_frame(self, data, bead_points: list[BeadDrop]) -> np.ndarray:
        self._renderer.update_scene(data, camera=CAMERA_NAME)
        _draw_bead_trail(self._renderer.scene, bead_points, self.bead_rgba)
        return self._renderer.render()

    def get_overview_frame(self, data, bead_points: list[BeadDrop]) -> np.ndarray:
        self._renderer.update_scene(data, camera=OVERVIEW_CAMERA_NAME)
        _draw_bead_trail(self._renderer.scene, bead_points, self.bead_rgba)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


class EpisodeTicker:
    """GUI는 블로킹 while을 돌릴 수 없으므로(프레임마다 그리고 돌아와야 함) 상태를 인스턴스에
    들고 tick()을 GUI 루프(또는 QTimer)에서 매 프레임 호출한다. 물리/비드/접촉/높이제한 로직은
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
        self.last_tip: np.ndarray | None = None
        self.last_pose: np.ndarray | None = None

    def tick(self, ctl: JoystickEEController | KeyboardEEController, dataset, force_end: bool, force_discard: bool) -> str | None:
        """한 프레임 진행. 끝났으면 "saved"/"discarded", 아니면 None."""
        if self.max_steps is not None and self.step >= self.max_steps:
            return self._finish(dataset, discard=False)

        ctl.poll()
        vx, vy, vz = ctl.ee_velocity(
            max_linear=self.args.max_linear_speed,
            invert_x=self.args.invert_x, invert_y=self.args.invert_y, invert_z=self.args.invert_z,
        )
        # 2026-10-06: 이동 방향을 월드 고정이 아니라 EE(손목) 자신의 현재 자세 기준으로 바꿨다
        # — W/S/A/D/R/F는 이제 "EE가 보는 앞/오른쪽/위"를 뜻하고, 도구를 돌린 뒤에도(회전 중에도
        # 매 틱 갱신되는 self.R_cmd를 그대로 쓰므로) "앞으로"가 계속 도구 기준 앞쪽을 의미한다.
        # R_cmd가 항등(회전 안 한 기본 자세)이면 기존과 완전히 동일(월드축 그대로).
        self.target_pos = self.target_pos + self.R_cmd @ (np.array([vx, vy, vz]) * self.dt)
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
        self.last_tip = tip
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
        self.last_pose = pose
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
