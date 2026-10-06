"""실로봇(SO-101 follower) EE-native 틱 루프.

ai_layer/tools/episode_ticker.py의 `EpisodeTicker`(시뮬)와 **똑같은 공개 인터페이스**를
구현한다 — `tick(ctl, dataset, force_end, force_discard)`, `scene`/`variant`/`step`/
`last_wrist_frame`/`last_overview_frame`/`last_contact`/`last_trigger`/`target_pos`/
`last_tip`/`bead_points`/`dual_cam`(.close() 가능). 그래서 GUI(ai_layer/gui/collect_tab.py)나
CLI 쪽 틱 호출 코드는 시뮬 EpisodeTicker와 이 RealRobotEpisodeTicker 중 뭘 들고 있는지 신경
쓸 필요가 없다 — "source" 선택만 다르고 나머지 흐름(에피소드 시작/저장/폐기, 상태 표시)은 동일.

**조이스틱/키보드 컨트롤러는 손댈 필요가 없다** — JoystickEEController/KeyboardEEController/
QtKeyboardEEController는 애초에 "EE 속도(m/s, rad/s)"만 돌려주는 하드웨어 중립 인터페이스라서
시뮬 mocap이든 실로봇 IK든 똑같이 넣어주면 된다(ee_velocity()/rotation_rate()/gripper_bit()).

**데이터셋 스키마는 시뮬과 완전히 동일하다**(2026-10-06 결정 — BC/RL 학습 파이프라인을 안
바꾸기 위해) — record_mujoco.py의 STATE_KEYS/ACTION_KEYS/CAMERA_HW를 그대로 쓰고, "wrist"
카메라 키도 동일하게 쓴다. 실로봇 쪽엔 당연히 "bead"(용접 비드) 시뮬레이션이나 접촉 센서가
없으므로 bead_points는 항상 빈 리스트, last_contact는 항상 False로 둔다(데이터셋 필드 자체에
비드/접촉은 안 들어가므로 영향 없음 — 그냥 GUI 상태 표시용 필드).

**IK**: ai_layer/kinematics.py의 build_arm_kinematics()(PAC_Supermoon URDF, 5관절 — yaw 없음)를
쓴다. 매 틱 "현재 관절각 근처에서" 풀어야 해가 널뛰지 않아서, 직전 틱의 해를 다음 틱의 초기
추정치로 재사용한다(placo 솔버가 current_joint_pos를 초기값으로 받음).

**안전장치**: SOFollowerRobotConfig의 `max_relative_target`(한 번의 send_action당 관절 이동량
상한, lerobot 자체 안전장치)을 반드시 쓴다 — 실제 하드웨어라 IK가 튀는 해를 내도 모터가
한번에 크게 안 튄다. 그래도 **이 코드는 실제 로봇으로 검증 못 했다**(이 환경엔 하드웨어가
없음) — 처음 돌릴 땐 낮은 속도로 충분히 떨어져서 지켜볼 것.
"""

from __future__ import annotations

import argparse

import numpy as np
from lerobot.datasets.utils import build_dataset_frame
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

from ai_layer.kinematics import JOINT_NAMES, pose_delta, pose_to_state, pose_to_xyzrotvec
from ai_layer.tools.joystick_input import JoystickEEController
from ai_layer.tools.keyboard_input import KeyboardEEController
from ai_layer.tools.record_mujoco import (
    ACTION_KEYS,
    MIN_TIP_Z,
    STATE_KEYS,
    WORKSPACE_X,
    WORKSPACE_Y,
    WORKSPACE_Z,
)

ARM_JOINT_NAMES = JOINT_NAMES[:5]  # gripper 제외 5관절 (build_arm_kinematics와 동일 순서)
GRIPPER_JOINT_NAME = JOINT_NAMES[5]
assert GRIPPER_JOINT_NAME == "gripper"


class _NoOpCam:
    """EpisodeTicker.dual_cam과 인터페이스를 맞추기 위한 더미 — 실로봇은 카메라가 SOFollower에
    묶여 있어 에피소드마다 새로 열고 닫을 게 없다(세션 전체에서 한 번만 connect/disconnect)."""

    def close(self) -> None:
        pass


class RealRobotEpisodeTicker:
    """EpisodeTicker(시뮬)와 동일한 tick() 계약을 구현 — robot/kin은 세션 시작 시 한 번만 만들어
    여러 에피소드에 걸쳐 재사용한다(시뮬처럼 에피소드마다 MjModel을 새로 로드하지 않음 — 실제
    팔의 물리적 현재 위치가 곧 다음 에피소드의 시작점이라 "리셋"이라는 개념이 없다)."""

    def __init__(
        self,
        args: argparse.Namespace,
        robot: SOFollower,
        kin: RobotKinematics,
        scene: str = "real_robot",
        variant: int = 0,
    ):
        self.args = args
        self.robot = robot
        self.kin = kin
        self.scene = scene
        self.variant = variant
        self.dual_cam = _NoOpCam()

        self.dt = 1.0 / args.fps
        self.max_steps = int(args.episode_seconds * args.fps) if args.episode_seconds else None
        self.step = 0
        self.bead_points: list = []  # 실로봇엔 비드 시뮬레이션 없음 — 인터페이스 호환용 항상 빈 리스트
        self.last_contact = False  # 실로봇엔 접촉 센서 없음 — 인터페이스 호환용 항상 False
        self.last_trigger = 0.0
        self.prev_pose: np.ndarray | None = None

        obs = robot.get_observation()
        arm_deg = np.array([obs[f"{j}.pos"] for j in ARM_JOINT_NAMES], dtype=float)
        self._last_arm_deg = arm_deg  # IK 초기 추정치로 매 틱 재사용 — 해가 연속적으로 나오게
        self.current_gripper_pos = float(obs[f"{GRIPPER_JOINT_NAME}.pos"])

        T0 = kin.forward_kinematics(arm_deg)
        pose0 = pose_to_xyzrotvec(T0)
        self.target_pos = pose0[:3].copy()
        self.R_cmd = Rotation.from_rotvec(pose0[3:6]).as_matrix()
        self.last_tip = self.target_pos.copy()

        self.last_wrist_frame = obs.get("wrist")
        self.last_overview_frame = obs.get("overview", self.last_wrist_frame)

    def tick(
        self,
        ctl: JoystickEEController | KeyboardEEController,
        dataset,
        force_end: bool,
        force_discard: bool,
    ) -> str | None:
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
        self.target_pos[2] = max(self.target_pos[2], MIN_TIP_Z)  # 실제 작업대 충돌 방지(안전)

        wx, wy, wz = ctl.rotation_rate(
            max_angular=self.args.max_angular_speed, invert_x=self.args.invert_roll, invert_y=self.args.invert_pitch
        )
        if wx or wy or wz:
            self.R_cmd = Rotation.from_rotvec(np.array([wx, wy, wz]) * self.dt).as_matrix() @ self.R_cmd
        # yaw(wz)는 넣어도 IK가 5DOF라 못 푼다 — ai_layer/kinematics.py의 zero_yaw()와 같은 이유
        # (PAC_Supermoon 실로봇 URDF 기준). position_weight 위주로 풀리며 yaw는 자연히 무시됨.

        bit = ctl.gripper_bit()
        self.last_trigger = bit
        self.current_gripper_pos = 100.0 if bit else 0.0  # 단순 열림/닫힘(0~100)

        target_T = np.eye(4)
        target_T[:3, :3] = self.R_cmd
        target_T[:3, 3] = self.target_pos
        arm_deg = self.kin.inverse_kinematics(self._last_arm_deg, target_T)
        self._last_arm_deg = arm_deg

        action = {f"{name}.pos": float(val) for name, val in zip(ARM_JOINT_NAMES, arm_deg)}
        action[f"{GRIPPER_JOINT_NAME}.pos"] = self.current_gripper_pos
        self.robot.send_action(action)  # max_relative_target(설정돼 있으면)이 여기서 과도한 이동을 막는다

        obs = self.robot.get_observation()
        self.last_wrist_frame = obs.get("wrist")
        self.last_overview_frame = obs.get("overview", self.last_wrist_frame)

        actual_arm_deg = np.array([obs[f"{j}.pos"] for j in ARM_JOINT_NAMES], dtype=float)
        T_actual = self.kin.forward_kinematics(actual_arm_deg)
        pose = pose_to_xyzrotvec(T_actual)  # 명령이 아니라 실측 — 센서 노이즈/지연까지 그대로 기록됨
        self.last_tip = pose[:3].copy()

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
