"""실로봇(SO-101 follower) EE-native 틱 루프.

ai_layer/tools/episode_ticker.py의 `EpisodeTicker`(시뮬)와 **똑같은 공개 인터페이스**를
구현한다 — `tick(ctl, dataset, force_end, force_discard)`, `scene`/`variant`/`step`/
`last_wrist_frame`/`last_overview_frame`/`last_contact`/`last_trigger`/`target_pos`/
`last_tip`/`bead_points`/`dual_cam`(.close() 가능). 그래서 GUI(ai_layer/gui/collect_tab.py)나
CLI 쪽 틱 호출 코드는 시뮬 EpisodeTicker와 이 RealRobotEpisodeTicker 중 뭘 들고 있는지 신경
쓸 필요가 없다 — "source" 선택만 다르고 나머지 흐름(에피소드 시작/저장/폐기, 상태 표시)은 동일.

**조이스틱/키보드 컨트롤러는 손댈 필요가 없다** — JoystickEEController/KeyboardEEController/
QtKeyboardEEController는 애초에 "EE 속도(m/s, rad/s)"만 돌려주는 하드웨어 중립 인터페이스라서
시뮬 mocap이든 실로봇이든 똑같이 넣어주면 된다(ee_velocity()/rotation_rate()/gripper_bit()).

**데이터셋 스키마는 시뮬과 완전히 동일하다**(2026-10-06 결정 — BC/RL 학습 파이프라인을 안
바꾸기 위해) — record_mujoco.py의 STATE_KEYS/ACTION_KEYS/CAMERA_HW를 그대로 쓰고, "wrist"
카메라 키도 동일하게 쓴다. 실로봇 쪽엔 당연히 "bead"(용접 비드) 시뮬레이션이나 접촉 센서가
없으므로 bead_points는 항상 빈 리스트, last_contact는 항상 False로 둔다(데이터셋 필드 자체에
비드/접촉은 안 들어가므로 영향 없음 — 그냥 GUI 상태 표시용 필드).

**2026-10-06(2차) — 제어 방식 변경**: 처음엔 이 ticker가 직접 IK를 풀어서 관절각을
`SOFollower.send_action()`으로 보냈는데, "실로봇 연결시 무조건 EEF delta 기준으로 전달하게
하고 제어단은 나중에 구현"이라는 결정으로 바꿨다 — ai_layer/kinematics.py가 이미 전제하고
있던 팀 규약(모듈 docstring: "제어 규약: 송지수 ActionChunk — 직전 스텝 대비 증분 EEF-delta",
zero_yaw()가 있는 이유도 실제 제어단이 5DOF EEF-delta만 받기 때문)과 일치시킨 것이다. 그래서
이 ticker는 더 이상 로컬 IK로 관절을 직접 구동하지 않는다 — 조이스틱/키보드 입력을 적분해
"이번 틱에 원하는 EEF-delta"만 계산해서 `_send_eef_delta()`에 넘기고, 그 함수는 지금은
**아무 것도 안 하는 자리표시자**다(제어단이 아직 없음 — TODO, 연결되면 여기서 실제로 전송).
즉 **지금 이 코드로는 실로봇이 실제로 움직이지 않는다** — 데이터 수집 파이프라인(관측 읽기 +
델타 계산 + 데이터셋 기록)의 틀만 미리 맞춰둔 것이고, 모터 구동은 제어단이 붙은 뒤의 일이다.

관측(카메라 프레임, 현재 관절각)은 계속 `robot.get_observation()`으로 읽는다 — 이건 "제어"가
아니라 센서 읽기라 그대로 둔다. 현재 pose는 FK(`kin.forward_kinematics`)로 구하고, 이걸
기준으로 "이번 틱 목표까지의 델타"를 계산해 액션으로 기록한다 — 실제로 로봇이 안 움직이므로
관측 pose 자체는 틱마다 거의 안 바뀌고(제어단 연결 전까지는), 액션(델타)만 사용자의 조작
의도를 담는다. 제어단이 붙으면 관측도 매 틱 실제로 움직인 결과를 반영하게 된다.
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
from ai_layer.tools.record_mujoco import ACTION_KEYS, STATE_KEYS

# 2026-10-06: WORKSPACE_X/Y/Z(작업공간 클리핑)와 MIN_TIP_Z(바닥/작업대 높이 제한)는 일부러 안
# 가져왔다 — record_mujoco.py에 있는 그 값들은 시뮬 EE-rig 기준으로 잡은 것이라 실제 SO-101의
# 물리적 도달 범위/작업대 높이와 안 맞을 수 있다("적용 못하는 constraint는 알아서 꺼지게"
# 요청). 틀린 숫자로 "제한 걸려 있는 것처럼" 보이는 게 더 위험하다 — 진짜 안전 제한(실제
# 도달 범위, 테이블 높이)은 나중에 붙는 제어단이 실제 로봇 기준으로 걸어야 한다.

ARM_JOINT_NAMES = JOINT_NAMES[:5]  # gripper 제외 5관절 (FK 입력 순서, build_arm_kinematics와 동일)
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

        obs = robot.get_observation()
        arm_deg = np.array([obs[f"{j}.pos"] for j in ARM_JOINT_NAMES], dtype=float)
        T0 = kin.forward_kinematics(arm_deg)
        pose0 = pose_to_xyzrotvec(T0)
        self.current_pose = pose0  # 매 틱 관측(FK)로 갱신 — 액션(델타)의 기준점
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
        # 이동 방향은 EE 자신의 현재 자세(R_cmd) 기준 — 시뮬(episode_ticker.py)과 동일 규약.
        # 작업공간 클리핑/바닥 높이 제한은 일부러 안 건다(모듈 docstring 상단 참고) — 시뮬
        # 기준 숫자를 실제 로봇에 잘못 적용하느니 아예 안 거는 쪽이 안전하다.
        self.target_pos = self.target_pos + self.R_cmd @ (np.array([vx, vy, vz]) * self.dt)

        wx, wy, wz = ctl.rotation_rate(
            max_angular=self.args.max_angular_speed, invert_x=self.args.invert_roll, invert_y=self.args.invert_pitch
        )
        if wx or wy or wz:
            self.R_cmd = Rotation.from_rotvec(np.array([wx, wy, wz]) * self.dt).as_matrix() @ self.R_cmd
        # yaw(wz)는 넣어도 실로봇 제어단이 5DOF(XYZ+roll/pitch)만 받는다 — ai_layer/kinematics.py의
        # zero_yaw()와 같은 이유(PAC_Supermoon 실로봇 URDF 기준, 실제 관절이 yaw를 못 냄).

        bit = ctl.gripper_bit()
        self.last_trigger = bit

        target_pose = np.concatenate([self.target_pos, Rotation.from_matrix(self.R_cmd).as_rotvec()])
        delta6 = pose_delta(self.current_pose, target_pose)  # "이번 틱에 원하는" EEF-delta
        self._send_eef_delta(delta6, bit)

        obs = self.robot.get_observation()  # 센서 읽기 — 제어가 아니므로 그대로 둠
        self.last_wrist_frame = obs.get("wrist")
        self.last_overview_frame = obs.get("overview", self.last_wrist_frame)

        actual_arm_deg = np.array([obs[f"{j}.pos"] for j in ARM_JOINT_NAMES], dtype=float)
        T_actual = self.kin.forward_kinematics(actual_arm_deg)
        pose = pose_to_xyzrotvec(T_actual)  # 실측 pose — 제어단이 붙기 전까진 거의 안 바뀜(기대된 동작)
        self.current_pose = pose
        self.last_tip = pose[:3].copy()

        if dataset is not None:
            state9 = pose_to_state(pose)
            obs_values = {**dict(zip(STATE_KEYS, state9.tolist())), "wrist": self.last_wrist_frame}
            if "observation.images.overview" in dataset.features:  # 고정 카메라를 연결했을 때만 저장
                obs_values["overview"] = obs["overview"]
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

    def _send_eef_delta(self, delta6: np.ndarray, gripper_bit: float) -> None:
        """실로봇 제어단(모터 구동)에 EEF-delta를 전달하는 자리 — 2026-10-06: 제어단이 아직
        없어서(나중에 구현 예정) 지금은 아무 것도 안 한다. 제어단이 붙으면 여기서 실제 전송
        (예: control_bridge의 ActionChunk 프로토콜)하면 된다 — 그 전까지는 로봇이 실제로
        움직이지 않고, 조작 의도(델타)만 데이터셋에 기록된다."""

    def _finish(self, dataset, discard: bool) -> str:
        if discard:
            if dataset is not None:
                dataset.clear_episode_buffer()
            return "discarded"
        if dataset is not None:
            dataset.save_episode()
        return "saved"
