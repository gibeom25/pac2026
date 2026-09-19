"""SO-101 URDF 기반 FK/델타 변환 유틸리티.

설계 원칙(docs/AI_추론계층_프레임워크.md 0절 "EEF-delta 통일")에 따라, lerobot이
기록하는 관절공간(joint-space) 데이터를 EEF-delta 시퀀스로 변환하는 데 쓴다.
FK/IK 자체는 lerobot.model.kinematics.RobotKinematics(placo 기반)를 그대로 사용하고,
여기서는 이 프로젝트의 URDF/관절 이름에 맞춘 얇은 래퍼 + delta 계산만 담당한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "so101"
URDF_PATH = ASSETS_DIR / "so101_new_calib.urdf"

# so101_follower/so101_leader 공통 관절 순서 (lerobot 표준)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# lerobot 관절명 -> IsaacLab/USD 관절명 (assets/so101_isaac/so101_cfg.py 기준).
# 같은 순서(JOINT_NAMES)로 대응되므로 zip(JOINT_NAMES, ISAAC_JOINT_NAMES)으로 매핑한다.
# 상세: assets/so101_isaac/README.md "관절 이름 매핑" 표 참고.
ISAAC_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
LEROBOT_TO_ISAAC_JOINT = dict(zip(JOINT_NAMES, ISAAC_JOINT_NAMES))


def build_kinematics(urdf_path: Path | str = URDF_PATH) -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES,
    )


def pose_to_xyzrotvec(T: np.ndarray) -> np.ndarray:
    """4x4 변환행렬 -> [x, y, z, rx, ry, rz] (rotation vector, world frame 기준)."""
    pos = T[:3, 3]
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec])


def joint_traj_to_eef_pose_traj(
    kin: RobotKinematics, joint_traj_deg: np.ndarray
) -> np.ndarray:
    """(T, 6) 관절각(deg) 시퀀스 -> (T, 6) EEF pose(m, rotvec) 시퀀스."""
    poses = np.zeros((joint_traj_deg.shape[0], 6), dtype=float)
    for t in range(joint_traj_deg.shape[0]):
        T = kin.forward_kinematics(joint_traj_deg[t])
        poses[t] = pose_to_xyzrotvec(T)
    return poses


def eef_pose_traj_to_delta_traj(eef_pose_traj: np.ndarray) -> np.ndarray:
    """(T, 6) 절대 pose 시퀀스 -> (T, 6) EEF-delta 시퀀스 (첫 프레임 delta=0).

    회전은 rotvec 단순 차분으로 근사한다 — 두 자세 사이 회전량이 작다는 전제(제어
    주기/AI 시퀀스 스텝 간격이 충분히 촘촘하다는 전제, docs 4.4절 dt_AI=20ms 참고)
    하에서 유효하다. 프레임 간 회전이 커지는 경우(끊긴 데모, 저주파 샘플링) 별도
    쿼터니언 기반 relative-rotation 계산으로 교체할 것.
    """
    delta = np.zeros_like(eef_pose_traj)
    delta[1:] = eef_pose_traj[1:] - eef_pose_traj[:-1]
    return delta


def joint_traj_to_eef_delta_traj(
    kin: RobotKinematics, joint_traj_deg: np.ndarray, gripper_signal: np.ndarray
) -> np.ndarray:
    """관절공간 궤적 -> (dx, dy, dz, drx, dry, drz, gripper_signal[0/1]) 시퀀스.

    gripper_signal은 그대로 통과시킨다 (docs 2장 인터페이스: 델타가 아니라 0/1 상태 신호).
    """
    eef_poses = joint_traj_to_eef_pose_traj(kin, joint_traj_deg)
    deltas = eef_pose_traj_to_delta_traj(eef_poses)
    return np.concatenate([deltas, gripper_signal.reshape(-1, 1)], axis=1)
