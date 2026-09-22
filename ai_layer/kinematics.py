"""PAC_Supermoon URDF 기반 FK/델타 변환.

하드웨어 근거: Minje0420/PAC_Supermoon 브랜치 `codex/add-so101-final-effector`
(D405 홀더, TCP=`tcp_link`). actioncam-vio는 적용하지 않는다.
제어 규약: 송지수 ActionChunk — 직전 스텝 대비 증분 EEF-delta, 회전은 회전벡터
(월드 왼쪽 곱). 그리퍼 채널은 유지한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "pac_supermoon"
URDF_PATH = ASSETS_DIR / "so_arm_d405.urdf"
TARGET_FRAME_NAME = "tcp_link"

# so101_follower/so101_leader 공통 관절 순서 (lerobot 표준). gripper 포함.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# lerobot 관절명 -> IsaacLab/USD 관절명 (assets/so101_isaac/so101_cfg.py 기준).
# Isaac USD는 아직 기본 SO-101. 실로봇 URDF와 끝단이 다르니 시뮬 자산은 후속 작업.
ISAAC_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
LEROBOT_TO_ISAAC_JOINT = dict(zip(JOINT_NAMES, ISAAC_JOINT_NAMES))


def build_kinematics(urdf_path: Path | str = URDF_PATH) -> RobotKinematics:
    return RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=TARGET_FRAME_NAME,
        joint_names=JOINT_NAMES,
    )


def build_arm_kinematics(urdf_path: Path | str = URDF_PATH) -> RobotKinematics:
    """IK 전용: 그리퍼(6번째 관절)를 제외한 5개 팔 관절만 풀이 대상으로 삼는다.

    그리퍼는 EE 위치에 영향을 주지 않는 독립 DOF라 IK에 섞으면 솔버가 임의로 움직일 수 있어
    분리한다 — RL env에서 그리퍼는 discrete action으로 별도 제어 (so101_seam_env.py 참고).
    """
    return RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=TARGET_FRAME_NAME,  # 실로봇 tcp_link (병합 전 "gripper_frame_link"은 PAC_Supermoon URDF에 없음)
        joint_names=JOINT_NAMES[:5],
    )


def apply_pose_delta(T: np.ndarray, delta_pos: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """현재 pose(4x4)에 world-frame 기준 위치/회전 delta를 적용한 목표 pose(4x4)를 반환.

    RL env의 EEF-delta 액션(dx,dy,dz,drx,dry,drz)을 IK 목표로 변환할 때 사용.
    """
    T_new = T.copy()
    T_new[:3, 3] = T[:3, 3] + delta_pos
    delta_R = Rotation.from_rotvec(delta_rotvec).as_matrix()
    T_new[:3, :3] = delta_R @ T[:3, :3]
    return T_new


def pose_to_xyzrotvec(T: np.ndarray) -> np.ndarray:
    """4x4 변환행렬 -> [x, y, z, rx, ry, rz] (rotation vector, world frame)."""
    pos = T[:3, 3]
    rotvec = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec])


def rotvec_to_rot6d(rotvec: np.ndarray) -> np.ndarray:
    """회전벡터 (3,) -> 6D 회전 표현 (6,) = 회전행렬의 첫 열 + 둘째 열.

    회전벡터는 회전각이 180°를 지나면 방향이 뒤집혀 값이 불연속으로 튄다. 6D 표현은 연속이라
    신경망 입력(observation.state)에 적합하다. 셋째 열은 앞 두 열의 외적으로 복원된다.
    """
    R = Rotation.from_rotvec(np.asarray(rotvec, dtype=float).reshape(3)).as_matrix()
    return np.concatenate([R[:, 0], R[:, 1]])


def rot6d_to_rotvec(rot6d: np.ndarray) -> np.ndarray:
    """6D 회전 표현 (6,) -> 회전벡터 (3,). Gram-Schmidt로 직교화 후 행렬 복원."""
    v = np.asarray(rot6d, dtype=float).reshape(6)
    a1, a2 = v[:3], v[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / max(np.linalg.norm(a2), 1e-8)
    b3 = np.cross(b1, b2)
    return Rotation.from_matrix(np.stack([b1, b2, b3], axis=1)).as_rotvec()


def pose_to_state(pose_xyzrotvec: np.ndarray) -> np.ndarray:
    """절대 pose (6,) [xyz, rotvec] -> observation.state (9,) [xyz, rot6d]."""
    p = np.asarray(pose_xyzrotvec, dtype=float).reshape(6)
    return np.concatenate([p[:3], rotvec_to_rot6d(p[3:6])])


def state_to_pose(state: np.ndarray) -> np.ndarray:
    """observation.state (9,) -> 절대 pose (6,) [xyz, rotvec]."""
    s = np.asarray(state, dtype=float).reshape(9)
    return np.concatenate([s[:3], rot6d_to_rotvec(s[3:9])])


def pose_delta(from_pose: np.ndarray, to_pose: np.ndarray) -> np.ndarray:
    """절대 pose (6,) -> 증분 (6,).

    위치: p_k = p_{k-1} + dp
    회전: R_k = Exp(w) @ R_{k-1}  (월드 기준 왼쪽 곱, 송지수 ActionChunk)
    """
    from_pose = np.asarray(from_pose, dtype=float).reshape(6)
    to_pose = np.asarray(to_pose, dtype=float).reshape(6)
    dp = to_pose[:3] - from_pose[:3]
    r_from = Rotation.from_rotvec(from_pose[3:6]).as_matrix()
    r_to = Rotation.from_rotvec(to_pose[3:6]).as_matrix()
    w = Rotation.from_matrix(r_to @ r_from.T).as_rotvec()
    return np.concatenate([dp, w])


# 회전 증분 (rx, ry, rz) 중 yaw(월드 z축 회전) 인덱스. 델타 7차원 기준 5번.
YAW_INDEX = 5


def zero_yaw(delta: np.ndarray) -> np.ndarray:
    """(T, 6 or 7) 증분에서 yaw 성분(drz)을 0으로 만든 사본을 돌려준다.

    민제씨 PAC_Supermoon 실로봇은 5관절이라 IK가 5D(XYZ + roll/pitch)만 푼다.
    HAL이 yaw를 버리므로 학습 타깃에서도 0으로 고정한다 (2026-09-21 결정).
    """
    out = np.array(delta, dtype=float, copy=True)
    out[..., YAW_INDEX] = 0.0
    return out


def joint_traj_to_eef_pose_traj(
    kin: RobotKinematics, joint_traj_deg: np.ndarray
) -> np.ndarray:
    """(T, 6) 관절각(deg) 시퀀스 -> (T, 6) EEF pose(m, rotvec) 시퀀스.

    placo(RobotWrapper.set_joint)는 C++ double만 받는다. LeRobotDataset은 float32 텐서를 주므로
    여기서 반드시 float64로 캐스팅한다 (안 하면 Boost.Python.ArgumentError).
    """
    joint_traj_deg = np.asarray(joint_traj_deg, dtype=np.float64)
    poses = np.zeros((joint_traj_deg.shape[0], 6), dtype=float)
    for t in range(joint_traj_deg.shape[0]):
        T = kin.forward_kinematics(joint_traj_deg[t])
        poses[t] = pose_to_xyzrotvec(T)
    return poses


def eef_pose_traj_to_delta_traj(
    eef_pose_traj: np.ndarray, anchor_pose: np.ndarray | None = None
) -> np.ndarray:
    """(T, 6) 절대 pose -> (T, 6) 증분 EEF-delta.

    anchor_pose가 있으면 delta[0] = anchor(관측 시점 t의 실제 pose) -> pose[0].
    호출 측(so101_bc_dataset)이 pose[0]을 t+dt 시점의 명령으로 넣으므로 delta[0]은
    "지금 위치에서 33ms 뒤 목표까지"가 된다 (ActionChunk 스텝 0 의미와 일치).
    없으면 delta[0] = 0 이고 이후는 프레임 간 증분.
    """
    eef_pose_traj = np.asarray(eef_pose_traj, dtype=float)
    delta = np.zeros_like(eef_pose_traj)
    if eef_pose_traj.shape[0] == 0:
        return delta
    if anchor_pose is not None:
        delta[0] = pose_delta(anchor_pose, eef_pose_traj[0])
    for t in range(1, eef_pose_traj.shape[0]):
        delta[t] = pose_delta(eef_pose_traj[t - 1], eef_pose_traj[t])
    return delta


def joint_traj_to_eef_delta_traj(
    kin: RobotKinematics,
    joint_traj_deg: np.ndarray,
    gripper_signal: np.ndarray,
    anchor_pose: np.ndarray | None = None,
) -> np.ndarray:
    """관절공간 궤적 -> (dx, dy, dz, drx, dry, drz, gripper) 시퀀스.

    gripper는 통과시킨다. LeRobot 0.4.x SO-101 팔로워는 그리퍼를 0~100(RANGE_0_100)으로
    정규화해서 녹화한다 (모터 로우값 1000~4000이 아님). 실로봇(민제씨) 캘리브 값으로의
    변환은 추론 출력 단계에서 별도로 한다 (7번째 채널 규약은 팀 회의에서 결정 예정).
    """
    eef_poses = joint_traj_to_eef_pose_traj(kin, joint_traj_deg)
    deltas = eef_pose_traj_to_delta_traj(eef_poses, anchor_pose=anchor_pose)
    return np.concatenate([deltas, gripper_signal.reshape(-1, 1)], axis=1)
