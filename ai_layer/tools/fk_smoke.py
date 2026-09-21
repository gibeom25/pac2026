"""PAC_Supermoon URDF FK 스모크 테스트.

실행: cd pac2026-team && PYTHONPATH=. python ai_layer/tools/fk_smoke.py

확인하는 것:
  1. LeRobot RobotKinematics가 실로봇 URDF를 읽고 tcp_link를 찾는가
  2. 관절 전부 0일 때 gripper_link 대비 tcp_link 오프셋이 URDF 값(약 0.15 m)과 맞는가
  3. 작은 관절 변화 -> 증분 델타 변환이 돌아가고, yaw 고정이 적용되는가
"""
from __future__ import annotations

import numpy as np

from ai_layer.configs.so101_act_bc import DT_AI_SEC, ZERO_YAW
from ai_layer.kinematics import (
    JOINT_NAMES,
    TARGET_FRAME_NAME,
    URDF_PATH,
    build_kinematics,
    joint_traj_to_eef_delta_traj,
    joint_traj_to_eef_pose_traj,
    pose_to_state,
    state_to_pose,
    zero_yaw,
)
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

np.set_printoptions(precision=4, suppress=True)

print(f"URDF      : {URDF_PATH}")
print(f"target    : {TARGET_FRAME_NAME}")
print(f"joints    : {JOINT_NAMES}")
print(f"dt_AI     : {DT_AI_SEC*1000:.1f} ms, ZERO_YAW={ZERO_YAW}")

kin_tcp = build_kinematics()
kin_grip = RobotKinematics(urdf_path=str(URDF_PATH), target_frame_name="gripper_link", joint_names=JOINT_NAMES)

zero = np.zeros((1, 6))
pose_tcp = joint_traj_to_eef_pose_traj(kin_tcp, zero)[0]
pose_grip = joint_traj_to_eef_pose_traj(kin_grip, zero)[0]
print("\n[1] 관절 전부 0")
print("  tcp_link     xyz+rotvec:", pose_tcp)
print("  gripper_link xyz+rotvec:", pose_grip)
dist = np.linalg.norm(pose_tcp[:3] - pose_grip[:3])
print(f"  gripper_link -> tcp_link 거리: {dist:.4f} m  (URDF 기대값: 0.01 + 0.14 = 0.15)")
assert abs(dist - 0.15) < 1e-3, "tcp_link 오프셋이 URDF와 다름"

print("\n[2] 작은 관절 궤적 -> 증분 델타 (7차원)")
T = 5
traj = np.zeros((T, 6))
traj[:, 0] = np.linspace(0, 10, T)   # shoulder_pan 0->10 deg
traj[:, 1] = np.linspace(0, -5, T)   # shoulder_lift
traj[:, 5] = np.linspace(20, 40, T)  # gripper (그냥 통과)
gripper_signal = traj[:, 5]
anchor = joint_traj_to_eef_pose_traj(kin_tcp, np.zeros((1, 6)))[0]
deltas = joint_traj_to_eef_delta_traj(kin_tcp, traj, gripper_signal, anchor_pose=anchor)
if ZERO_YAW:
    deltas = zero_yaw(deltas)
print(deltas)
assert deltas.shape == (T, 7)
if ZERO_YAW:
    assert np.all(deltas[:, 5] == 0), "yaw 고정 실패"
assert np.allclose(deltas[:, 6], gripper_signal), "그리퍼 채널 훼손"

print("\n[3] 델타 누적 복원 검사: 앵커 + 델타 합 == 마지막 절대 위치?")
poses = joint_traj_to_eef_pose_traj(kin_tcp, traj)
recon = anchor[:3] + deltas[:, :3].sum(axis=0)
print("  복원 xyz:", recon, " 실제 xyz:", poses[-1, :3])
assert np.allclose(recon, poses[-1, :3], atol=1e-6)

print("\n[4] state(9D rot6d) 왕복 변환")
st = pose_to_state(pose_tcp)
back = state_to_pose(st)
print("  state:", st, " 복원 pose:", back)
assert st.shape == (9,) and np.allclose(back[:3], pose_tcp[:3]) and np.allclose(
    Rotation.from_rotvec(back[3:]).as_matrix(), Rotation.from_rotvec(pose_tcp[3:]).as_matrix(), atol=1e-6)

print("\nFK SMOKE OK")
