# PAC_Supermoon 실로봇 URDF

출처: `Minje0420/PAC_Supermoon` 브랜치 `codex/add-so101-final-effector`.

- 파일: `so_arm_d405.urdf` (PAC_Supermoon 원본 이름은 `so_arm_with_gopro_final.urdf`였으나 D405 홀더라 여기선 바꿈)
- `actioncam-vio`는 쓰지 않음. 예전 유사 프로젝트 참고용.
- FK 타깃: `tcp_link` (`root → tcp` z = -0.14 m)
- 손목 카메라: `camera_holder_405_final_so101_aligned.stl`
- 그리퍼: UMI, `gripper` + mimic slider. 시연에서 집기를 안 써도 action 채널은 유지.

AI FK는 `ai_layer/kinematics.py`가 이 폴더를 사용한다.
