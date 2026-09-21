# ai_layer/tools

## LeRobot 환경 (`/home/dy/pac2026/env_lerobot`)

2026-09-21 기준 검증된 설치 순서. Isaac 환경(`env_isaaclab`)과 별도.

```bash
cd /home/dy/pac2026
uv venv env_lerobot --python 3.11
export VIRTUAL_ENV=/home/dy/pac2026/env_lerobot
uv pip install "lerobot[kinematics,feetech,intelrealsense]==0.4.4"
# placo/pin 휠이 urdfdom 4·tinyxml2 10에 링크되어 있어 아래 두 개는 내려야 import 됨
uv pip install "cmeel-urdfdom>=4,<5" "cmeel-tinyxml2>=10,<11"
# seam_cv 의존성 (lerobot 기본 설치에 없음)
uv pip install "scipy>=1.11" "scikit-image>=0.22"
```

검증된 버전: lerobot 0.4.4, placo 0.9.16, pin 3.4.0, cmeel-urdfdom 4.0.1, cmeel-tinyxml2 10.0.0, torch 2.10.0+cu128.

## fk_smoke.py

```bash
cd /home/dy/pac2026/pac2026-team
PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/tools/fk_smoke.py
```

확인 항목: URDF 로드 + `tcp_link` 탐색, `gripper_link→tcp_link` 0.15 m, 7차원 델타 변환, yaw 고정, 델타 누적 복원.
URDF 중립 자세 self-collision 경고는 민제씨 URDF의 충돌 메시 문제이며 FK에는 영향 없음.

## bc_synthetic_test.py — BC 경로 끝-끝 실행 검증 (실로봇 없이)

```bash
cd /home/dy/pac2026/pac2026-team
PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/tools/bc_synthetic_test.py
```

so101_follower 녹화와 같은 키/단위의 가짜 LeRobotDataset(2 에피소드 × 70 프레임, 30 fps, 검은 선 이미지)을
임시 폴더에 만들고: 데이터셋 변환(state 9D, action 32×7, yaw=0, 그리퍼 0~100, seam 특징) → 정규화 통계 →
ACT 6스텝 학습 → 체크포인트 저장(정책+전/후처리) → `bc_inference.load_bc_checkpoint`로 복원 → 청크 예측
단위 확인까지 한 번에 돈다. 2026-09-21 통과. 코드를 고치면 이걸 먼저 돌릴 것.
