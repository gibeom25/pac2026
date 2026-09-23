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
# RL (MuJoCo 환경, 2026-09-22 기범 선배님 전환)
uv pip install "mujoco>=3.1" "gymnasium>=0.29"
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

## chunk_bridge_test.py — AI → 제어 규약 검증

```bash
cd /home/dy/pac2026/pac2026-team
PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/tools/chunk_bridge_test.py \
    --upstream /path/to/PAC_Supermoon   # jisu/control-layer clone (선택, 있으면 원본 코덱과 바이트 대조)
```

`ai_layer/control_bridge/`(규약 복사본, 청크 조립, 스냅샷 변환) 검사. 원본 clone을 주면 송지수 선배 코덱으로
우리 바이트를 읽고/다시 써서 동일한지, 원본 `ChunkValidator`(policy 1, v_max 0.15)가 통과시키는지까지 본다.

## AI 노드 실행 (`ai_layer/control_bridge/ai_node.py`)

```bash
# 배관 점검 (모델/로봇/카메라 없이)
PYTHONPATH=. python ai_layer/control_bridge/ai_node.py --dry-run --fake-snapshot --iterations 3
# 제어 계층과 실제 연결 (제어 쪽: control/tools/run_live.py --ai external)
PYTHONPATH=. python ai_layer/control_bridge/ai_node.py --checkpoint outputs/bc_act/last --camera realsense
```

옵션: `--anchor obs|commit`(기본 commit = L1), `--eef-mode off|on|gripper_threshold|from_channel`(회의 전 기본 off),
`--policy-id 1`(seam-welding), `--n-steps N`(청크 앞부분만), `--min-period 초`.

## 로봇 오는 날 준비물 (2026-09-21)

| 파일 | 용도 |
|---|---|
| `RECORD_DAY.md` | 당일 체크리스트 (연결 → 녹화 → 점검 → 학습 → 제어 연결) |
| `record_so101.sh` | `lerobot-record` 명령 템플릿. 포트/카메라 시리얼만 채우면 됨. fps 30, 카메라 이름 wrist, 320×240 고정 |
| `check_dataset.py` | 녹화 직후 점검: fps, 관절 순서, 이미지 키, 단위(deg/0~100), timestamp, 증분 크기 vs 제어 한계, 선 인식률 |
| `seam_preview.py` | 휴대폰 사진으로 선 인식 미리 튜닝. `--no-invert` = 밝은 선(분필/밝은 실리콘) |

```bash
PYTHONPATH=. python ai_layer/tools/check_dataset.py --repo-id <repo> --root <path>
PYTHONPATH=. python ai_layer/tools/seam_preview.py photos/*.jpg --out preview/ [--no-invert]
```

## rl_env_smoke.py — MuJoCo RL 환경 스모크 (2026-09-22)

```bash
PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/tools/rl_env_smoke.py [--bc-checkpoint outputs/bc_act/last]
```

home 자세, 2초 드리프트, 경로 영역 안 이동/회전 추종(정지 후 ±1 cm/±0.03 rad), BC teacher 보상, 도달 한계 경고. 통과 기준은
경로 영역(x 0.15~0.35, z 0.05~0.10) 안이다. x 0.45 같은 한계 근처 목표는 5DOF 팔이 물리적으로 못 가므로 검사하지 않는다.

## record_mujoco.py — SO-101 leader → MuJoCo follower 미러링 데이터 수집 (2026-09-22, 기범)

실물 팔로워/카메라 없이 **leader 하나만으로** BC 학습용 LeRobotDataset을 만든다. leader가 읽은 관절각을
MuJoCo(`assets/so101/scene_a4.xml` = `so101_new_calib_camera.xml`(손목 카메라 포함) + 바닥 +
용접선이 그려진 A4 용지, `textures/a4_weld_seam.png`)의 팔로워에 그대로 명령하고, 물리 스텝 후 실제
도달한 관절각(observation.state) + 렌더링된 손목 이미지 + 명령값(action)을 기록한다. 관절공간 그대로
저장하므로 실물 `lerobot-record` 결과물과 포맷이 동일 — `train_bc.py`에 바로 사용 가능.

A4 용지는 세계좌표 x=0.25 중심(so101_seam_env.py 경로 영역 x 0.15~0.35가 안쪽에 들어옴), 긴 축(297mm)이
x축. 배치는 근사값 — 손목 카메라 extrinsic이 아직 근사 배치라(위 캐비어트 참고) 홈 자세에서는 카메라가
용지를 바로 보지 않을 수 있음, 뷰어로 확인 후 필요하면 카메라/용지 위치 조정할 것.

```bash
PYTHONPATH=. python ai_layer/tools/record_mujoco.py \
    --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm \
    --repo-id <user>/so101-mujoco-demo --root ./datasets/so101-mujoco-demo \
    --num-episodes 5 --episode-seconds 15
```

에피소드 사이 Enter로 다음 녹화 시작(리더를 시작 자세로 되돌릴 시간). 카메라 extrinsic(`wrist` 카메라의
pos/quat)은 실측 캘리브레이션이 아니라 근사 배치 — 실물과 비교해 보정 필요.

**그리퍼는 2026-09-23부터 구동하지 않는다.** 설계문서 2절의 `gripper_signal[0/1]`을 그대로 쓰기로
해서, MuJoCo 그리퍼 조는 고정값에 묶어두고 리더의 raw 그리퍼 값(`MotorNormMode.RANGE_0_100`, 0~100,
각도 아님)을 임계값 50으로 이진화해 action/observation의 그리퍼 채널에 그대로 기록한다
(`gripper_bit()`). 실물 대체 도구(5cm 막대 + LED, `assets/so101/so101_new_calib_camera.xml`의
`tool_rod`/`tool_led`)로 뷰어에서 켜짐/꺼짐을 눈으로 확인 가능. 기본은 raw>=50 -> 1(켜짐) —
반대면 `--gripper-invert`.

## check_gripper.py — 그리퍼 이진 신호(LED) 확인 (2026-09-23, 기범)

record_mujoco.py로 전체 녹화를 돌리지 않고 팔 5관절 미러링 + 그리퍼 LED 동작만 빠르게 확인. 리더를
움직이면 뷰어 속 팔이 따라 움직이고, 그리퍼를 반 이상 닫으면 LED(`tool_led`)가 켜진다.

```bash
PYTHONPATH=. python ai_layer/tools/check_gripper.py \
    --leader-port /dev/ttyACM0 --leader-id my_awesome_leader_arm
```

리더 그리퍼가 닫힘(raw>=50)일 때 LED가 켜지고 열림(raw<50)일 때 꺼져야 정상. 반대로 켜지길 원하면
`--gripper-invert`를 붙여 재실행 (record_mujoco.py도 같은 플래그로 맞출 것).
