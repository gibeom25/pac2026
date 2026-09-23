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

## 2026-09-23: 리더암 -> 조이스틱 전환, 로봇 몸통 제거 (EE 전용 리그)

시뮬레이션에서는 EE(도구 끝)만 참고하면 되고 5관절 체인/캘리브레이션은 신경 쓸 필요 없다는 결정 +
"리더암은 무게·신호지연 때문에 오히려 사람 의도를 정확히 반영하기 어렵다"는 판단으로, MuJoCo 미러링
도구를 **SO-101 leader -> Logitech Extreme 3D Pro 조이스틱**, **5관절 로봇 몸통 -> mocap+weld EE
전용 리그**(`assets/so101/ee_rig.xml`)로 전면 교체했다. 로봇 팔 메시/조인트가 전부 사라지고,
`mocap_target`(조이스틱 적분값으로 매 스텝 직접 위치 지정, kinematic) <- weld(spring) - `ee_body`
(freejoint, 실제 물리 바디, `tool_rod`가 붙어 있어 바닥과 부딪히면 진짜 반발력이 생김)만 남는다.
`scene_a4_<이름>.xml`들은 이제 `ee_rig.xml`을 include한다 (기존 so101_new_calib_camera.xml 대신).

물리 안정성 검증: mocap을 바닥 8~10cm 아래로 계속 명령해도(극단적 스트레스 테스트) NaN 없이 반발력으로
막힘, 평상시 추종 오차 <0.4mm.

### joystick_input.py — Extreme 3D Pro(evdev) -> EE 속도/회전/그리퍼 신호

실제 장치(`/dev/input/eventNN`, evdev로 자동 탐색)에서 **직접 읽어 확인한** 축/버튼 매핑을 쓴다
(추측 아님): `ABS_X`(좌우 스틱) -> EE y속도, `ABS_Y`(전후 스틱) -> EE x속도, `ABS_THROTTLE`(슬라이더,
자체복원 없음 — **연결 시점의 현재 위치를 정지 기준으로 잡는다**, 안 그러면 안 만져도 계속 움직임,
연결 시점에 끝 근처에 있으면 경고 출력) -> EE z속도, `BTN_TRIGGER` -> 그리퍼/도구 신호(누르는 동안
1, 임계값 없음). x/z 기본 부호는 실사용 확인 후 반전해뒀다 — 반대로 느껴지면
`--invert-x/--invert-y/--invert-z`로 바로 뒤집을 것(`record_mujoco.py`/`check_ee.py` 둘 다 지원).

**roll/pitch는 베이스 버튼, yaw는 트위스트 축으로 조절한다** (`rotation_rate()`, 2026-09-23
최종 결정): `BTN_BASE`/`BTN_BASE2` = roll -/+, `BTN_BASE3`/`BTN_BASE4` = pitch -/+ (throttle처럼
누르고 있는 동안만 그 방향으로 회전, 레이트 컨트롤) / `ABS_RZ`(트위스트) = yaw 각속도(연속값 —
손목을 실제로 돌리는 축이라 버튼보다 자연스러움). `--max-angular-speed`(기본 1.0 rad/s)로 속도 조절.

```bash
PYTHONPATH=. python ai_layer/tools/joystick_input.py [--list]
```

단독 실행하면 라이브 진단 모드(축/버튼 값 + 눌린 버튼 이름 실시간 출력, `--list`는 연결된
입력 장치 목록만 출력) — 어느 물리 버튼이 어떤 코드인지 헷갈리면 이걸로 직접 눌러서 확인할 것.

## record_mujoco.py — 조이스틱 -> MuJoCo EE 리그 미러링 데이터 수집 (2026-09-22, 기범 / 2026-09-23 조이스틱 전환)

실물 팔로워/카메라 없이 **조이스틱 하나만으로** BC 학습용 LeRobotDataset을 만든다. 더 이상 관절이
없으므로 데이터셋도 관절공간이 아니라 **EE-native 포맷으로 직접 기록**한다:
`observation.state`(9,)=[x,y,z,rot6d(6)], `observation.images.wrist`, `action`(7,)=[dx,dy,dz,drx,dry,drz,gripper]
— 이미 `configs/so101_act_bc.py`의 ACTConfig 입출력 스펙과 형태가 같다.
⚠️ `train_bc.py`가 쓰는 `SO101BCDataset`은 아직 관절공간 -> EEF 변환을 전제로 하므로 이 포맷을 바로
못 읽는다 — 학습에 쓰려면 SO101BCDataset에 "이미 EEF 포맷인 데이터셋은 변환 없이 통과" 경로를
추가하는 후속 작업이 필요하다 (아직 안 함).

**`--scene`으로 용접선 형태를 고른다** (기본 `curve`): `curve`(완만한 곡선) / `straight`(직선) /
`sharp_curve`(급곡선) / `corner`(코너) / `branch`(분기점, seam_cv 분기점 순서 로직 검증용) /
`dashed`(점선, seam_cv KD-tree 갭브리징 검증용). A4 용지는 모든 씬에서 공통으로 세계좌표 x=0.25
중심(so101_seam_env.py 경로 영역 x 0.15~0.35가 안쪽에 들어옴), 긴 축(297mm)이 x축. 손목 카메라는
`ee_body`에 직접 달려 있고 위치는 근사값(뷰어로 확인 후 조정 가능) — top-down에 가까운 각도로
바로 아래 용접선이 보이게 맞춰뒀다.

**`--variant`로 형태별 가우시안 노이즈 변형을 고른다** (기본 `-1` = 매번 무작위, `0`=노이즈 없는
기준, `1~4`=고정 시드로 재현 가능한 노이즈 버전). 형태마다 곡선 진폭/주기/위상, 코너 꺾이는
위치/각도, 분기 위치/각도, 점선 간격, **선 굵기**까지 흔들어서 형태당 5종씩(총 30개 씬) 미리
구워뒀다(`textures/gen_seam_textures.py --variants N`). 레퍼런스 6종만 계속 쓰면 BC가 "선은
항상 이 모양"이라고 암기할 위험이 있어서, 매 녹화 세션마다 다른 인스턴스를 보게 하는 게 목적
— 왜 런타임 랜덤화(텍스처를 매 에피소드 새로 그려 넣기) 대신 오프라인 사전 생성인지는 파일
상단 docstring 참고(GPU 텍스처 재업로드가 필요해서 뷰어 켠 채로는 번거로움).

```bash
PYTHONPATH=. python ai_layer/tools/record_mujoco.py \
    --repo-id <hf-user>/so101-mujoco-demo --root ./datasets/so101-mujoco-demo \
    --scene dashed --num-episodes 5
```

**에피소드 길이는 기본적으로 무제한이다 — `BTN_THUMB`(엄지 버튼)를 누르면 그 자리에서 바로
저장하고 끝난다.** 시간을 미리 정해두고 쫓기듯 그릴 필요 없이 다 그렸을 때 직접 끝내는 방식.
필요하면 `--episode-seconds`로 상한을 줄 수도 있다(그 전에 BTN_THUMB를 눌러도 됨, 시간이
차면 자동 종료). 에피소드가 끝나면 EE 위치/자세가 즉시 홈으로 리셋되고(뷰어에도 바로 반영),
Enter로 다음 에피소드를 시작한다.

**실수했으면 `BTN_THUMB2`로 폐기** — 지금까지 그 에피소드에 쌓인 프레임(이미지 포함)을
`dataset.clear_episode_buffer()`로 버리고 같은 에피소드 번호를 다시 시도한다(에피소드
슬롯을 소모하지 않음). 어느 버튼이 어떤 건지 헷갈리면 `joystick_input.py`를 단독 실행하면
누른 버튼 이름이 그대로 출력된다.

**막대가 바닥/용지에 물리적으로 닿으면(`_contact_pos()`) 그 즉시 자동으로 폐기되고 같은
에피소드 번호로 재시도한다** (2026-09-23 최종 — BTN_THUMB2를 누른 것과 동일 경로). 도구는
표면에 닿지 않고 일정 간격을 띄운 채로 작업해야 한다는 제약을 물리 접촉으로 직접 강제하는 것.

`--max-linear-speed`(기본 0.05 m/s)/`--max-angular-speed`(기본 1.0 rad/s)로 조이스틱 최대
속도 조절. EE 위치는 작업공간(x 0.05~0.45, y -0.20~0.20, z -0.02~0.35, `WORKSPACE_*`)으로
clamp된다 — 관절 리치 제약이 없어진 대신 슬라이더/스틱을 오래 누르고 있어도 화면 밖으로
날아가지 않게.

**그리퍼/도구 신호**: `BTN_TRIGGER`를 누르는 동안 1이고, **높이/접촉과 무관하게** 그동안
매 스텝 비드가 찍힌다(2026-09-23 최종 — 이전엔 접촉 중일 때만 찍혔으나, 이제 접촉은 실패
조건이라 애초에 접촉 상태에서 찍을 일이 없음). "실리콘이 중력의 영향을 받는다"는 걸 단순화해서
표현 — 비드는 도구 끝 위치가 아니라 도구의 (x, y) 바로 아래 바닥/용지 면(`FLOOR_Z`)에 찍힌다
(수직 낙하만 가정, 실제 유체 시뮬레이션 아님). **그리스/실리콘 느낌**은
`BEAD_RGBA`/`BEAD_RADIUS`(1.8mm)/`BEAD_STRIDE`(매 스텝, 촘촘하게 찍어서 거의 이어진 선처럼
보임)와 `specular`/`shininess`/`reflectance`를 낮게 준 무광에 가까운 살짝 반투명한 재질로
낸다. 렌더된 손목 카메라 이미지에도 남기 때문에(뷰어 전용 아님) BC가 이미 도포된 구간을
시각적으로 구분할 수 있다.

## check_ee.py — 조이스틱 방향/회전/실패 확인 (2026-09-23, 기범)

record_mujoco.py로 전체 녹화를 돌리지 않고 EE가 조이스틱을 잘 따라가는지, 베이스 버튼으로
회전이 도는지, 트리거로 비드가 찍히는지, 막대가 바닥에 닿으면 "실패"가 뜨는지 빠르게 확인.
(예전 check_gripper.py — 로봇 몸통이 사라지면서 리더 그리퍼 방향 확인이라는 원래 목적이
없어져 조이스틱/EE 확인 도구로 대체함.)

```bash
PYTHONPATH=. python ai_layer/tools/check_ee.py --scene dashed
```

스틱을 움직여서 EE가 기대한 방향(앞/뒤=x, 좌/우=y)으로 가는지, 슬라이더로 z가 오르내리는지,
베이스 버튼으로 roll/pitch/yaw가 도는지, 트리거를 누르면 높이 상관없이 비드가 찍히는지, 막대를
일부러 바닥에 대면 "실패(접촉!)"이 뜨는지 확인한다. 방향이 반대면 `joystick_input.py`의
`ee_velocity()`에서 부호만 뒤집을 것.
