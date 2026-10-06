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

## rl_env_smoke.py — MuJoCo RL 환경 스모크 (2026-10-03, ee_rig 기반 재작성)

```bash
PYTHONPATH=. python ai_layer/tools/rl_env_smoke.py [--bc-checkpoint outputs/bc_act/last]
```

`so101_seam_env.py`가 실제 데이터 수집과 같은 물리(ee_rig.xml)로 전면 재작성됐다(아래 "리더암 ->
조이스틱 전환" 절 참고 — RL도 이제 같은 리그를 쓴다). 확인 항목: home tip 위치(5cm 막대 오프셋
반영, z≈0.10), 3초 제로-액션 드리프트(<2mm), +x 액션 스케일 적분 정확도, **바닥/용지 접촉 시
즉시 terminated + floor_contact_penalty**(record_mujoco.py의 "접촉=자동 폐기"와 같은 규약),
**선에서 off_seam_safety_dist(기본 3cm)보다 멀면 트리거를 켜도 비드가 강제로 꺼지는 안전 컷오프**
(reward.py의 coverage soft penalty와는 별개의 하드 제약), BC teacher 보상 계산.

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

### keyboard_input.py / teleop_input.py — 조이스틱 없을 때 키보드로 대체 (2026-10-06)

`KeyboardEEController`가 `JoystickEEController`와 **똑같은 공개 인터페이스**(poll/ee_velocity/
gripper_bit/rotation_rate/episode_end_requested/discard_requested/close)를 구현해서 드롭인으로
바꿔 끼울 수 있다. `teleop_input.build_ee_controller(input_mode, recalibrate)`가 실제 선택을
담당하고 `record_mujoco.py`/`check_ee.py`/`record_gui.py` 전부 이걸 쓴다 — 셋 다 공통으로
`--input {auto,joystick,keyboard}`를 받는다(기본 `auto`: 조이스틱을 찾아보고 없으면 키보드로
자동 전환, 전환되면 콘솔에 안내 메시지 출력).

키 배치(조이스틱의 "누르는 동안 레이트" 관례 그대로 — 아날로그가 없어서 전부 on/off):

| 기능 | 키 |
|---|---|
| EE +x / -x | W / S |
| EE +y / -y | D / A |
| EE +z / -z | R / F |
| roll -/+ | Q / E |
| pitch -/+ | Z / X |
| yaw -/+ | C / V |
| 그리퍼/도구 신호(누르는 동안) | SPACE |
| 에피소드 저장+종료 (BTN_THUMB) | ENTER |
| 에피소드 폐기+재시도 (BTN_THUMB2) | BACKSPACE |

```bash
PYTHONPATH=. python ai_layer/tools/keyboard_input.py [--list]   # 단독 진단 모드
PYTHONPATH=. python ai_layer/tools/record_mujoco.py --input keyboard --dry-run --num-episodes 1
```

joystick_input.py와 같은 evdev 기반이라 키보드도 보통 `input` 그룹 권한이 필요하다(0단계 참고).
`--recalibrate-joystick`은 키보드 입력일 때는 그냥 무시된다(스로틀 자체가 없으므로).

## record_mujoco.py — 조이스틱 -> MuJoCo EE 리그 미러링 데이터 수집 (2026-09-22, 기범 / 2026-09-23 조이스틱 전환)

실물 팔로워/카메라 없이 **조이스틱 하나만으로** BC 학습용 LeRobotDataset을 만든다. 더 이상 관절이
없으므로 데이터셋도 관절공간이 아니라 **EE-native 포맷으로 직접 기록**한다:
`observation.state`(9,)=[x,y,z,rot6d(6)], `observation.images.wrist`, `action`(7,)=[dx,dy,dz,drx,dry,drz,gripper]
— 이미 `configs/so101_act_bc.py`의 ACTConfig 입출력 스펙과 형태가 같다.
2026-10-03: `train_bc.py`는 이제 `ai_layer/data/load_bc_dataset()`으로 robot_type을 보고 자동으로
`SO101EEDataset`(이 도구가 만드는 EE-native 포맷, 변환 불필요)과 `SO101BCDataset`(실물 lerobot-record
관절공간 포맷)을 구분해서 쓴다 — 더 이상 후속 작업이 필요 없다. `check_dataset.py`도 같은 판별을 쓴다.

**`--scene`으로 용접선 형태를 고른다** (기본 `balanced`): `balanced`는 에피소드마다 `curve`(완만한
곡선)/`straight`(직선)/`sharp_curve`(급곡선)/`corner`(코너)/`branch`(분기점, seam_cv 분기점 순서
로직 검증용)/`dashed`(점선, seam_cv KD-tree 갭브리징 검증용) × variant 30가지 조합 중 **지금까지
가장 적게 기록된 조합**을 무작위로 골라 씬을 다시 로드한다(`BalancedSceneSampler`) — 완전 무작위면
적은 에피소드 수에서 특정 형태가 몰리거나 아예 안 나올 수 있어서, 랜덤하되 분포가 거의 균등하게
수렴하도록 했다. 카운트는 `<root>/meta/scene_balance.json`에 저장되고 **에피소드가 저장될 때만**
올라가므로(폐기분 제외), 세션을 여러 번 나눠 돌려도 데이터셋 전체 분포가 유지된다. 특정 형태
이름을 주면 그 형태 안에서만(variant로) 균형 샘플링한다. A4 용지는 모든 씬에서 공통으로 세계좌표
x=0.25 중심, 긴 축(297mm)이 x축. 손목 카메라는 `ee_body`에 직접 달려 있고 위치는 근사값(뷰어로
확인 후 조정 가능) — top-down에 가까운 각도로 바로 아래 용접선이 보이게 맞춰뒀다.

**`--variant`로 형태별 가우시안 노이즈 변형을 고른다** (기본 `-1` = BalancedSceneSampler로 고름,
`--scene balanced`일 땐 무시됨. `0`=노이즈 없는 기준, `1~4`=고정 시드로 재현 가능한 노이즈 버전).
형태마다 곡선 진폭/주기/위상, 코너 꺾이는 위치/각도, 분기 위치/각도, 점선 간격, **선 굵기**까지
흔들어서 형태당 5종씩(총 30개 씬) 미리 구워뒀다(`textures/gen_seam_textures.py --variants N`).
레퍼런스 6종만 계속 쓰면 BC가 "선은 항상 이 모양"이라고 암기할 위험이 있어서, 매 녹화 세션마다
다른 인스턴스를 보게 하는 게 목적 — 왜 런타임 랜덤화(텍스처를 매 에피소드 새로 그려 넣기) 대신
오프라인 사전 생성인지는 파일 상단 docstring 참고(GPU 텍스처 재업로드가 필요해서 뷰어 켠 채로는
번거로움). 2026-10-03: 텍스처 PNG 옆에 같은 이름의 `.json`(실제로 그린 경로 좌표+선 굵기)도 같이
저장한다 — RL(`so101_seam_env.py`)이 reward 계산에 쓰는 ground-truth 경로의 출처다
(`envs/seam_ground_truth.py`).

**`--root`를 안 주면 이제 HF 캐시가 아니라 프로젝트 로컬 `datasets/<repo-id>`에 저장한다**
(`.gitignore`에 이미 포함돼 커밋되지 않음) — 팀원끼리 캐시 경로가 흩어지는 문제 방지.

```bash
PYTHONPATH=. python ai_layer/tools/record_mujoco.py \
    --repo-id <hf-user>/so101-mujoco-demo --num-episodes 30
# --scene balanced(기본)가 30개 조합을 거의 균등하게 돌아가며 고른다. 특정 형태만 모으려면:
PYTHONPATH=. python ai_layer/tools/record_mujoco.py \
    --repo-id <hf-user>/so101-mujoco-demo --scene dashed --num-episodes 5
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

**2026-10-06: 막대가 바닥/용지에 닿아도 더 이상 에피소드가 자동 폐기되지 않는다.** 예전엔
`_contact_pos()`로 접촉을 감지하면 그 즉시 BTN_THUMB2를 누른 것과 같은 경로로 폐기됐는데,
순간적으로 한 번 내려간 것 때문에 멀쩡한 녹화를 통째로 날리는 문제가 있었다("비트 부러짐
방지" 취지로는 접촉 자체를 막는 게 맞다는 판단). 대신 `MIN_TIP_Z`(1cm) 밑으로는 도구 끝이
아예 못 내려가게 매 프레임 명령 위치 자체를 클램프한다(세게 계속 눌러도 막힘, 스트레스 테스트로
확인됨) — 접촉은 이제 그냥 "닿을 일이 거의 없는" 상태고 실패 조건이 아니다.

`--max-linear-speed`(기본 0.05 m/s)/`--max-angular-speed`(기본 1.0 rad/s)로 조이스틱 최대
속도 조절. EE 위치는 작업공간(x 0.05~0.45, y -0.20~0.20, z -0.02~0.35, `WORKSPACE_*`)으로
clamp된다 — 관절 리치 제약이 없어진 대신 슬라이더/스틱을 오래 누르고 있어도 화면 밖으로
날아가지 않게.

**그리퍼/도구 신호**: `BTN_TRIGGER`를 누르는 동안 1이고, **높이/접촉과 무관하게** 그동안
매 스텝 비드가 찍힌다. "실리콘이 중력의 영향을 받는다"는 걸 단순화해서 표현 — 비드는 도구
끝 위치가 아니라 도구의 (x, y) 바로 아래 바닥/용지 면(`FLOOR_Z`)에 찍힌다
(수직 낙하만 가정, 실제 유체 시뮬레이션 아님). **그리스/실리콘 느낌**은
`BEAD_RGBA`/`BEAD_RADIUS`(1.8mm)/`BEAD_STRIDE`(매 스텝, 촘촘하게 찍어서 거의 이어진 선처럼
보임)와 `specular`/`shininess`/`reflectance`를 낮게 준 무광에 가까운 살짝 반투명한 재질로
낸다. 렌더된 손목 카메라 이미지에도 남기 때문에(뷰어 전용 아님) BC가 이미 도포된 구간을
시각적으로 구분할 수 있다.

**`--dry-run`**: 저장(add_frame/save_episode/finalize)만 전부 건너뛰고 나머지(조작/씬 전환/
버튼/뷰어/높이 제한)는 실제 수집과 동일하게 돈다 — `--repo-id` 없이 바로 실행 가능, 절차
연습용. 씬 분포 카운트도 메모리에서만 세고 파일에 저장 안 함.

**GUI 버전**(`record_gui.py`, 아래)도 있지만 2026-10-06 기준 이 머신에서 간헐적인 GLX 크래시
문제로 보류 중이다 — 지금은 이 터미널 버전을 기본으로 쓸 것.

## bench_inference.py — 추론 지연 벤치마크 (시연 PC 비교용)

```bash
PYTHONPATH=. python ai_layer/tools/bench_inference.py --checkpoint outputs/bc_act/last   # 또는 --synthetic
```

seam/전처리/ACT forward/후처리/인코딩/끝-끝 p50·p95·max, 첫 추론 워밍업, 제어 max_age 300 ms 대비 여유를 찍는다.
2026-09-23 A6000: 끝-끝 p50 9.3 / p95 14.6 ms. 다른 PC(5090 등)에서 같은 명령으로 재면 바로 비교된다.

## record_gui.py — 데이터 수집 GUI (2026-10-06, 기범) — ⚠️ 보류 중

**이 머신에서 간헐적으로 `X Error of failed request: BadAccess ... X_GLXMakeCurrent`로 죽는다**
— 같은 코드를 그대로 여러 번 돌려도 될 때도 있고 안 될 때도 있어서 재현 조건을 못 찾았다
(순수 Dear PyGui만 단독으로는 멀쩡함, mujoco/torch 임포트나 실제 조이스틱 연결을 더해도 단독
재현 안 됨 — record_gui.py 전체를 그대로 돌릴 때만 간헐적으로 발생). 코드/문서는 남겨두지만
지금은 **`record_mujoco.py`(터미널)를 기본으로 쓸 것**. 다시 시도하고 싶으면 아래 그대로
실행해보고, 또 같은 에러가 나면 NVIDIA Optimus GPU 선택 문제일 수 있어 `__NV_PRIME_RENDER_OFFLOAD=1
__GLX_VENDOR_LIBRARY_NAME=nvidia` 를 앞에 붙여서 재시도해볼 것(아직 검증 안 됨).

`record_mujoco.py`와 똑같은 물리/조이스틱/데이터셋/balanced 샘플링을 쓰되, MuJoCo 자유시점
뷰어 대신 **손목 카메라(데이터셋에 실제 저장되는 화면) + 오버뷰 카메라(scene_common.xml의 고정
`overview` 카메라)**를 나란히 보여주고, 에피소드 시작/저장/폐기를 화면 버튼으로도 조작할 수
있고(조이스틱 BTN_THUMB/BTN_THUMB2와 동일 기능 — 조이스틱이 메인, 버튼은 보조), (형태, variant)별
수집 현황을 막대그래프로 실시간 표시한다(`Dear PyGui` 필요, `pip install dearpygui`).

```bash
PYTHONPATH=. python ai_layer/tools/record_gui.py --repo-id <hf-user>/so101-weld-demo --num-episodes 30
PYTHONPATH=. python ai_layer/tools/record_gui.py --dry-run --num-episodes 3   # 저장 없이 연습
```

카메라 소스는 `MujocoDualCamera` 클래스 하나로 묶어뒀다 — 나중에 실로봇으로 녹화할 때는 이
클래스만 실제 카메라 드라이버(예: `cv2.VideoCapture`, RealSense SDK)로 교체하면 되고, GUI/버튼/
에피소드 진행 로직은 그대로 재사용 가능하게 설계했다. `record_mujoco.py`의 터미널 while 루프는
그대로 남아 있다(안 건드림) — 터미널 플로우가 더 익숙하면 계속 그걸 써도 된다.

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
일부러 바닥에 대면 "실패(접촉!)"이 뜨는지 확인한다(2026-10-06: 이 "실패" 표시는 check_ee.py
전용 진단이다 — record_mujoco.py/record_gui.py는 접촉 시 더 이상 실패 처리하지 않고 높이로
막는다, 위 record_mujoco.py 절 참고). 방향이 반대면 `joystick_input.py`의
`ee_velocity()`에서 부호만 뒤집을 것.
