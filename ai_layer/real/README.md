# 실로봇(SO-101) 데이터 수집 백엔드 (2026-10-06)

시뮬레이션(`ai_layer/tools/episode_ticker.py`의 `EpisodeTicker`)과 **똑같은 공개 인터페이스**를
구현한 실로봇 버전 — `ai_layer/gui/app.py`의 "소스" 드롭다운(시뮬레이션/실로봇)으로 전환해서
쓴다. 데이터셋 스키마는 시뮬과 **완전히 동일**(observation.state 9차원 pose, action 6차원
delta+그리퍼, "wrist" 카메라 키) — `train_bc.py`/`train_rl.py` 등 BC/RL 학습 코드는 전혀 안
바꿔도 된다.

## 왜 새로 만들 게 거의 없었나

- **로봇 드라이버**: `lerobot`(이미 설치돼 있음)에 SO-100/101 팔로워 드라이버가 이미 있다 —
  `lerobot.robots.so_follower.so_follower.SOFollower`(시리얼로 연결, `get_observation()`/
  `send_action()`/`connect()`/`disconnect()`). 직접 안 만들고 그대로 가져다 썼다.
- **IK**: `ai_layer/kinematics.py`의 `build_arm_kinematics()`가 이미 있었다(PAC_Supermoon URDF,
  5관절 — yaw 없음, `zero_yaw()`가 있는 이유와 같음). `RobotKinematics.forward_kinematics()`/
  `inverse_kinematics()`를 그대로 썼다.
- **카메라**: `lerobot.cameras.opencv.OpenCVCameraConfig`로 일반 USB 웹캠을 그대로 쓴다 —
  `SOFollower`가 `cameras` 설정을 받아서 `get_observation()`에 같이 실어준다.
- **EE 컨트롤러(조이스틱/키보드)**: 손댈 필요가 전혀 없었다 — `JoystickEEController`/
  `KeyboardEEController`/`QtKeyboardEEController`는 애초에 "EE 속도(m/s, rad/s)"만 돌려주는
  하드웨어 중립 인터페이스라서, 시뮬 mocap이든 실로봇 IK든 똑같이 넣어주면 된다.
- **`MujocoDualCamera`**가 애초에 이 교체를 염두에 두고 설계돼 있었다("실로봇 전환 시 이
  클래스만 바꿔 끼우면 된다"는 주석이 2026-09-23부터 있었음).

새로 만든 건 `real_robot_ticker.py`(물리 스텝 대신 IK+send_action을 매 틱 돌리는 glue)와
`connection.py`(SOFollower+카메라+IK를 한 번에 구성하는 헬퍼)뿐이다.

## 사용법

1. (최초 1회) 실제 하드웨어로 lerobot 공식 보정 도구를 터미널에서 먼저 돌려둔다:
   ```bash
   python -m lerobot.scripts.lerobot_calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0
   ```
2. GUI(`ai_layer/gui/app.py`)의 데이터 수집 탭에서 **소스 = 실로봇**으로 바꾸고, "실로봇 연결"
   섹션에 port/카메라 index/max-relative-target을 입력한다.
3. "조작 테스트"로 먼저 움직여보고(설정 시작 전에 로봇이 바로 연결됨), 만족스러우면 "테스트
   통과 → 바로 녹화 시작"으로 넘어간다(로봇 재연결 없이 그대로 이어서 씀).

## 시뮬레이션과 다른 점

- **"씬"이라는 개념이 없다**: 시뮬은 에피소드마다 새 MJCF 씬 파일을 로드하지만, 실로봇은 물리적
  리셋이 없다 — 로봇의 현재 자세가 곧 다음 에피소드의 시작점이다. GUI의 scene/variant 설정은
  실로봇 모드에선 무시된다(`collect_tab.py`의 `_make_ticker()`가 분기).
- **비드/접촉 시뮬레이션이 없다**: `bead_points`는 항상 빈 리스트, `last_contact`는 항상
  False — 데이터셋 필드 자체엔 비드/접촉이 안 들어가므로(그냥 GUI 상태 표시용) 학습에 영향
  없다.
- **형태별 수집 현황 차트가 비어 있다**: `BalancedSceneSampler`는 시뮬 전용(씬 종류별 균형
  샘플링)이라 실로봇 세션에선 안 쓴다(`self._sampler = None`).
- **카메라가 하나만 있어도 된다**: 오버뷰 카메라 index를 -1(없음)로 두면 손목 카메라 화면을
  그대로 복사해서 보여준다.

## 안전 — 반드시 읽을 것

**이 코드는 실제 하드웨어로 검증하지 못했다** — 이 작업을 한 환경엔 실물 로봇이 없다. IK가
극단적인 해를 낼 가능성, 카메라 인덱스가 다른 장치를 가리킬 가능성, 시리얼 포트 문제 등을
직접 테스트해보지 못했으니 처음 돌릴 땐:

- `max-relative-target`(한 번에 보낼 수 있는 관절 이동량 상한, lerobot 자체 안전장치)을
  **낮게**(1~2도) 잡고 시작할 것 — GUI 기본값은 5도인데 이것도 보수적으로 잡은 값이지 검증된
  값이 아니다.
- 사람이 비상정지 가능한 거리에서, 저속(`max-linear-speed`를 작게)으로 먼저 테스트할 것.
- IK가 작업공간 경계 근처에서 특이점(singularity)에 걸려 관절이 급격히 틀어질 수 있는 자세가
  있을 수 있다 — `WORKSPACE_X/Y/Z`(record_mujoco.py, 시뮬 기준으로 잡힌 값)를 그대로 재사용
  했는데, 실제 SO-101의 도달 범위와 정확히 안 맞을 수 있다.

## 아직 안 한 것(의도적으로 범위 밖)

- **배포(학습된 정책을 실로봇에서 실행)**: 팀메이트(송지수)가 이미 만든
  `ai_layer/control_bridge/ai_node.py`(ZeroMQ 브리지)가 있어서, 이번엔 건드리지 않았다
  (2026-10-06 결정 — "일단 수집만, 배포는 나중에"). 나중에 배포까지 연결하려면 그 브리지와
  어떻게 맞물릴지(그대로 쓸지, GUI에서 그 프로세스를 띄우기만 할지) 먼저 정해야 한다.
- **그리퍼 비례 제어**: 지금은 트리거 on/off를 그리퍼 열림(0)/닫힘(100) 이진값으로만 매핑한다
  — 시뮬의 "bit" 그대로 가져온 것과 일관되게 단순화한 것, 더 정교한 제어가 필요하면 추가 작업.
