# AI 추론 계층 구현 — BC(ACT) + RL(SAC) 프레임워크

설계 문서: [`../docs/AI_추론계층_프레임워크.md`](../docs/AI_추론계층_프레임워크.md). 이 폴더는 그 문서
3.3절(BC), 3.4절(RL)을 실제로 동작하는 코드로 구현한 것이다. 지연보정(4장)은 아직 미구현.

> **지금 뭘 실행해야 하는지는 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)를 볼 것.** 이 문서는
> 그 아래 설계 결정/히스토리 쪽에 더 가깝다 — 특히 "엔드투엔드 흐름"/"검증 상태" 절은 2026-09-21~22
> (리더암 실물 녹화 전제) 시점 기록이고, 2026-09-23 조이스틱+EE-only 전환 이후로는
> TRAINING_GUIDE.md/tools/README.md 쪽이 실제 현재 상태를 반영한다.

## 왜 Transformer를 새로 짜지 않았는가

lerobot에 이미 검증된 ACT(Action Chunking Transformer) 구현(`lerobot.policies.act`)이 있고,
SO-101 leader/follower 드라이버(`so101_leader`/`so101_follower`)도 내장되어 있다. 이 프로젝트가 추가한
것은 **관측/액션 스펙**(EEF-delta 통일, seam CV 특징)과 **데이터 변환**(관절공간 → EEF-delta)뿐이다.

## 왜 IsaacLab 대신 MuJoCo인가 (2026-09-22 전환)

처음엔 IsaacLab으로 구현했었다(git 히스토리 참고). 이 컴퓨터(8GB VRAM 노트북 GPU)에서 Isaac Sim/Kit이
계속 불안정했고(CUDA P2P 검증 행, GPU 드라이버 이슈, `pip_prebundle` 의존성 충돌 등 이번 세션에서만
수십 차례 디버깅), SAC는 off-policy라 PPO만큼 대규모 병렬환경이 필요 없으며 BC가 이미 기초 정책을
제공하므로 RL은 국소 탐색 위주라 무거운 병렬 시뮬레이터가 필수가 아니라는 판단으로 MuJoCo로 전환했다.
MuJoCo는 GPU 없이도 가볍고 안정적이며, **이 코딩 세션 안에서 직접 실행·검증이 가능**하다는 실질적 이점도
크다(IsaacLab은 이 세션의 샌드박스에서 항상 멈춰서 매번 사용자 터미널에 의존해야 했음).

## 구성

```
ai_layer/
  kinematics.py              PAC_Supermoon URDF FK/IK (`tcp_link`) + ActionChunk 증분 EEF-delta + rot6d state
                              (실물/관절공간 경로용. EE-native 경로는 이미 EEF라 이 모듈을 거의 안 씀)
  perception/seam_cv.py      Seam/Groove 고전 CV 모듈 (docs 3.1절)
  configs/so101_act_bc.py    ACTConfig 프리셋: chunk_size=32, use_vae=False, EEF-delta 7dim 액션, ZERO_YAW
  configs/so101_sac.py       SACConfig 프리셋: 연속 6dim(EEF-delta) + 이산 1(그리퍼), discount=0.97
  data/__init__.py            detect_dataset_kind()/load_bc_dataset()/require_local_dataset() — robot_type으로
                              joint/ee 포맷을 자동 판별해 맞는 Dataset을 골라줌 (train_bc.py, check_dataset.py가 씀)
  data/so101_bc_dataset.py   lerobot 원본(관절공간, 실물 lerobot-record) 데이터셋 -> EEF-delta + seam 특징 wrapper
  data/so101_ee_dataset.py   record_mujoco.py/record_gui.py가 만든 EE-native 데이터셋 wrapper(변환 불필요,
                              yaw만 BC 타깃에서 마스킹)
  bc_inference.py             체크포인트 폴더 로드(정책+전/후처리) + 청크 예측. RL teacher·ActionChunk 직렬화 공용
  perception/seam_features.py 이미지 -> seam 특징(5). 학습과 추론이 같은 함수 사용
  control_bridge/protocol.py  송지수 제어 규약 복사본 (ActionChunk/StateSnapshot + 고정 크기 코덱, 원본 da9f29f)
  control_bridge/chunk_builder.py   모델 청크 (T,7) -> ActionChunk. 스텝 한계 클립, eef 매핑 스위치(EefMode)
  control_bridge/snapshot_adapter.py StateSnapshot -> observation.state(9D), anchor 선택 (OBS_POSE / COMMIT_END)
  control_bridge/ai_node.py   실행 노드: 스냅샷 SUB -> 이미지 -> ACT -> ActionChunk PUB (ZeroMQ ipc)
  tools/record_mujoco.py      조이스틱 -> MuJoCo EE 리그(ee_rig.xml) -> EE-native LeRobotDataset 녹화 (터미널)
  tools/record_gui.py         위와 동일 로직, Dear PyGui GUI(카메라 화면/버튼/수집 현황 차트)
  tools/joystick_input.py     Logitech Extreme 3D Pro(evdev) -> EE 속도/회전/그리퍼, 스로틀 영점 보정/저장
  tools/check_ee.py           녹화 없이 조이스틱->EE 매핑/충돌 빠르게 확인
  tools/check_dataset.py      녹화 직후 데이터셋 검증 (fps/차원/seam인식률/증분크기 등)
  tools/chunk_bridge_test.py  규약 왕복 + 원본 코덱/validator 대조
  tools/fk_smoke.py           실로봇 URDF FK 스모크
  tools/bc_synthetic_test.py  가짜 데이터셋으로 BC 경로 끝-끝 실행 검증(관절공간 경로)
  tools/rl_env_smoke.py       so101_seam_env.py 스모크(홈 위치/드리프트/안전 컷오프/바닥접촉 종료 등)
  tools/README.md             도구별 상세 옵션 + 환경 설치 순서
  rl/reward.py                R_imitation + R_track + R_smooth + R_coverage, 가중치 스케줄링
  rl/replay_buffer.py         단일 프로세스 SAC용 최소 리플레이 버퍼
  envs/so101_seam_env.py      MuJoCo 기반 Gymnasium 환경 — ee_rig.xml(EE-only, IK 없음) 기반, 매 에피소드
                              30가지 (형태,variant) 중 무작위 ground-truth 경로, 안전 컷오프+바닥접촉 종료
  envs/seam_ground_truth.py   textures/gen_seam_textures.py가 구운 실제 경로(.json) -> world 좌표 폴리라인
  train_bc.py                 BC 학습 진입점 (lerobot ACTPolicy 그대로 사용, robot_type 자동 판별)
  train_rl.py                 RL 학습 진입점 (lerobot SACPolicy 그대로 사용, BC 체크포인트를
                               R_imitation teacher로 선택적 로드, metrics.jsonl 로깅) — 일반 파이썬
                               스크립트, IsaacLab 의존성 없음
```

## 엔드투엔드 흐름

**현재(2026-10-06) 실제 쓰는 흐름은 데이터 수집부터 RL까지 전부 시뮬레이션(MuJoCo) +
조이스틱/GUI다** — 실물 SO-101 leader로 티칭하는 아래 1단계는 아직 미착수 상태 그대로다.
전체 단계별 실행 명령/옵션은 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)에 있다 — 요약만 적으면:

```
0. 환경 설치/활성화, 1. 데이터 수집(record_gui.py 또는 record_mujoco.py, 조이스틱),
2. 데이터 점검(check_dataset.py), 3. BC 학습(train_bc.py), 4. RL 환경 스모크(rl_env_smoke.py),
5. RL 학습(train_rl.py, BC 체크포인트를 참고 teacher로 선택 사용) — 전부 TRAINING_GUIDE.md 참고.
```

3단계(추론 노드, 제어 계층 연결)는 아래 "2026-09-21" 검증 기록 당시 그대로이고 아직 실물
연결은 안 했다:

```
PYTHONPATH=. python ai_layer/control_bridge/ai_node.py \
    --checkpoint outputs/bc_act/last --camera realsense --anchor commit --eef-mode off
(배관 점검: --dry-run --fake-snapshot --iterations 3)
```

## 검증 상태 (2026-09-21, LeRobot 0.4.4 / env_lerobot)

- ✅ `tools/fk_smoke.py`: 실로봇 URDF `tcp_link` FK, 증분 델타, yaw=0, rot6d 왕복 — 통과.
- ✅ `tools/bc_synthetic_test.py`: 가짜 LeRobotDataset → `SO101BCDataset` → 통계 → `ACTPolicy` 학습 6스텝
  → 폴더 체크포인트 저장 → `bc_inference.load_bc_checkpoint` 복원 → 청크 예측(단위 확인) — 통과.
  (이 테스트가 잡은 버그: placo FK가 float32를 거부 → `kinematics.joint_traj_to_eef_pose_traj`에서 float64 캐스팅)
- ✅ `tools/chunk_bridge_test.py`: 송지수 규약 코덱 바이트 일치 + 원본 ChunkValidator 통과. `control_bridge/ai_node.py`
  는 `run_live.py --ai external` 과 ZeroMQ 실연결로 청크 수락 확인.
- ✅ `envs/so101_seam_env.py` (MuJoCo, 기범 선배님 2026-09-22) + `train_rl.py`: 병합 후 env_lerobot 에서 재실행
  (reset/step, FK+IK, 관측 9D, SAC 학습 루프 수백 스텝) — 아래 "2026-09-22 병합" 참고.
- ⚠️ 학습 속도: 학습 스텝당 대략 0.2초 내외 — 장시간(수만~수십만 스텝) 학습은 백그라운드로.

### 2026-09-21 수정 요약 (리뷰 후 기범 선배님 승인)

- **정규화**: LeRobot 0.4.x는 정책 밖 processor가 정규화한다. train_bc/train_rl이 이를 안 써서 원시값으로
  학습되던 것을 `make_act_pre_post_processors` / `make_sac_pre_post_processors` 사용으로 수정.
- **체크포인트**: state_dict 단독 → `save_pretrained` 폴더(정책 config + 가중치 + 전/후처리 통계).
- **시간 정렬**: action 청크 k = t+(k+1)dt. delta[0] = state(t)→action(t+dt) (예전 action(t)−state(t)는 추종오차).
- **관측 state**: 회전벡터(180° 불연속) → 9D (xyz + rot6d). BC/RL 동일.
- **이미지 키**: BC `observation.images.wrist` / RL `observation.image.wrist` 불일치 → 전부 `images`.
- **그리퍼**: LeRobot 0.4.x는 0~100(RANGE_0_100)으로 녹화. 주석의 1000~4000은 모터 로우값이라 정정.
- **RL**: SAC 통계 직접 제공, 리플레이 링버퍼, target_speed 0.05→0.01(< action_scale_pos 0.02),
  IK 관절 그리퍼 제외, BC teacher 출력을 env 스케일로 변환, 가중치 스케줄 분모 = 전체 학습 길이.
- **seam_cv**: 끝점 탐색 컨볼루션화, 분기점에서 진행 방향 유지, 끊김 점프 KD-tree, 학습 시 특징 캐시, 밝은 선 부호 수정.

### 2026-09-22 병합 (기범 선배님 main: MuJoCo 전환)

- 선배님 MuJoCo env/train_rl 을 받아들이고 위 RL 정합(9D state, processor, BC teacher 스케일, 스케줄, 폴더 체크포인트)을 다시 적용.
- `build_arm_kinematics` 의 타깃을 `gripper_frame_link` → `TARGET_FRAME_NAME`(`tcp_link`). PAC_Supermoon URDF 에는
  `gripper_frame_link` 가 없어 그대로 두면 IK 가 실패한다.
- FK/IK 는 실로봇 URDF, MuJoCo 는 기본 SO-101 MJCF 로 관절 동역학만 담당 → 관측 EE pose 는 qpos→우리 FK 라 학습·추론과 일관.
  MJCF 형상을 UMI+D405 로 바꾸는 것은 후속.
- 병합 검증(env_lerobot, mujoco 3.13)에서 잡은 env 문제 3개와 수정:
  1. `model.opt.timestep = 1/60` 은 sts3215 서보(kp≈998)에 수치적으로 불안정 → ctrl 고정인데 1초 뒤 |qvel| 1.6 rad/s, 팔이 무너짐.
     physics_dt 1/300 × decimation 10 (정책 30 Hz 유지)으로 변경. MJCF 기본 0.002 s 에선 1초 처짐 0.03°.
  2. 증분을 **측정** pose 에 누적 → 서보 처짐이 매 스텝 목표로 흡수되어 흘러내림. **명령** pose 에 누적(`_T_cmd`, 송지수 규약과 동일)
     + 도달 불가 시 스냅. 결과: 명령 0 으로 60 스텝 드리프트 0.
  3. qpos=0(팔 완전히 뻗음) 시작 + lerobot IK 1회 풀이 → 자세 명령 몇 스텝에 관절 한계(−100°/95°)로 밀려 접힘.
     시작 자세를 경로 영역 위 굽힌 자세(`home_joint_deg`, 한계 여유 24°)로, IK 5회 반복, ctrlrange 클램프.
- Isaac 자산(`assets/so101_isaac`)과 `sim/smoke_test_so101.py` 는 선배님 결정대로 삭제.

## 자산

- `../assets/pac_supermoon/` — 실로봇 URDF (PAC_Supermoon, D405 홀더, FK/IK 타깃 `tcp_link`). 관절공간
  경로(`kinematics.py`, `so101_bc_dataset.py`)만 쓴다 — EE-native 경로는 이 URDF를 안 씀.
- `../assets/so101/` — 공식 SO-ARM100 SO-101 URDF + 메시(참고, FK/IK는 안 씀) + 현재 실제로 쓰는
  **MJCF**: `ee_rig.xml`(EE-only 리그, mocap+weld, 관절 없음), `scene_common.xml`(책상/바닥/벽
  배경 + A4 용지 + 손목/오버뷰 카메라, 30개 `scene_a4*.xml`이 공유), `scene_a4*.xml`(형태×variant
  30종, 텍스처 참조만 다름). `so101_new_calib.xml`(구 5관절 몸통 MJCF)은 더 이상 안 쓴다.
- `../assets/so101/textures/gen_seam_textures.py` — 용접선 텍스처(.png) + ground-truth 경로(.json)를
  같이 굽는다. `envs/seam_ground_truth.py`가 그 .json을 world 좌표로 읽어 RL reward에 쓴다.

## 알려진 제약 / TODO

- gripper 채널(7번째)은 EE-native 경로에서 이미 바이너리(0/1, 트리거 신호)로 고정됐다. 실로봇
  집게/펌프 중 뭐가 될지는 실물 연결 시점에 결정.
- ActionChunk 직렬화/송신은 `control_bridge/` 에 있음. 남은 것: 실카메라(D405) 촬영시각-측정시각
  오프셋 캘리브레이션, 실로봇 HAL(송지수·민제씨 쪽), world==base 좌표계 가정 확인 — 전부 실물
  연결 전제라 아직 미착수.
- seam CV 특징(5dim)은 depth 없이 2D 픽셀 기준 축약 벡터. Wrist RGBD를 실제로 쓸 때는
  `seam_cv.py::detect()`에 depth+intrinsics를 넘겨 3D 특징으로 확장할 것.
- RL 환경(`so101_seam_env.py`) 관측의 손목 카메라는 아직 0 placeholder다(렌더링 비용 때문에
  매 스텝 렌더 안 함) — 목표 경로 자체는 더 이상 절차적 직선이 아니라 `seam_ground_truth.py`로
  실제 30종 텍스처 경로를 읽는다(2026-10-06 해결).
- 지연보정 모듈은 이 폴더에 아직 없음 — 다음 작업 대상.
- SO-101 leader(실물)로 실제 데이터 수집·BC 학습·RL fine-tuning 실행은 아직 미착수 — 지금까지
  전부 MuJoCo 시뮬레이션 + 조이스틱/GUI로 검증됨.
