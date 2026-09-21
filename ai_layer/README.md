# AI 추론 계층 구현 — BC(ACT) + RL(SAC) 프레임워크

설계 문서: [`../docs/AI_추론계층_프레임워크.md`](../docs/AI_추론계층_프레임워크.md). 이 폴더는 그 문서
3.3절(BC), 3.4절(RL)을 실제로 동작하는 코드로 구현한 것이다. 지연보정(4장)은 아직 미구현.

## 왜 Transformer를 새로 짜지 않았는가

lerobot(0.4.4)에 이미 검증된 ACT(Action Chunking Transformer) 구현(`lerobot.policies.act`)이 있고,
SO-101 leader/follower 드라이버(`so101_leader`/`so101_follower`)도 내장되어 있다. 이 프로젝트가 추가한
것은 **관측/액션 스펙**(EEF-delta 통일, seam CV 특징)과 **데이터 변환**(관절공간 → EEF-delta)뿐이다.

## 구성

```
ai_layer/
  kinematics.py              PAC_Supermoon URDF FK (`tcp_link`) + ActionChunk 증분 EEF-delta
  perception/seam_cv.py      Seam/Groove 고전 CV 모듈 (docs 3.1절)
  configs/so101_act_bc.py    ACTConfig 프리셋: chunk_size=32, use_vae=False, EEF-delta 7dim 액션
  configs/so101_sac.py       SACConfig 프리셋: 연속 6dim(EEF-delta) + 이산 1(그리퍼), discount=0.97
  data/so101_bc_dataset.py   lerobot 원본(관절공간) 데이터셋 -> EEF-delta + seam 특징 변환 wrapper
                              (fps/관절순서 검사, seam 특징 캐시, 정규화 통계 compute_stats)
  bc_inference.py             체크포인트 폴더 로드(정책+전/후처리) + 청크 예측. RL teacher·ActionChunk 직렬화 공용
  tools/fk_smoke.py           실로봇 URDF FK 스모크
  tools/bc_synthetic_test.py  가짜 데이터셋으로 BC 경로 끝-끝 실행 검증
  tools/README.md             env_lerobot 설치 순서
  rl/reward.py                R_imitation + R_track + R_smooth (docs 3.4절), 가중치 스케줄링
  rl/replay_buffer.py         단일 프로세스 SAC용 최소 리플레이 버퍼
  envs/so101_seam_env.py      IsaacLab DirectRLEnv — SO-101 + 절차적 경로 + DifferentialIK
  sim/smoke_test_so101.py     Isaac Sim에 SO-101 USD가 정상 로드되는지 확인하는 헤드리스 스모크테스트
  train_bc.py                 BC 학습 진입점 (lerobot ACTPolicy 그대로 사용)
  train_rl.py                 RL 학습 진입점 (lerobot SACPolicy 그대로 사용, BC 체크포인트를
                               R_imitation teacher로 선택적 로드)
```

## 엔드투엔드 흐름

```
0. 환경: /home/dy/pac2026/env_lerobot (lerobot 0.4.4). 설치 순서는 tools/README.md.

1. 데이터 수집 (SO-101 leader로 티칭, 실물). fps는 반드시 30 (= DT_AI_SEC 1/30).
   lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM_follower \
                   --teleop.type=so101_leader --teleop.port=/dev/ttyACM_leader \
                   --robot.cameras='{"wrist": ...}' --dataset.fps=30 \
                   --dataset.repo-id=<user>/so101-weld-demo ...

2. BC 학습 (정규화 통계는 변환 후 값으로 자동 계산, 체크포인트는 폴더 단위)
   cd pac2026-team
   PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/train_bc.py \
       --repo-id <user>/so101-weld-demo --root <로컬경로> --epochs 100
   → outputs/bc_act/last/ (config.json, model.safetensors, policy_preprocessor.json, policy_postprocessor.json ...)

3. RL(SAC) 학습 — ⚠️ IsaacLab 필요, 반드시 사용자 터미널에서 직접 실행 (AI 세션의 Bash 안에서는
   CUDA P2P 검증 단계에서 멈춤)
   cd /home/dy/pac2026/IsaacLab
   ./isaaclab.sh -p /home/dy/pac2026/pac2026-team/ai_layer/train_rl.py --headless \
       --bc-checkpoint /home/dy/pac2026/pac2026-team/outputs/bc_act/last
```

## 검증 상태 (2026-09-21, LeRobot 0.4.4 / env_lerobot)

- ✅ `tools/fk_smoke.py`: 실로봇 URDF `tcp_link` FK, 증분 델타, yaw=0, rot6d 왕복 — 통과.
- ✅ `tools/bc_synthetic_test.py`: 가짜 LeRobotDataset → `SO101BCDataset` → 통계 → `ACTPolicy` 학습 6스텝
  → 폴더 체크포인트 저장 → `bc_inference.load_bc_checkpoint` 복원 → 청크 예측(단위 확인) — 통과.
  (이 테스트가 잡은 버그: placo FK가 float32를 거부 → `kinematics.joint_traj_to_eef_pose_traj`에서 float64 캐스팅)
- ⚠️ `configs/so101_sac.py`, `rl/*`, `envs/so101_seam_env.py`, `train_rl.py`: 2026-09-21 수정 후 **문법 검사만**.
  IsaacLab 필요라 사용자 터미널에서 최초 실행 시 디버깅 필요 — 특히 `DifferentialIKController` 연동,
  `matrix_from_quat` 기반 rot6d, BC teacher 스케일 변환.
- ✅ `sim/smoke_test_so101.py`: SO-101 USD 로드 + 시뮬 스텝, 사용자 터미널에서 성공 확인 (이전 세션).

### 2026-09-21 수정 요약 (리뷰 후 기범 선배님 승인)

- **정규화**: LeRobot 0.4.x는 정책 밖 processor가 정규화한다. train_bc/train_rl이 이를 안 써서 원시값으로
  학습되던 것을 `make_act_pre_post_processors` / `make_sac_pre_post_processors` 사용으로 수정.
- **체크포인트**: state_dict 단독 → `save_pretrained` 폴더(정책 config + 가중치 + 전/후처리 통계).
- **시간 정렬**: action 청크 k = t+(k+1)dt. delta[0] = state(t)→action(t+dt) (예전 action(t)−state(t)는 추종오차).
- **관측 state**: 회전벡터(180° 불연속) → 9D (xyz + rot6d). BC/RL 동일.
- **이미지 키**: BC `observation.images.wrist` / RL `observation.image.wrist` 불일치 → 전부 `images`.
- **그리퍼**: LeRobot 0.4.x는 0~100(RANGE_0_100)으로 녹화. 주석의 1000~4000은 모터 로우값이라 정정.
- **RL**: SAC 통계 직접 제공, 리플레이 링버퍼, target_speed 0.05→0.01(< action_scale_pos 0.02),
  IK 관절 Jaw 제외, BC teacher 출력을 env 스케일로 변환, 가중치 스케줄 분모 = 전체 학습 길이.
- **seam_cv**: 끝점 탐색 컨볼루션화, 분기점에서 진행 방향 유지, 끊김 점프 KD-tree, 학습 시 특징 캐시.

## 자산

- `../assets/pac_supermoon/` — 실로봇 URDF (PAC_Supermoon, D405 홀더, FK 타깃 `tcp_link`)
- `../assets/so101/` — 기본 SO-ARM100 URDF (참고. FK는 더 이상 이 파일을 쓰지 않음)
- `../assets/so101_isaac/` — IsaacLab USD. 아직 기본 SO-101. 실로봇 끝단과 다름 (후속)

## 알려진 제약 / TODO

- `SO101BCDataset`은 `observation.state`/`action`이 `kinematics.JOINT_NAMES` 순서인지 메타로 검사하고
  다르면 오류를 낸다 (REORDER_INDEX 보정은 아직 없음).
- gripper 채널(7번째)은 녹화값(0~100)을 그대로 통과한다. 이 채널이 펌프 0/1인지 집게 값인지, 집게 값이면
  실로봇(민제씨 캘리브)에 어떤 경로로 가는지는 **팀 회의에서 결정 예정**. 펌프 0/1 자리는 송지수 선배
  ActionChunk의 `eef` 필드.
- ActionChunk 직렬화(추론 출력 → 제어 프로세스)는 아직 없음. `bc_inference.predict_chunk` 출력을 쓰면 됨.
- seam CV 특징(5dim, `so101_bc_dataset.py::_seam_features`)은 depth 없이 2D 픽셀 기준 축약 벡터.
  Wrist RGBD를 실제로 쓸 때는 `seam_cv.py::detect()`에 depth+intrinsics를 넘겨 3D 특징으로 확장할 것.
- RL 환경의 목표 경로는 아직 절차적 직선 생성(ground truth)이며, 실제 카메라+seam_cv 인식을
  루프에 넣는 건 후속 작업 (`envs/so101_seam_env.py` 모듈 docstring 참고). 카메라도 아직 미연결(placeholder).
- 지연보정 모듈은 이 폴더에 아직 없음 — 다음 작업 대상.
