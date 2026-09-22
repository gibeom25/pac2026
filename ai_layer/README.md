# AI 추론 계층 구현 — BC(ACT) + RL(SAC) 프레임워크

설계 문서: [`../docs/AI_추론계층_프레임워크.md`](../docs/AI_추론계층_프레임워크.md). 이 폴더는 그 문서
3.3절(BC), 3.4절(RL)을 실제로 동작하는 코드로 구현한 것이다. 지연보정(4장)은 아직 미구현.

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
  kinematics.py              SO-101 URDF 기반 FK/IK + EEF-delta 변환 (docs 0절 "EEF-delta 통일")
  perception/seam_cv.py      Seam/Groove 고전 CV 모듈 (docs 3.1절)
  configs/so101_act_bc.py    ACTConfig 프리셋: chunk_size=32, use_vae=False, EEF-delta 7dim 액션
  configs/so101_sac.py       SACConfig 프리셋: 연속 6dim(EEF-delta) + 이산 1(그리퍼), discount=0.97
  data/so101_bc_dataset.py   lerobot 원본(관절공간) 데이터셋 -> EEF-delta + seam 특징 변환 wrapper
  rl/reward.py                R_imitation + R_track + R_smooth (docs 3.4절), 가중치 스케줄링
  rl/replay_buffer.py         단일 프로세스 SAC용 최소 리플레이 버퍼
  envs/so101_seam_env.py      MuJoCo 기반 Gymnasium 환경 — SO-101 + 절차적 경로 + placo IK
  train_bc.py                 BC 학습 진입점 (lerobot ACTPolicy 그대로 사용)
  train_rl.py                 RL 학습 진입점 (lerobot SACPolicy 그대로 사용, BC 체크포인트를
                               R_imitation teacher로 선택적 로드) — 일반 파이썬 스크립트, IsaacLab
                               의존성 없음
```

## 엔드투엔드 흐름

⚠️ **1단계(실물 데이터 수집)는 아직 시작 전**: SO-101 leader 하드웨어 통신 문제(장시간 디버깅 끝에
원인이 USB-시리얼 어댑터 보드 자체의 하드웨어 결함으로 확인됨)는 보드 교체로 해결됨 — 캘리브레이션까지
완료된 상태. 2~3단계(BC/RL 학습 코드와 시뮬레이션 파이프라인)는 하드웨어와 무관하게 이미 동작 확인됨.

```
1. 데이터 수집 (SO-101 leader로 티칭, 실물)
   lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM_follower \
                   --teleop.type=so101_leader --teleop.port=/dev/ttyACM_leader \
                   --dataset.repo-id=<user>/so101-weld-demo ...

2. BC 학습
   python -m ai_layer.train_bc --repo-id <user>/so101-weld-demo --epochs 100

3. RL(SAC) 학습 — IsaacLab 불필요, 아무 터미널에서나(이 세션 포함) 바로 실행 가능
   python -m ai_layer.train_rl --num-steps 200000 \
       --bc-checkpoint outputs/bc_act/act_epoch0099.pt
```

## 검증 상태

- ✅ `kinematics.py`(FK+IK), `perception/seam_cv.py`, `rl/reward.py`, `rl/replay_buffer.py`: 이 세션에서 직접 단위 테스트 통과.
- ✅ `configs/so101_act_bc.py` + lerobot `ACTPolicy`: 더미 배치로 forward/inference 검증 완료.
- ✅ `configs/so101_sac.py` + lerobot `SACPolicy`: 더미 배치로 critic/discrete_critic/actor/temperature loss·업데이트·`select_action`까지 전부 검증 완료.
- ✅ `envs/so101_seam_env.py` (MuJoCo): 이 세션에서 직접 실행 — reset/step, FK+IK 기반 EEF-delta 액션 적용, 관측/보상 생성까지 정상 동작 확인.
- ✅ `train_rl.py`: 이 세션에서 직접 실행 — 실제 SAC 학습 루프(critic/actor/temperature 업데이트, replay buffer)가 수백 스텝 동안 정상 진행되는 것을 확인(2026-09-22).
- ⚠️ 학습 속도: 이 세션 환경에서 학습 스텝당 대략 0.2초 내외 — 장시간(수만~수십만 스텝) 학습은 실제 학습용 머신에서 백그라운드로 돌리는 걸 권장.

## 자산

- `../assets/so101/` — 공식 SO-ARM100 저장소의 SO-101 URDF + 메시(FK/IK용) + **MJCF**(`so101_new_calib.xml`,
  MuJoCo 물리 시뮬레이션용, 실측 서보 게인 반영됨)

## 알려진 제약 / TODO

- `SO101BCDataset`은 `observation.state`/`action`이 `kinematics.JOINT_NAMES` 순서로 저장되어 있다고
  가정한다 — 실제 `lerobot-record`로 첫 데이터셋을 만들면 `dataset.meta.features`로 순서를 반드시 검증할 것.
- gripper 채널은 현재 그대로 통과(관절각) — pump 신호(0/1)로 바꾸려면 이 파일에 임계값 매핑 추가 필요.
- seam CV 특징(5dim, `so101_bc_dataset.py::_seam_features`)은 depth 없이 2D 픽셀 기준 축약 벡터.
  Wrist RGBD를 실제로 쓸 때는 `seam_cv.py::detect()`에 depth+intrinsics를 넘겨 3D 특징으로 확장할 것.
- RL 환경의 목표 경로는 아직 절차적 직선 생성(ground truth)이며, 실제 카메라+seam_cv 인식을
  루프에 넣는 건 후속 작업 (`envs/so101_seam_env.py` 모듈 docstring 참고). 카메라도 아직 미연결(placeholder)
  — MuJoCo는 `mujoco.Renderer`로 오프스크린 렌더링이 가벼워서 이 작업은 IsaacLab보다 수월할 전망.
- 지연보정 모듈은 이 폴더에 아직 없음 — 다음 작업 대상.
- SO-101 leader로 실제 데이터 수집·BC 학습·RL fine-tuning 실행은 아직 미착수.
