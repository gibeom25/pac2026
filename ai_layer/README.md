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
  kinematics.py              SO-101 URDF 기반 FK + EEF-delta 변환 (docs 0절 "EEF-delta 통일")
  perception/seam_cv.py      Seam/Groove 고전 CV 모듈 (docs 3.1절)
  configs/so101_act_bc.py    ACTConfig 프리셋: chunk_size=32, use_vae=False, EEF-delta 7dim 액션
  configs/so101_sac.py       SACConfig 프리셋: 연속 6dim(EEF-delta) + 이산 1(그리퍼), discount=0.97
  data/so101_bc_dataset.py   lerobot 원본(관절공간) 데이터셋 -> EEF-delta + seam 특징 변환 wrapper
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
1. 데이터 수집 (SO-101 leader로 티칭, 실물)
   lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM_follower \
                   --teleop.type=so101_leader --teleop.port=/dev/ttyACM_leader \
                   --dataset.repo-id=<user>/so101-weld-demo ...

2. BC 학습
   python -m ai_layer.train_bc --repo-id <user>/so101-weld-demo --epochs 100

3. RL(SAC) 학습 — ⚠️ IsaacLab 필요, 반드시 사용자 터미널에서 직접 실행 (이 코딩 세션의 샌드박스
   에서는 CUDA P2P 검증 단계에서 멈춤 — 알려진 세션 제약, 하드웨어 문제 아님)
   cd /home/robot/IsaacLab
   ./isaaclab.sh -p /home/robot/pac2026/ai_layer/train_rl.py --headless \
       --bc-checkpoint /home/robot/pac2026/outputs/bc_act/act_epoch0099.pt
```

## 검증 상태

- ✅ `kinematics.py`, `perception/seam_cv.py`, `rl/reward.py`, `rl/replay_buffer.py`: 이 세션에서 직접 단위 테스트 통과.
- ✅ `configs/so101_act_bc.py` + lerobot `ACTPolicy`: 더미 배치로 forward/inference 검증 완료.
- ✅ `configs/so101_sac.py` + lerobot `SACPolicy`: 더미 배치로 critic/discrete_critic/actor/temperature loss·업데이트·`select_action`까지 전부 검증 완료.
- ✅ `sim/smoke_test_so101.py`: SO-101 USD 로드 + 관절 인식 + 시뮬레이션 스텝, 사용자 터미널에서 성공 확인.
- ⚠️ `envs/so101_seam_env.py`: IsaacLab 의존성 때문에 이 세션에서 미검증 (문법 체크만 통과). 사용자 터미널에서 최초 실행 시 디버깅 필요할 수 있음 — 특히 `DifferentialIKController` 연동, 관측 스페이스 반환 형식.

## 자산

- `../assets/so101/` — 공식 SO-ARM100 저장소의 SO-101 URDF + 메시 (FK/시각화용)
- `../assets/so101_isaac/` — NVIDIA 공식 Sim-to-Real-SO-101-Workshop의 IsaacLab USD 자산 + `ArticulationCfg`
  (URDF→USD 수동 변환 대신 재사용, 관절 이름 매핑 주의 — 해당 폴더 README 참고)

## 알려진 제약 / TODO

- `SO101BCDataset`은 `observation.state`/`action`이 `kinematics.JOINT_NAMES` 순서로 저장되어 있다고
  가정한다 — 실제 `lerobot-record`로 첫 데이터셋을 만들면 `dataset.meta.features`로 순서를 반드시 검증할 것.
- gripper 채널은 현재 그대로 통과(관절각) — pump 신호(0/1)로 바꾸려면 이 파일에 임계값 매핑 추가 필요.
- seam CV 특징(5dim, `so101_bc_dataset.py::_seam_features`)은 depth 없이 2D 픽셀 기준 축약 벡터.
  Wrist RGBD를 실제로 쓸 때는 `seam_cv.py::detect()`에 depth+intrinsics를 넘겨 3D 특징으로 확장할 것.
- RL 환경의 목표 경로는 아직 절차적 직선 생성(ground truth)이며, 실제 카메라+seam_cv 인식을
  루프에 넣는 건 후속 작업 (`envs/so101_seam_env.py` 모듈 docstring 참고). 카메라도 아직 미연결(placeholder).
- 지연보정 모듈은 이 폴더에 아직 없음 — 다음 작업 대상.
