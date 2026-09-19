# AI 추론 계층 구현 — BC(ACT) 프레임워크

설계 문서: [`../docs/AI_추론계층_프레임워크.md`](../docs/AI_추론계층_프레임워크.md). 이 폴더는 그 문서
3.3절(BC)을 실제로 동작하는 코드로 구현한 것이다. RL(SAC, 3.4절)과 지연보정(4장)은 아직 미구현.

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
  data/so101_bc_dataset.py   lerobot 원본(관절공간) 데이터셋 -> EEF-delta + seam 특징 변환 wrapper
  train_bc.py                학습 진입점 (lerobot ACTPolicy 그대로 사용)
```

## 엔드투엔드 흐름

```
1. 데이터 수집 (SO-101 leader로 티칭, 실물)
   lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM_follower \
                   --teleop.type=so101_leader --teleop.port=/dev/ttyACM_leader \
                   --dataset.repo-id=<user>/so101-weld-demo ...

2. BC 학습
   python -m ai_layer.train_bc --repo-id <user>/so101-weld-demo --epochs 100

3. (RL 단계는 아직 미구현 — docs 3.4절 참고, SAC + 3항 보상)
```

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
- RL(SAC) 단계, 지연보정 모듈은 이 폴더에 아직 없음 — 다음 작업 대상.
