# SO-101 IsaacLab 자산 (NVIDIA 공식 재사용)

출처: [isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) (Apache-2.0, `LICENSE_NVIDIA` 참고).
사용자가 공유한 NVIDIA 공식 학습자료(https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html)에서 링크된 워크숍 저장소.

URDF를 직접 USD로 변환하는 대신, NVIDIA가 이미 검증한 USD 자산과 IsaacLab `ArticulationCfg`를
그대로 재사용한다. **GR00T/Docker 기반 워크숍 전체 파이프라인은 채택하지 않음** — 이 프로젝트는
자체 BC(ACT)+RL(SAC) 아키텍처를 쓰므로, 여기서 가져오는 건 로봇 USD 자산과 액추에이터 게인뿐이다.

## 파일
- `usd/SO-ARM101-USD.usd` — 카메라 포함 SO-101 USD (23MB)
- `usd/SO-ARM101-USD-NO-CAMERA.usd` — 카메라 제외 버전 (관절 동역학만 필요할 때 가벼움)
- `so101_cfg.py` — IsaacLab `ArticulationCfg` (USD 경로, 초기 관절각, 관절별 stiffness/damping — NVIDIA가 실측 기어비/토크 스펙 기반으로 튜닝한 값)

## ⚠️ 관절 이름 매핑 (lerobot ↔ IsaacLab/USD)

같은 물리적 관절이지만 두 곳에서 이름이 다르다. EEF-delta 변환(`ai_layer/kinematics.py`, lerobot
관절 순서 기준)과 IsaacLab 시뮬레이션(`so101_cfg.py`, USD 관절 이름 기준)을 연결할 때 반드시 이 표로
매핑할 것 — 순서를 혼동하면 FK/IK 결과가 조용히 틀어진다.

| lerobot (motor name) | IsaacLab/USD (joint name) |
|---|---|
| shoulder_pan | Rotation |
| shoulder_lift | Pitch |
| elbow_flex | Elbow |
| wrist_flex | Wrist_Pitch |
| wrist_roll | Wrist_Roll |
| gripper | Jaw |

## 하드웨어 참고 (RTX 4060 8GB 노트북 제약)

NVIDIA 워크숍은 RTX 6000 Pro / RTX 5090 / RTX 6000 Ada 급에서만 테스트됨(GR00T 파운데이션 모델
파인튜닝 전제). 이 프로젝트의 GPU(8GB)로는 GR00T 파인튜닝은 비현실적이므로, 가벼운 ACT+SAC
조합(docs/AI_추론계층_프레임워크.md)을 유지하는 근거가 된다.
