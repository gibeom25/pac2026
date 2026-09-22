# 로봇 오는 날 체크리스트 (AI 추론 파트)

목표: 녹화 → 점검 → 학습 → 제어 연결까지 **하루 안에 한 바퀴** 돌려본다. 품질은 그 다음.

## 0. 전날
- [ ] `env_lerobot` 동작 확인: `PYTHONPATH=. python ai_layer/tools/bc_synthetic_test.py`
- [ ] 다리 동작 확인: `PYTHONPATH=. python ai_layer/tools/chunk_bridge_test.py`
- [ ] `record_so101.sh` 의 `<...>` 자리 채울 값 목록 준비 (포트, 카메라 시리얼)
- [ ] 회의 결정 반영: 7번째 채널 → `ai_node.py --eef-mode ...`
- [ ] 선 인식 튜닝: 실제 선/재료 사진으로 `tools/seam_preview.py` 돌려 `SeamCVConfig` 확정

## 1. 로봇 연결 (민제씨·송지수 선배와)
- [ ] 팔로워/리더 포트: `lerobot-find-port`
- [ ] 캘리브: `lerobot-calibrate` 팔로워/리더 각각 (record_so101.sh 상단 참고)
- [ ] D405 시리얼: `lerobot-find-cameras realsense`
- [ ] 텔레옵으로 팔이 리더를 따라오는지 확인
- [ ] 리더암을 **천천히** 움직이는 연습. 제어 한계는 한 걸음(33 ms)에 5 mm = **0.15 m/s**. 이보다 빠르면 다리가 잘라서 로봇이 느려진다.

## 2. 녹화
- [ ] `ai_layer/tools/record_so101.sh` (에피소드 20개, 30초씩부터)
- [ ] 첫 에피소드 1개 녹화 후 바로 `check_dataset.py` → 문제 없으면 계속
- [ ] 선 종류(펜/실리콘/분필)별로 최소 5 에피소드

## 3. 점검 → 학습
- [ ] `check_dataset.py` 전부 ✅ (fps 30, 관절 순서, 이미지 키, 단위, 증분 크기, 선 인식률)
- [ ] 과적합 시험: `train_bc.py --epochs 30 --batch-size 8` → loss 가 내려가는지
- [ ] 체크포인트 `outputs/bc_act/last/` 생성 확인

## 4. 제어 연결
- [ ] 송지수 선배 제어: `run_live.py --ai external` (또는 실로봇 HAL)
- [ ] `ai_node.py --checkpoint outputs/bc_act/last --camera realsense --anchor commit --eef-mode <회의값> --dry-run` 먼저 (송신 없이 로그만)
- [ ] `--dry-run` 빼고 실연결. 제어 로그의 `ingest accepted` 와 `validator rejected` 확인
  - `step_pos_limit`/`step_speed_limit` → 티칭이 빨랐음. 다리 클립이 걸리는 중
  - `workspace` → 모델이 작업공간 밖으로 밈 (데이터 부족/불량)
  - `too_old` → 추론이 300 ms 넘게 걸림 (GPU 확인)

## 5. 끝나기 전
- [ ] 데이터셋 폴더 백업 (`/home/dy/pac2026/datasets/`)
- [ ] 카메라 촬영시각 ↔ 측정시각 오프셋 대략값 기록 (후속 캘리브용)
