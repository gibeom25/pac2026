# 실행 명령어 모음

시뮬레이션 데이터 생성 → 사전학습 → 실물 녹화 → co-training → 실물 추론까지의 명령어.
모든 명령은 저장소 루트에서 실행한다. `<id>`는 본인/팀 이름으로 바꿀 것.
배경 설명과 단계별 주의사항은 [`ai_layer/TRAINING_GUIDE.md`](ai_layer/TRAINING_GUIDE.md) 참고.

## 0. 환경 (처음 한 번)

```bash
uv sync
```

`pyproject.toml`/`uv.lock` 기준으로 `.venv/`가 만들어진다(Python 3.10, lerobot 0.4.4, torch cu128).

## 1. 시뮬레이션 데이터 자동 생성

```bash
# 생성 과정을 눈으로 확인 (2개, 실시간 속도)
PYTHONPATH=. uv run python ai_layer/tools/generate_demos.py --repo-id <id>/sim-test --num-episodes 2 --view

# 본 생성 (창 없이, 1000개 ≈ 5GB). --workers로 CPU 코어를 나눠 쓴다(1개면 ≈ 35분)
PYTHONPATH=. uv run python ai_layer/tools/generate_demos.py --repo-id <id>/so101-weld-scripted --num-episodes 1000 --workers 24

# 중간에 멈췄으면 이어서 (--resume은 --workers 없이만 된다)
PYTHONPATH=. uv run python ai_layer/tools/generate_demos.py --repo-id <id>/so101-weld-scripted --num-episodes 500 --resume
```

- 저장 위치: `datasets/<repo-id>/`. 기존 데이터셋이 있으면 `--resume`(이어서) 또는 `--overwrite`(새로)를 줘야 한다.
- 그리다가 잠깐 옆으로 밀려났다 돌아오는 건 **외란**이다. 의도된 동작이며, 경로를 벗어났을 때 돌아오는 법을 학습시키려는 것이다. 끄려면 `--max-disturbances 0`.
- 점선은 선이 끊긴 구간에서 분사를 멈춘다. 끊긴 구간도 이어서 그리려면 `--dashed bridge`.
- 데이터셋에는 손목 카메라만 저장된다. 고정 카메라 칸은 학습할 때 검은 화면으로 채워진다. 정책에서 그 칸을 아예 빼려면 학습 때 `--no-overview`를 준다.

### (권장) Piper 부스 시뮬레이션 데이터

실물 Piper 부스(흰 상자, 고정캠, 손목캠)를 본뜬 시뮬레이션이다. 실물 데이터셋과 좌표계/형식이 같고 고정캠
화면도 렌더된다(위 1번 시뮬은 로봇 팔이 없어서 고정캠 칸이 검은 화면). 보정 근거와 가정은
`ai_layer/envs/piper_sim.py` 맨 위 설명 참고 — 펜 끝 위치와 로봇 베이스 높이는 실측해서 고칠 것.

```bash
# 생성 (1000개 ≈ 15분, 워커 16개)
PYTHONPATH=. uv run python ai_layer/tools/generate_piper_demos.py --repo-id <id>/piper-sim --num-episodes 1000 --workers 16

# 사전학습 → 실물 파인튜닝 (실물 에피소드 ep%10==4는 평가용으로 뺌)
PYTHONPATH=. uv run python ai_layer/train_bc.py --repo-id <id>/piper-sim --epochs 5 --batch-size 64 --out-dir outputs/bc_psim
PYTHONPATH=. uv run python ai_layer/train_bc.py --repo-id <실물 repo> <id>/piper-sim --weights 2 1 \
    --holdout-every 10 0 --holdout-offset 4 --init-from outputs/bc_psim/last --epochs 3 --batch-size 64 --out-dir outputs/bc_piper_psim
```

## 2. 데이터 점검

```bash
PYTHONPATH=. uv run python ai_layer/tools/check_dataset.py --repo-id <id>/so101-weld-scripted
```

❌가 하나라도 있으면 학습 전에 원인을 고칠 것.

## 3. 시뮬레이션 사전학습

```bash
PYTHONPATH=. uv run python ai_layer/train_bc.py --repo-id <id>/so101-weld-scripted --epochs 50 --out-dir outputs/bc_sim
```

처음엔 `--epochs 2 --batch-size 8`로 loss가 내려가는지만 확인하고 본 학습으로 넘어갈 것.

## 4. 실물 데이터 녹화 (부스마다)

```bash
PYTHONPATH=. uv run python ai_layer/gui/app.py
```

GUI 설정:

1. 소스 = **실로봇**
2. repo-id는 부스마다 다르게: `<id>/real-booth1`, `<id>/real-booth2`, ...
3. **손목 카메라 index**와 **오버뷰 카메라 index**를 둘 다 입력
4. "조작 테스트"로 움직여 본 다음 녹화 시작

녹화 후 고정 카메라도 저장됐는지 확인한다. 결과에 `observation.images.overview`가 보여야 한다.

```bash
PYTHONPATH=. uv run python ai_layer/tools/check_dataset.py --repo-id <id>/real-booth1
```

## 5. 실물 + 시뮬레이션 co-training

```bash
PYTHONPATH=. uv run python ai_layer/train_bc.py \
    --repo-id <id>/real-booth1 <id>/real-booth2 <id>/so101-weld-scripted \
    --weights 1 1 1 --init-from outputs/bc_sim/last --epochs 50 --out-dir outputs/bc_cotrain
```

- `--weights`: 데이터셋 크기와 무관한 샘플링 비율이다(위 예시는 부스1 : 부스2 : 시뮬 = 1 : 1 : 1).
- 부스 하나는 학습에 넣지 말고 평가용으로 남겨둘 것.

## 6. 실물 추론

```bash
# 배관 점검 (로봇 없이)
PYTHONPATH=. uv run python ai_layer/control_bridge/ai_node.py --dry-run --fake-snapshot --iterations 3 --checkpoint outputs/bc_cotrain/last

# 실제 연결 (N = 부스 고정 카메라 OpenCV index)
PYTHONPATH=. uv run python ai_layer/control_bridge/ai_node.py --checkpoint outputs/bc_cotrain/last --camera realsense --overview-index N
```

고정 카메라를 연결하지 않으면 그 입력은 검은 화면으로 채워져서 손목 카메라만으로 동작한다.

## 평가

```bash
# 오프라인: 학습에서 뺀 실물 시연에서 1초 경로 오차 [mm] (holdout은 체크포인트 train_info.json에서 읽음)
PYTHONPATH=. uv run python ai_layer/tools/eval_bc_offline.py --repo-id <실물 repo> --checkpoint outputs/A/last outputs/B/last

# 시뮬 closed-loop: Piper 부스 시뮬에서 직접 그려 보고 coverage/progress/추종 오차 채점 (--video-dir로 영상 저장)
PYTHONPATH=. uv run python ai_layer/tools/eval_piper_sim.py --checkpoint outputs/A/last outputs/B/last --episodes-per-scene 2
```

시뮬 점수는 실물 점수가 아니다 — 같은 시뮬에서 모델끼리 순위를 매기는 용도.

## (선택) RL

```bash
PYTHONPATH=. uv run python ai_layer/tools/rl_env_smoke.py --bc-checkpoint outputs/bc_sim/last
PYTHONPATH=. uv run python ai_layer/train_rl.py --num-steps 200000 --bc-checkpoint outputs/bc_sim/last
```

RL 보상도 생성기와 같은 점선 규칙(끊긴 구간에서 분사 멈춤)을 쓴다(`SO101SeamEnvCfg.dashed`).

## 자주 바꾸는 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `generate_demos.py --max-tilt-deg` | 5 | 펜 위쪽이 로봇 쪽으로 기우는 최대 각도 [°] |
| `generate_demos.py --hover-mm` | 9 11 | 펜 끝 높이 범위 [mm] (실물은 약 1cm) |
| `generate_demos.py --bead-color` | random | `fixed`면 미색 고정 |
| `generate_demos.py --max-disturbances` | 2 | 에피소드당 외란 횟수, 0이면 끔 |
| `generate_demos.py --dashed` | cut | `bridge`면 점선의 끊긴 구간도 이어서 그림 |
| `generate_demos.py --seed` | 0 | `--seed`와 `--workers`가 같으면 같은 데이터가 나온다 |
| `generate_demos.py --workers` | 1 | 병렬 생성 프로세스 수. 조각으로 나눠 만든 뒤 하나로 합친다 |
| `train_bc.py --overview-dropout` | 0.3 | 실물 고정 카메라를 학습 중 가리는 확률 |
| `train_bc.py --no-overview` | 꺼짐 | 손목 카메라만 사용. 사전학습과 co-training에 똑같이 줘야 한다 |
| `train_bc.py --holdout-every N.. --holdout-offset K` | 없음 | 데이터셋별로 episode % N == K인 에피소드를 학습에서 빼 평가용으로 남김 (0이면 안 뺌) |
