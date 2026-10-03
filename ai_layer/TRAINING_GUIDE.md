# 학습 가이드 (데이터 수집 → BC → RL)

2026-10-03 기준 현재 파이프라인(EE-only 리그 + 조이스틱 + balanced 데이터 수집) 전체를 처음부터
끝까지 돌리는 실전 순서다. 각 단계는 이전 단계가 통과해야 다음으로 넘어가는 게 맞다 — 중간에
건너뛰면 다음 단계에서 원인 찾기 어려운 에러가 난다.

아키텍처/설계 배경은 [`README.md`](README.md)(3.3절 BC/3.4절 RL 구현 설명)와
[`tools/README.md`](tools/README.md)(도구별 상세 옵션)를 참고. 이 문서는 "지금 뭘 실행해야
하는가"에 집중한 실행 가이드다.

## 0. 환경

```bash
conda activate pac2026
cd ~/pac2026   # 또는 저장소 루트
```

모든 명령은 `PYTHONPATH=.` 를 붙여서 저장소 루트에서 실행한다(아래 예시 전부 포함돼 있음).

사전 확인:
- MuJoCo 뷰어 창이 뜨는가 (시뮬레이션 도구는 기본적으로 GUI 창을 띄운다 — `--headless`는 opt-in):
  ```bash
  PYTHONPATH=. python ai_layer/tools/check_ee.py --scene curve
  ```
  스틱을 움직여 EE가 기대한 방향(앞/뒤=x, 좌/우=y)으로 가는지, 슬라이더로 z가 오르내리는지,
  베이스 버튼으로 회전이 도는지, 트리거로 비드가 찍히는지 확인. 방향이 반대면
  `joystick_input.py`의 `ee_velocity()`/`rotation_rate()` 호출부 `--invert-*` 플래그로 고친다.
- Logitech Extreme 3D Pro가 안 잡히면 `joystick_input.py`를 단독 실행해서 눌린 버튼 이름이
  출력되는지 확인 (버튼 매핑 디버그용).

## 1단계 — 데이터 수집 (`tools/record_mujoco.py`)

```bash
PYTHONPATH=. python ai_layer/tools/record_mujoco.py \
    --repo-id <본인id>/so101-weld-demo --num-episodes 30
```

- `--root`를 안 주면 `datasets/<repo-id>`에 저장된다(`.gitignore`에 포함돼 커밋 안 됨). HF 캐시가
  아니므로 팀원끼리 경로가 흩어지지 않는다.
- `--scene`은 기본값 `balanced` — 에피소드마다 6형태(`curve`/`straight`/`sharp_curve`/`corner`/
  `branch`/`dashed`) × 5variant = 30가지 조합 중 **지금까지 가장 적게 기록된 조합**을 무작위로
  골라 돌아간다. 한 번에 30개를 다 못 모아도(세션을 여러 날 나눠 돌려도) `datasets/<repo-id>/
  meta/scene_balance.json`에 누적 카운트가 저장되므로 전체 분포는 계속 균등하게 수렴한다 —
  "오늘은 curve만 모아야지" 식으로 신경 쓸 필요 없이 그냥 계속 돌리면 된다.
  - 특정 형태만 집중적으로 모으고 싶으면 `--scene dashed` 처럼 이름을 직접 주면 그 형태 안에서만
    (variant로) 균형 샘플링한다.
- 조작: 트리거를 누르고 있는 동안 비드가 찍힌다. **BTN_THUMB**로 그 자리에서 저장+종료(에피소드
  길이는 기본 무제한), **BTN_THUMB2**로 폐기+재시도. 막대가 바닥/용지에 닿으면 자동 폐기된다 —
  도구는 표면에 닿지 않고 살짝 띄운 채로 작업해야 한다.
- 에피소드 사이 Enter로 다음 녹화 시작. Ctrl+C로 중단해도 그때까지 저장된 에피소드는 유지된다.

## 2단계 — 데이터 점검 (`tools/check_dataset.py`)

**학습 전에 반드시 통과시킬 것.**

```bash
PYTHONPATH=. python ai_layer/tools/check_dataset.py \
    --repo-id <본인id>/so101-weld-demo --root datasets/<본인id>/so101-weld-demo
```

fps, observation.state/action 차원, 이미지 키, 그리퍼(트리거) 바이너리 여부, 에피소드 길이,
timestamp 간격, 청크 증분 크기(너무 빠르게 티칭하면 제어 validator가 자르는 비율), seam 인식
실패율(카메라에 선이 안 보인 프레임 비율)을 확인한다. ❌가 하나라도 있으면 원인을 고치고
다시 녹화 — 특히 "선 인식 실패" ❌가 뜨면 `tools/seam_preview.py`로 `SeamCVConfig`를 먼저
튜닝할 것.

## 3단계 — BC(모방학습) 학습 (`train_bc.py`)

```bash
PYTHONPATH=. python ai_layer/train_bc.py \
    --repo-id <본인id>/so101-weld-demo --root datasets/<본인id>/so101-weld-demo \
    --epochs 100
```

- `robot_type`으로 데이터셋 포맷(EE-native/관절공간)을 자동 판별해서 맞는 Dataset 클래스를
  고른다 — 설정할 것 없음.
- 먼저 `--epochs 20~30 --batch-size 8` 정도로 짧게 돌려서 loss가 내려가는지(과적합 시험) 확인한
  다음 본 학습으로 늘리는 걸 추천.
- 결과: `outputs/bc_act/last/`(config.json, model.safetensors, preprocessor/postprocessor —
  정규화 통계까지 포함된 폴더 하나, `outputs/`도 gitignore 대상).

## 4단계 — RL 환경 점검 (`tools/rl_env_smoke.py`)

RL을 오래 돌리기 전에 환경 자체가 멀쩡한지 먼저 확인 — 몇 초면 끝난다.

```bash
PYTHONPATH=. python ai_layer/tools/rl_env_smoke.py --bc-checkpoint outputs/bc_act/last
```

홈 위치, 3초 드리프트, 액션 스케일 추종, **바닥 접촉 시 즉시 종료되는지**, **선에서 멀 때
트리거를 눌러도 비드가 강제로 꺼지는지**(안전 컷오프), BC teacher 보상 계산까지 6가지를
확인한다. `--bc-checkpoint`는 선택(3단계 체크포인트가 있으면 같이 확인, 없으면 생략 가능).

## 5단계 — RL(강화학습) 학습 (`train_rl.py`)

```bash
PYTHONPATH=. python ai_layer/train_rl.py \
    --num-steps 200000 --bc-checkpoint outputs/bc_act/last
```

- `--bc-checkpoint`를 주면 BC 정책을 "참고"(R_imitation, 가중치 1.0 → 0.3로 학습이 진행될수록
  점점 줄어듦 — 처음엔 BC를 많이 따라가다가 점점 스스로 판단하는 비중이 커짐)하는 teacher로
  쓴다. 안 주면 R_imitation=0으로 순수 R_track(선 추종)+R_smooth+R_coverage만으로 학습한다.
- 매 에피소드 씬(형태+variant)이 무작위로 바뀐다 — 한 가지 모양에 과적합되지 않게.
- 안전장치(코드 수정 없이 항상 켜져 있음): 선에서 3cm(`SO101SeamEnvCfg.off_seam_safety_dist`)
  이상 떨어지면 정책이 트리거를 켜도 비드 분사가 강제로 꺼진다. 막대가 바닥/용지에 닿으면 그
  즉시 에피소드가 끝나고 큰 음의 보상이 들어간다(`floor_contact_penalty`) — 데이터 수집 때
  "접촉=자동 폐기"와 같은 규약을 RL 보상으로 반영한 것.
- 결과: `outputs/rl_sac/sac_final`(및 `--ckpt-every` 간격 중간 체크포인트).
- 학습 속도 체감이 필요하면 먼저 `--num-steps 2000` 정도로 짧게 돌려서 `critic_loss`/`reward`
  로그가 정상 범위에서 움직이는지 보고, 문제 없으면 본 학습(수만~수십만 스텝)으로 늘릴 것 —
  오래 걸리는 작업이니 백그라운드로 돌리는 걸 추천.

## 요약 (복붙용)

```bash
conda activate pac2026
cd ~/pac2026

# 1. 수집 (balanced 샘플링, 30개 모으면 6형태 고르게 들어감)
PYTHONPATH=. python ai_layer/tools/record_mujoco.py --repo-id me/so101-weld-demo --num-episodes 30

# 2. 점검 (❌ 없을 때까지)
PYTHONPATH=. python ai_layer/tools/check_dataset.py --repo-id me/so101-weld-demo --root datasets/me/so101-weld-demo

# 3. BC 학습
PYTHONPATH=. python ai_layer/train_bc.py --repo-id me/so101-weld-demo --root datasets/me/so101-weld-demo --epochs 100

# 4. RL 환경 스모크
PYTHONPATH=. python ai_layer/tools/rl_env_smoke.py --bc-checkpoint outputs/bc_act/last

# 5. RL 학습
PYTHONPATH=. python ai_layer/train_rl.py --num-steps 200000 --bc-checkpoint outputs/bc_act/last
```

## 문제가 생기면

| 증상 | 확인할 것 |
|---|---|
| 뷰어 창이 안 뜸 | `--headless`를 실수로 안 줬는지(기본은 뜸), GPU/디스플레이 환경(`DISPLAY`) |
| 조이스틱 방향이 반대 | `joystick_input.py` 단독 실행으로 버튼/축 이름 확인 후 `--invert-x/y/z/roll/pitch` |
| `check_dataset.py`에서 선 인식 실패율 높음 | `tools/seam_preview.py`로 `SeamCVConfig` 재조정 |
| `check_dataset.py`에서 증분 크기 초과 비율 높음 | 녹화할 때 조이스틱을 더 천천히 움직이거나 `--max-linear-speed` 낮춰서 재녹화 |
| BC loss가 안 내려감 | 에피소드 수/다양성 부족(1단계 balanced 분포 확인), `--epochs` 늘리기, 데이터셋에
`check_dataset.py` ❌ 없는지 재확인 |
| RL reward가 이상하게 널뜀 | 먼저 `rl_env_smoke.py` 통과하는지, `--bc-checkpoint` 없이(R_imitation=0) 먼저 돌려서
R_track만으로도 학습이 되는지 분리 확인 |
| RL이 선 밖에서도 계속 비드를 뿌림 | 안전 컷오프는 "비드 분사"만 강제로 막지 "이동"은 막지 않는다 — 정책이 선을
못 따라가는 거면 R_track/BC teacher 쪽 문제(위 BC loss 항목부터 확인) |
