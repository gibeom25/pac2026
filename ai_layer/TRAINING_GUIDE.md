# 학습 가이드 (데이터 수집 → BC → RL)

2026-10-06 기준 현재 파이프라인(EE-only 리그 + 조이스틱 + balanced 데이터 수집) 전체를
처음부터 끝까지 돌리는 실전 순서다. 각 단계는 이전 단계가 통과해야 다음으로 넘어가는 게 맞다 —
중간에 건너뛰면 다음 단계에서 원인 찾기 어려운 에러가 난다.

**최근 변경사항 요약** (이 문서가 안 맞는 것 같으면 먼저 여기부터 확인):
- ⚠️ 데이터 수집 GUI(`tools/record_gui.py`)를 시도했는데 이 머신에서 간헐적으로 `X Error ...
  BadAccess ... X_GLXMakeCurrent`로 죽는 문제가 있어 **보류**했다 — 재현 조건을 못 찾았고(같은
  코드를 그대로 여러 번 돌려도 될 때도 안 될 때도 있음), 코드는 남겨뒀지만 지금은 **터미널
  (`record_mujoco.py`)을 기본으로 쓸 것**. GUI를 다시 시도하고 싶으면 tools/README.md의
  record_gui.py 절 참고.
- 녹화 중 바닥 접촉은 더 이상 에피소드를 폐기하지 않는다 — 도구 끝이 1cm 밑으로 안 내려가게
  막는 걸로 바뀜(1단계 참고). **RL 쪽은 그대로 접촉=즉시 종료+패널티**(5단계 참고) — 데이터
  수집과 RL 환경의 규약이 이제 서로 다르다는 점에 주의.
- 조이스틱 스로틀(z축) 영점을 처음 한 번만 물어보고 저장해서 재사용한다(0단계 참고).
- `train_rl.py`가 `metrics.jsonl`에 보상/BC 추종 점수를 기록한다(5단계 참고).
- 시뮬레이션 배경이 책상/바닥/벽으로 꾸며졌다(자동 적용, 할 일 없음).
- BC 학습 타깃에서 yaw(회전 중 z축)만 제외하고 roll/pitch는 유지한다(3단계 참고).

아키텍처/설계 배경은 [`README.md`](README.md)(3.3절 BC/3.4절 RL 구현 설명)와
[`tools/README.md`](tools/README.md)(도구별 상세 옵션)를 참고. 이 문서는 "지금 뭘 실행해야
하는가"에 집중한 실행 가이드다.

## 0. 처음 설치 (새 머신에서)

이미 `pac2026` conda 환경이 있으면 이 절은 건너뛰고 바로 "0-1. 환경 활성화"로.

```bash
conda create -n pac2026 python=3.10 -y
conda activate pac2026

# lerobot 본체 + feetech(실물 서보 SDK, 아직 안 써도 미리 깔아둠) + kinematics(placo/pin, FK/IK)
pip install "lerobot[feetech,kinematics]==0.4.4"
# placo/pin 휠이 urdfdom 4 / tinyxml2 10에 링크돼 있어 최신 버전으로는 import가 깨진다 — 내려서 고정
pip install "cmeel-urdfdom>=4,<5" "cmeel-tinyxml2>=10,<11"
# seam_cv(perception/seam_cv.py) 의존성 — lerobot 기본 설치엔 없음
pip install "scipy>=1.11" "scikit-image>=0.22"
# RL 환경(MuJoCo) + 조이스틱 입력(joystick_input.py)
pip install "mujoco>=3.1" "gymnasium>=0.29" evdev
```

실제 이 머신에 검증돼 있는 조합(`pip list` 기준, 2026-10): Python 3.10.21, lerobot 0.4.4, placo
0.9.25, pin 3.8.0, cmeel-urdfdom 4.0.1, cmeel-tinyxml2 10.0.0, torch 2.10.0+cu128(CUDA 자동 설치됨,
별도 index-url 불필요 — `pip install torch`만으로 GPU 빌드가 잡힌다), mujoco 3.13.0, gymnasium 1.3.0,
scipy 1.15.3, scikit-image 0.25.2, evdev 1.9.3. GPU는 RTX 4060 Laptop(8GB) 기준.

설치 후 확인:
```bash
python -c "import lerobot, mujoco, gymnasium, evdev, cv2, skimage; print('OK')"
nvidia-smi   # GPU 인식 확인 (torch.cuda.is_available()도 True여야 함)
```

(`dearpygui`는 `record_gui.py`용인데 지금 보류 상태라 기본 설치에서 뺐다 — 다시 시도하려면
`pip install dearpygui`, tools/README.md의 record_gui.py 절 참고.)

조이스틱(Logitech Extreme 3D Pro)을 쓸 거면 USB로 연결 후 `/dev/input/eventN`이 잡히는지(`ls
/dev/input/ | grep event`) 확인 — 권한 문제로 evdev가 장치를 못 열면(Permission denied) 사용자를
`input` 그룹에 추가(`sudo usermod -aG dialout,input $USER` 후 재로그인)하거나 udev 규칙을 추가할 것.

**스로틀(z축) 영점 보정**: 처음 조이스틱을 쓰는 녹화 도구를 실행하면 "슬라이더를 중립 위치에
놓고 Enter"를 한 번 물어보고 `~/.config/pac2026/joystick_calibration.json`에 저장한다 — 그 뒤로는
슬라이더가 실제로 어디 있든 그 저장값을 기준으로 삼는다(다시 물어볼 필요 없음). 슬라이더 느낌이
이상하거나 조이스틱을 바꿨으면 `--recalibrate-joystick`(record_mujoco.py/check_ee.py/
record_gui.py 공통)로 다시 잡을 것.

## 0-1. 환경 활성화

```bash
conda activate pac2026
cd ~/pac2026   # 또는 저장소 루트
```

⚠️ 이 머신은 conda 설치본이 두 개다(`~/miniforge3`가 셸에 `conda init`돼 있고, `pac2026` 환경은
별도의 `~/anaconda3` 밑에 있음). 위 `conda activate pac2026`가 `EnvironmentNameNotFound` 또는
`CondaError: Run 'conda init' before 'conda activate'`로 실패하면 아래 중 하나로 고친다(한 번만
하면 됨):

```bash
# 방법 A (권장, 한 줄, 다시 로그인할 필요 없음) — ~/.condarc에 다른 설치본의 envs 경로를 등록
conda config --append envs_dirs /home/robot/anaconda3/envs

# 방법 B (매번 새 터미널마다 다시 해야 함, 당장 한 번만 쓸 때)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate pac2026

# 방법 C (항상 전체 경로로 직접 실행, activate 자체를 안 씀)
/home/robot/anaconda3/envs/pac2026/bin/python ai_layer/tools/check_ee.py --scene curve
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
  길이는 기본 무제한), **BTN_THUMB2**로 폐기+재시도. 도구 끝이 일정 높이(1cm, "비트 부러짐
  방지") 밑으로는 안 내려가게 막혀 있다 — 바닥에 닿아도 더 이상 에피소드가 폐기되지 않는다
  (2026-10-06 변경).
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
- 액션 중 yaw(회전 z축, drz)는 학습 타깃에서 0으로 마스킹된다 — 실로봇 IK가 5D(XYZ+roll/pitch)만
  풀어서 yaw는 애초에 반영이 안 되기 때문(roll/pitch는 실제로 쓰이는 자유도라 그대로 학습함).
  녹화 원본에는 yaw도 그대로 남아 있다.
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
  이상 떨어지면 정책이 트리거를 켜도 비드 분사가 강제로 꺼진다. **막대가 바닥/용지에 닿으면
  그 즉시 에피소드가 끝나고 큰 음의 보상이 들어간다**(`floor_contact_penalty`) — 1단계(실제
  녹화)는 접촉 시 높이 제한으로만 막고 에피소드를 안 버리도록 바뀌었지만, RL 환경은 여전히
  "접촉=즉시 종료+패널티"다. 둘이 다른 규약이라는 점에 주의할 것 — RL이 접촉 자체를 피하는
  법을 배우게 하려는 의도라 일부러 안 맞췄다.
- 결과: `outputs/rl_sac/sac_final`(및 `--ckpt-every` 간격 중간 체크포인트)와
  `outputs/rl_sac/metrics.jsonl` — `--log-every` 윈도우마다 step/critic_loss/reward_mean/
  bc_action_distance_mean(RL 행동이 BC 행동과 정규화 공간에서 얼마나 떨어져 있는지, BC
  teacher 없으면 null) 한 줄씩. 학습 끝나고 plot해서 추이 보는 용도.
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
| `conda activate pac2026`가 안 됨 | "0-1. 환경 활성화"의 방법 A/B/C 참고 (이 머신은 conda 설치본이 두 개라 이름 등록이 필요) |
| 뷰어 창이 안 뜸 | `--headless`를 실수로 안 줬는지(기본은 뜸), GPU/디스플레이 환경(`DISPLAY`). 창이 뜨기 전에 터미널에서 멈춘 것처럼 보이면 조이스틱 보정(처음 1회) 또는 데이터셋 덮어쓰기 확인 프롬프트 — 터미널을 보면 질문이 떠 있을 것(창은 그 전에 이미 떠 있어야 함) |
| `record_gui.py`가 `X Error ... BadAccess ... X_GLXMakeCurrent`로 죽음 | 알려진 간헐적 문제, 보류 중(맨 위 요약 참고) — `record_mujoco.py`(터미널)를 쓸 것 |
| 조이스틱 방향이 반대 | `joystick_input.py` 단독 실행으로 버튼/축 이름 확인 후 `--invert-x/y/z/roll/pitch` |
| z축(스로틀)이 한쪽 방향으로만 움직임 | 영점 보정이 끝 쪽에 잡혔을 가능성 — `--recalibrate-joystick`로 다시 잡을 것(0단계 참고) |
| `check_dataset.py`에서 선 인식 실패율 높음 | `tools/seam_preview.py`로 `SeamCVConfig` 재조정 |
| `check_dataset.py`에서 증분 크기 초과 비율 높음 | 녹화할 때 조이스틱을 더 천천히 움직이거나 `--max-linear-speed` 낮춰서 재녹화 |
| BC loss가 안 내려감 | 에피소드 수/다양성 부족(1단계 balanced 분포 확인), `--epochs` 늘리기, 데이터셋에
`check_dataset.py` ❌ 없는지 재확인 |
| RL reward가 이상하게 널뜀 | 먼저 `rl_env_smoke.py` 통과하는지, `--bc-checkpoint` 없이(R_imitation=0) 먼저 돌려서
R_track만으로도 학습이 되는지 분리 확인 |
| RL이 선 밖에서도 계속 비드를 뿌림 | 안전 컷오프는 "비드 분사"만 강제로 막지 "이동"은 막지 않는다 — 정책이 선을
못 따라가는 거면 R_track/BC teacher 쪽 문제(위 BC loss 항목부터 확인) |
