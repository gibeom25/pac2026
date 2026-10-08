# PyQt6 데이터 수집 + 학습 GUI (2026-10-06)

`ai_layer/tools/record_gui.py`(Dear PyGui)가 간헐적 GLX 크래시로 보류된 뒤 PyQt6로 새로
만든 GUI. 데이터 수집뿐 아니라 BC/RL 학습 실행·모니터링까지 한 화면에서 한다.

## 설치/실행

```bash
conda activate pac2026
pip install PyQt6 pyqtgraph   # 처음 한 번만
PYTHONPATH=. python ai_layer/gui/app.py
```

창이 안 뜨고 `xcb-cursor0` 관련 에러가 나면 `sudo apt install -y libxcb-cursor0` (Qt6.5+가
요구하는 시스템 라이브러리, pip로는 안 깔림).

## 데이터 수집 탭

- **손목/오버뷰 카메라**를 실시간으로 보여준다(손목 카메라 화면이 데이터셋에 실제 저장되는
  바로 그 화면). `ai_layer/tools/episode_ticker.py`의 `MujocoDualCamera`/`EpisodeTicker`를
  그대로 쓴다 — `record_mujoco.py`/`record_gui.py`와 물리/데이터셋 로직이 완전히 동일(단일
  소스, GUI 프레임워크만 다름).
- **로봇 상태**: 씬/variant, 경과 시간, EE 목표 위치, 막대 끝(tip) z, 트리거, 접촉 여부,
  비드 점수, 진행도(저장된/목표 에피소드 수).
- **형태별 수집 현황**: pyqtgraph 막대그래프로 실시간 갱신(`BalancedSceneSampler`의 카운트,
  `meta/scene_balance.json`과 같은 소스).
- **모든 설정이 화면에 있다** — repo-id/root/scene/variant/num-episodes/episode-seconds/fps/
  속도/반전/입력장치/재보정까지 CLI 플래그 없이 조정. 세션 시작 전에만 수정 가능(세션 중엔
  잠김).
- **입력**: `input` 드롭다운이 auto(기본)/joystick/keyboard. keyboard는 터미널이 아니라
  **이 창에 포커스가 있을 때 Qt가 직접 주는 진짜 press/release 키 이벤트**를 쓴다
  (`qt_keyboard_input.py`) — `ai_layer/tools/keyboard_input.py`(터미널 raw 모드, CLI 도구
  전용)와는 별개 구현. 키 배치는 CLI 쪽과 동일: WASD(xy) R/F(z) Q/E(roll) Z/X(pitch) C/V(yaw)
  SPACE(트리거) ENTER(저장+종료) BACKSPACE(폐기+재시도).
- **조작 테스트**(2026-10-06): "에피소드 시작"을 누르기 전에 설정(속도/반전 등)을 바꿔가며
  실제로 움직여볼 수 있는 모드 — 데이터셋/에피소드 카운트 없이 카메라+로봇 상태만 보면서
  자유롭게 조종한다(check_ee.py와 같은 성격). 테스트 중엔 속도/반전 설정이 **실시간으로
  반영**된다(다른 설정은 다음 테스트 씬이 열릴 때 반영). 만족스러우면 **"테스트 통과 → 바로
  녹화 시작"**을 누르면 입력 장치를 다시 연결하지 않고 그대로 들고 실제 녹화 세션으로 넘어간다.
- **카메라 화면 크기**: 실제 녹화/추론 해상도(320x240)는 그대로 두고, 화면에 보여주는 크기만
  1.5배(480x360)로 키웠다(`collect_tab.py`의 `_DISPLAY_SCALE`).
- **기존 데이터셋 발견 시**(repo-id/root가 이미 있고 에피소드가 있을 때) 터미널 대신
  QMessageBox로 덮어쓰기/이어서 기록/취소를 묻는다(`dataset_prep.py`).
- 조이스틱을 **처음** 쓰는 거라 저장된 스로틀 보정값이 없으면, 세션 시작 시 터미널에 보정
  프롬프트가 뜬다(GUI 다이얼로그 아님 — 처음 한 번만 겪는 일이라 V1에서는 그대로 둠). 한 번
  보정하면 `~/.config/pac2026/joystick_calibration.json`에 저장되고, 다음부턴 안 뜬다.

## 학습 탭

BC/RL 하위 탭이 있고 각각 `ai_layer/train_bc.py`/`ai_layer/train_rl.py`를 subprocess로
실행한다 — 학습 로직 자체는 안 건드림, GUI는 그냥 그 두 스크립트를 실행하고 지켜볼 뿐이다.

- 설정(repo-id, epochs, batch-size, out-dir, xyz-only 등 / num-steps, bc-checkpoint 등)을
  화면에서 입력하고 **학습 시작**을 누르면 그 설정으로 CLI와 똑같은 커맨드가 실행된다.
- **로그**: subprocess의 stdout/stderr를 그대로 실시간으로 보여준다.
- **그래프**: 콘솔 출력을 긁는 게 아니라 두 스크립트가 `<out-dir>/metrics.jsonl`에 주기적으로
  쌓는 줄(step당 loss / reward_mean·critic_loss·bc_action_distance_mean)을 1초마다 tail해서
  pyqtgraph로 그린다 — train_bc.py의 metrics.jsonl 로깅은 이 GUI를 위해 2026-10-06에 추가함
  (train_rl.py는 원래부터 있었음).
- **중지**: SIGTERM(`terminate()`)을 보내고 3초 기다려도 안 죽으면 강제 종료(`kill()`).

## 설계 메모

- `ai_layer/tools/episode_ticker.py`(물리 스텝/카메라/비드/데이터셋 기록)와
  `ai_layer/tools/record_mujoco.py`(BalancedSceneSampler, 씬 경로, 상수)를 그대로 재사용 —
  GUI 쪽에서 새로 만든 건 Qt 위젯/이벤트 배선뿐이다. 이 ticker를 고치면 `record_gui.py`
  (Dear PyGui)와 이 PyQt GUI 둘 다 같이 영향받는다.
- 시뮬레이션 틱은 QTimer가 GUI 메인 스레드에서 그대로 돌린다(record_gui.py의 while 루프와
  동일한 "프레임마다 한 스텝" 패턴) — 별도 워커 스레드 안 씀, mujoco 컨텍스트/Qt 위젯 갱신을
  스레드 간에 넘길 필요가 없어서 단순하다. 체감 렉이 느껴지면 그때 워커 스레드로 옮기는 걸
  고려할 것(아직은 불필요).
- 창을 닫으면(`MainWindow.closeEvent`) 켜져 있던 세션(컨트롤러 종료, 데이터셋 finalize)과
  돌고 있던 학습 subprocess(terminate)를 자동으로 정리한다.

## 시뮬레이션 / 실로봇 전환 (2026-10-06)

데이터 수집 탭 맨 위 **소스** 드롭다운으로 시뮬레이션/실로봇을 고른다. 조이스틱/키보드
컨트롤러는 그대로 재사용되고(하드웨어 중립 인터페이스), 백엔드만 `EpisodeTicker`(MuJoCo) ↔
`RealRobotEpisodeTicker`(lerobot SOFollower + IK)로 바뀐다 — 데이터셋 스키마가 동일해서
BC/RL 학습 코드는 안 바뀐다. 자세한 건 `ai_layer/real/README.md`(안전 주의사항 포함 —
**실제 하드웨어로 검증 못 한 코드**).

## BC fine-tuning / RL→BC 증류 (2026-10-08)

학습 탭에 두 가지가 추가됐다 — "시뮬 가중치에 실로봇/RL 데이터를 fine-tuning하고 싶다"는
요청에 대한 답:

- **BC 탭의 `init-checkpoint` 필드**: 비우면 기존처럼 처음부터 학습. 채우면(예:
  `outputs/bc_act/last`) 그 체크포인트의 가중치+정규화 통계를 그대로 불러와 이어서
  학습(fine-tuning) — `train_bc.py --init-checkpoint`. 실로봇 데이터로 이어서 학습할 때 씀.
- **"RL→BC 증류 (롤아웃)" 탭**: RL(SAC)과 BC(ACT)는 서로 다른 신경망이라 SAC 가중치를 ACT로
  직접 못 옮긴다 — 그래서 학습된 SAC 정책을 시뮬에서 굴려(rollout) BC 데이터셋 포맷으로
  기록하는 중간 단계(`tools/rl_rollout_to_dataset.py`)를 거친다. 이 탭에서 롤아웃 데이터셋을
  만든 뒤, 그 repo-id를 BC 탭의 repo-id + init-checkpoint(기존 BC 체크포인트)에 넣고 적은
  epoch로 돌리면 "RL이 다듬은 동작"이 BC 가중치에 증류된다. 반복 학습이 아니라 한 번 쭉 도는
  배치 작업이라 그래프는 없고 로그만 나온다.

**주의**: RL 학습 환경(`so101_seam_env.py`)은 항상 같은 텍스처(scene_a4.xml, "curve" 모양)만
렌더링하고 ground-truth 경로는 매 에피소드 무작위로 다른 형태를 고른다(RL 정책이 이미지를
관측으로 안 써서 학습 자체엔 문제없었음) — 그래서 롤아웃 스크립트는 실제 뽑힌 (형태, variant)에
맞는 텍스처 MJCF를 에피소드마다 새로 로드해서 렌더링한다(안 그러면 화면과 행동이 안 맞는
엉터리 데이터가 기록됨 — 실제로 찾아서 고친 문제).
