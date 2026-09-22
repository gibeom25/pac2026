# AI 추론 계층 프레임워크 설계서

- 상위 과제: PAC2026 HD현대로보틱스 미션4 — AI 추론·실시간 제어 통합형 RTOS 기반 로봇 상위제어 플랫폼
- 이 문서 범위: 3계층(AI 추론 / 제어 / 통신·HAL) 중 **AI 추론 계층** 전체 설계
- 표기 규칙: ✅ 확정 사항 / 🔶 권장안(사용자 확인 필요) / ❓ 미해결 항목
- 최종 갱신: 2026-09-19

---

## 0. 설계 원칙

1. **모델 교체 가능성**: task가 바뀌어도(용접 → 다른 작업) 하위 제어·HAL 계층이 그대로 동작해야 하므로, AI 추론 계층이 밖으로 내보내는 인터페이스는 모델 구현과 무관하게 고정.
2. **EEF-delta 통일**: 특정 로봇(SOArm 등)에 종속되지 않기 위해 모든 액션은 EEF 기준 delta로 표현.
3. **비주기 추론 / 주기 제어의 분리**: AI 추론은 자체 페이스(비동기, 가변시간)로 동작하고 결과를 시퀀스 단위로 제어 계층에 전달, 제어 계층이 이를 1kHz로 소비.
4. **지연은 예측이 아니라 사후 재동기화로 흡수**: 정밀한 t_infer 사전 예측보다, 결과가 나온 시점의 실측 로봇 상태에 리싱크하는 방식을 채택 (2절 참고).
5. **채점 우선순위 반영**: 발주처 원문 — "AI 추론 정확도 자체보다 추론 결과가 제어에 반영되는 과정의 실시간성·안정성·확장성을 중점 평가". BC/RL 정확도보다 아래 5장(지연보정)·6장(모델교체 검증)에 구현 우선순위를 둘 것.

---

## 1. 전체 파이프라인

### 1.1 학습 시 (BC-teacher 구조)

```
SOArm-leader 티칭 (EEF-delta 기록)
        ↓
IsaacLab 시각화/궤적 데이터셋 구성
        ↓
BC (Transformer, 🔶미확정) ── teacher ──▶ RL (Conv+MLP, 이미지+BC 궤적 참고)
```

### 1.2 배포 시 (RL 단일망 고정) — ✅확정

```
카메라(Wrist RGBD, Agent RGB)
        ↓
RL 네트워크 (frozen, Conv+MLP 단일망)   ← BC는 배포 파이프라인에서 제외
        ↓
action trajectory (EEF-delta 시퀀스)
        ↓
지연보정 (state re-anchoring, 4장 참고)
        ↓
제어 계층으로 전달
```

**결정 배경**: 학습 때는 BC→RL 2단 구조(BC teacher)를 쓰지만, 배포 시 두 네트워크를 순차 실행하면 t_infer 분산이 커져 "AI 추론시간 비일정성 대응"이라는 과제 핵심 목표와 상충한다. 따라서 배포용 추론 경로는 RL 단일망만 실행하도록 확정.

---

## 2. 인터페이스 스펙 (I/O 계약) — ✅확정

이 계약은 내부 모델이 무엇으로 바뀌든 불변이어야 함 (모델 교체 가능성 원칙).

### 입력
| 항목 | 출처 | 비고 |
|---|---|---|
| Wrist RGBD, Agent RGB | 카메라 | 학습/추론 공통 입력 |
| EEF 현재 상태 | 통신·HAL 계층 (FK + base→world 역변환) | 지연보정에 필수 |

### 출력 (→ 제어 계층)
| 항목 | 형식 |
|---|---|
| Action trajectory | (dx, dy, dz, droll, dpitch, dyaw, gripper_signal[0/1]) 시퀀스 |
| 메타데이터 | 생성 시각 timestamp, 시퀀스 내 진행 인덱스, (선택) 신뢰도 |

---

## 3. 서브모듈 구성

### 3.1 Vision/Seam-Groove 인식 — ✅확정: 고전 CV 모듈
BC/RL과 분리된 경량 고전 CV 모듈로 구성한다. 펜/프린트 선은 고전 CV로 안정적으로 검출 가능한 대상이므로, 이를 별도 모듈로 빼서 BC/RL은 "경로를 어떻게 따라갈지(속도·부드러움)"만 학습하면 되도록 데이터 효율을 높인다.

**파이프라인 (권장 세부안)**:
1. 색상/대비 기반 이진화 (HSV threshold 또는 adaptive threshold) → 형태학적 노이즈 제거 (opening/closing)
2. Skeletonization으로 1px 중심선 추출 (`skimage.morphology.skeletonize` 또는 `cv2.ximgproc.thinning`)
3. 중심선을 순서 있는 polyline으로 정렬 (끝점에서 시작해 인접 픽셀을 따라가는 그래프 순회)
4. 끊김(gap) 처리: 스켈레톤 끝점 간 거리가 임계값 이내면 직선으로 연결 (원본 요구사항 "끊김 빈도" 학습 대상과는 별개로, 인식 단계의 끊김은 여기서 보강)
5. Depth(RGBD)로 픽셀 → 카메라 좌표 → world/EEF 좌표 변환
6. 로컬 특징 추출: 각 경로 지점에서 곡률(주변 점들로 추정), 선 굵기(거리변환 기반) 산출

**출력** (BC/RL 입력으로 사용): 정제된 경로 좌표 polyline(world/EEF 기준) + 지점별 (곡률, 선 굵기) 특징. 이는 팀 프레임워크 다이어그램의 "Seam/Groove 인식 → EEF 기준 현재/목표 위치 상대오차 계산" 단계에 정확히 대응.

### 3.2 학습 데이터 파이프라인 — ✅확정
- SOArm-leader 티칭 → EEF-delta 궤적 기록 (로봇 종속성 배제).
- IsaacLab에서 시각화 및 궤적 데이터셋 구성.
- 🔶 권장: 아래 4개 축을 명시적 체크리스트로 두고 수집 커버리지 관리
  - 궤적 길이 / 경로 형태(직선·곡선·지그재그) / 끊김 빈도 / 선 굵기별 속도 매핑

### 3.3 BC (Behavior Cloning) — ✅확정: ACT 계열 Transformer, action chunking
- **입력**: 이미지 임베딩(경량 CNN backbone, 예: ResNet18) + 3.1 CV 모듈의 경로 특징(로컬 목표점, 곡률, 선 굵기) + 현재 proprioception(EEF 상태) + 최근 액션 히스토리(작업 맥락 반영).
- **구조**: ACT(Action Chunking Transformer) 계열.
  - Transformer 인코더: 위 입력 시퀀스(짧은 history window)를 인코딩.
  - Transformer 디코더: 학습된 K개의 위치 쿼리(queries)로 향후 K-step action chunk를 한 번에 디코딩 (DETR 스타일).
  - **Chunk 길이 K**: AI 출력 스텝 간격 `dt_AI = 20ms` 기준 **K=32** (= 640ms 분량) 권장 — 근거는 4.4절 지연보정 파라미터와 연동.
  - CVAE 잠재변수(원조 ACT의 스타일 다양성 모델링)는 **1차 버전에서는 제외**(단순 결정적 chunk 예측). 데모 스타일 편차(선 굵기별 속도 등)가 학습을 흐리게 하면 2차 버전에서 추가 검토.
- **역할**: 학습 단계 teacher 전용. 배포 파이프라인에는 포함되지 않음 (1.2절) — RL 학습 중 보상 계산에만 사용되고, 배포된 RL 단일망은 BC를 런타임에 호출하지 않음.

### 3.4 RL (강화학습) — ✅확정: SAC, Residual-style reward shaping
- **알고리즘**: SAC (Soft Actor-Critic). 연속 액션(6DOF EEF-delta)에 적합, off-policy라 시뮬레이션 샘플 재사용 효율이 좋음.
  - 정책망: Conv(이미지) + MLP(3.1 경로 특징 + proprioception + BC chunk 참조값) → 6DOF 델타의 평균/표준편차(tanh-squashed Gaussian).
  - 그리퍼/펌프 신호(0/1)는 SAC의 연속 액션에 로짓으로 포함 후 추론 시 임계값으로 이진화 (별도 정책 헤드 분리는 1차 버전에서 불필요 — 데이터로 충분히 학습 가능한 단순 신호).
  - 비평망: Twin Q-network (Conv+MLP), target network soft update (τ≈0.005), 자동 엔트로피 온도 튜닝.
  - 감가율 γ ≈ 0.95~0.98 권장 (용접 한 구간 단위의 비교적 짧은 호라이즌 태스크이므로 표준 0.99보다 약간 낮게).
- **보상 함수 (세부 설계)**:

  `R_t = w1·R_imitation + w2·R_track + w3·R_smooth`

  | 항 | 정의 | 의도 |
  |---|---|---|
  | `R_imitation` | `-‖a_RL − a_BC‖²` (같은 시점 BC 예측 델타와의 L2 거리) | "따라가는거" — BC 경로를 teacher로 추종 |
  | `R_track` | `-\|d_perp\|` (3.1 CV 경로 중심선까지 수직거리) `+ progress_t − λ·\|Δprogress_t − Δprogress_{t-1}\|` | "선을 일정하게 잘 따라가는지" — 중심선 유지 + 진행속도의 일관성(급가속/급감속 페널티) |
  | `R_smooth` | `-‖a_t − a_{t-1}‖²` | 실제 로봇 동작 부드러움 (펌프 도포 품질과 직결) |

  - **속도-곡률 적응**: `R_track`의 progress 목표값은 상수가 아니라 3.1에서 얻은 로컬 곡률·선 굵기의 함수 `target_speed = f(curvature, thickness)`로 설정 (곡률 클수록/선 얇을수록 감속) — 원 요구사항 "선 굵기에 따른 속도 조절" 반영.
  - **가중치 스케줄링**: 학습 초반 `w1`(모방)을 높게(예: 1.0), `w2·w3`(과제 최적화)를 낮게 시작 → 학습 진행에 따라 `w1`을 floor(예: 0.3)까지 선형 감소, `w2·w3`는 상대적으로 증가. BC를 그대로 복제하고 끝나거나 반대로 위험하게 이탈하는 두 극단을 모두 방지.
  - **초기 가중치 권장값** (실측 후 튜닝 필요): `w1=1.0→0.3`, `w2=0.3→1.0`, `w3=0.2`(고정), `λ=0.5`.
- **배포**: ✅ 학습 완료 후 RL 네트워크만 고정(frozen)하여 단일망으로 추론 (1.2절). BC는 학습 중 보상 계산(`R_imitation`)에만 관여하며, 배포된 RL 네트워크의 입력에는 BC 출력이 필요 없음 — RL이 학습 과정에서 BC의 행동을 이미 내재화했기 때문.

---

## 4. 지연보정 모듈 (Latency Compensation) — 핵심 설계

### 4.1 위치 — ✅확정
파이프라인 최종 출력 직후, 제어 계층 전달 **직전 단일 지점**에서만 수행한다. (팀 프레임워크 다이어그램상 "지연 보정" 블록의 위치는 컨셉 참고용일 뿐이며, 이 지점이 유일한 실제 구현 위치임 — 다이어그램과 실제 구현 위치가 다르다는 점에 유의.)

### 4.2 핵심 방식 — ✅확정: State Re-anchoring
추론이 끝난 시점에 실측 EEF 현재 상태(통신·HAL 계층에서 피드백)를 받아, 새로 생성된 action trajectory 중 **가장 유사한 지점을 찾아 그 지점부터** 시퀀스를 제어 계층에 전달 시작한다. 정밀한 t_infer 사전 예측 대신, 결과가 나온 시점의 실측 상태에 사후적으로 리싱크하는 방식이므로 추론시간 자체의 불규칙성에 강건하다.

### 4.3 강건화를 위한 권장 알고리즘 — 🔶확인 필요
순수 "위치 최근접점 탐색"만으로는 두 가지 위험이 있다: (a) 경로가 자기교차/지그재그일 때 위상(phase) 모호성으로 엉뚱한 지점에 매칭될 수 있음, (b) 위치는 맞아도 접합부에서 속도·가속도가 불연속해 저크(jerk) 발생 가능. 이를 보완하는 하이브리드 방식을 권장:

1. **Coarse anchor**: 실측 t_infer로 1차 후보 인덱스 `j0 = last_progress_idx + round(measured_t_infer / dt)` 계산.
2. **좁은 윈도우 탐색**: `j0` 주변 `[j0-W, j0+W]` 범위 내에서만 탐색 (전역 탐색 금지 — 연산비용 및 자기교차 모호성 방지).
3. **위상 매칭**: 매칭 기준에 위치뿐 아니라 최근 진행방향/속도 벡터도 포함.
4. **단조증가 제약**: 직전 진행 인덱스보다 반드시 앞선 지점만 후보로 허용 (역행 금지).
5. **접합부 블렌딩**: 매칭점을 찾은 후 현재 명령값 → 매칭점까지 짧은 구간(수십 ms)을 minimum-jerk(또는 선형) 보간으로 연결.

```text
def compensate_latency(new_trajectory, current_eef_state, last_progress_idx,
                        measured_t_infer, dt, W, blend_ms):
    j0 = last_progress_idx + round(measured_t_infer / dt)
    candidates = [j for j in range(j0 - W, j0 + W) if j > last_progress_idx]
    best_j = argmin_{j in candidates} state_distance(new_trajectory[j], current_eef_state)
    spliced = blend(current_eef_state, new_trajectory[best_j:], blend_ms)
    return spliced, best_j
```

### 4.4 파라미터 권장값 — 🔶초기 추천(실측 후 튜닝)

실측 데이터가 없는 상태의 초기값이며, t_infer 분포 측정 후 재조정할 것.

| 파라미터 | 권장 초기값 | 근거 |
|---|---|---|
| AI 시퀀스 스텝 간격 `dt_AI` | 20ms (50Hz) | 1kHz 제어주기 대비 50배 업샘플링 여유 확보하면서 AI 연산 부담은 과도하지 않은 절충점. 3.3의 BC chunk 길이 K와 연동. |
| Chunk 길이 `K` (BC/RL 출력 길이) | 32 스텝 (≈640ms) | 예상 t_infer(작은 Conv+MLP 기준 수십~200ms대 추정)의 2~3배 여유. 큐가 바닥나는 것을 방지. |
| 재동기화 윈도우 `W` | ±5 스텝 (±100ms) | coarse anchor(`j0`, 실측 t_infer 기반) 오차가 이 범위를 크게 벗어나지 않는다고 가정. 좁게 유지해 자기교차 구간 모호성과 탐색 비용을 최소화. |
| 블렌딩 시간 `blend_ms` | 50~100ms | 1kHz 기준 50~100틱. 전형적 용접 이동속도에서 수 mm급 위치 잔차를 흡수하기에 충분하면서, 데모 상 체감 지연은 최소화. |
| `state_distance` 가중치 | `w_pos=1.0`(m), `w_rot=0.1`(rad), `w_vel=0.5`(방향 코사인 유사도 기반) | 위치 정합을 우선하되 자기교차 구간에서는 방향 벡터로 tie-break. 순수 초기 추정치. |

이 표의 모든 값은 **1차 구현의 출발점**이며, 실제 배포 RL 네트워크의 t_infer 실측 분포와 초기 시연 결과를 확보한 뒤 재조정한다.

---

## 5. 모델 교체 시나리오 검증 체크리스트 — 🔶

새로운 task로 Vision-Action 계층을 교체할 때 다음을 반드시 확인:
- [ ] 새 모델이 2장의 출력 인터페이스(EEF-delta 시퀀스 + timestamp/진행 인덱스)를 그대로 준수하는가?
- [ ] 4장 지연보정 모듈이 필요로 하는 메타데이터(시퀀스별 시간/진행 인덱스)를 새 모델도 제공하는가?
- [ ] 새 모델의 t_infer 통계(평균/표준편차)를 사전 측정해 4.3절의 `W`, `blend_ms`를 재조정했는가?

---

## 6. 성능 측정 계획

상위 과제 목표([[project_pac2026_hyundai_robotics]]: 1kHz 제어주기, ±100μs 지터)와 연동하여 AI 추론 계층 자체에서 측정할 지표:
- t_infer 분포(평균/표준편차) — 배포 시 RL 단일망 기준.
- End-to-End 지연: 카메라 입력 → 제어 계층 반영까지.
- 지연보정 적용 전/후 궤적 연속성 비교 (접합부 가속도 불연속 크기).
- AI 연산 부하 유무에 따른 제어주기/지터 변화는 제어 계층 벤치마크와 연동해 비교.

---

## 7. 설계 확정 현황 (2026-09-19 갱신)

1. ✅ Seam/Groove 인식 → 고전 CV 모듈로 분리 확정 (3.1)
2. ✅ BC 아키텍처 → ACT 계열 Transformer, action chunking(K=32) 채택 확정 (3.3)
3. ✅ RL 보상함수 → `R_imitation + R_track + R_smooth` 3항 구조 + 가중치 스케줄링 확정 (3.4)
4. ✅ RL 알고리즘 → SAC 확정 (3.4)
5. 🔶 지연보정 파라미터(`dt_AI`, `K`, `W`, `blend_ms`, `state_distance` 가중치) → 초기 추천값 확정, 실측 후 튜닝 예정 (4.4)

**남은 실질적 미해결 항목**:
- `R_track`의 `target_speed = f(curvature, thickness)` 구체 함수형 — 데모 데이터 통계 확보 후 회귀/룩업테이블로 결정.
- 4.4절 파라미터들의 실측 기반 재조정 — RL 배포망 t_infer 분포 측정 후.
- CV 모듈의 gap-bridging 최대 거리 임계값 — 실제 용접선 샘플로 튜닝.
- RL 환경(`ai_layer/envs/so101_seam_env.py`)의 목표 경로는 현재 절차적 직선(ground truth)이며,
  실제 카메라+seam_cv 인식을 RL 루프에 넣는 것은 후속 작업.

## 8. 구현 현황 (2026-09-22 갱신)

`ai_layer/` 폴더에 BC(ACT)·RL(SAC) 프레임워크 코드를 구현했다 (상세는 `ai_layer/README.md`).
BC/RL 정책 자체는 lerobot(0.4.4)의 기존 `ACTPolicy`/`SACPolicy` 구현을 그대로 사용 — 이 프로젝트가
새로 짠 것은 관측/액션 스펙, 데이터 변환(FK 기반 EEF-delta), 보상함수, 환경(IsaacLab)뿐이다.

- SO-101 URDF(`TheRobotStudio/SO-ARM100`)와 IsaacLab USD 자산(NVIDIA 공식
  `Sim-to-Real-SO-101-Workshop`에서 재사용)을 `assets/`에 확보.
- Isaac Sim 4.5.0.0 + IsaacLab v2.1.0을 `pac2026_isaaclab` conda 환경에 설치, SO-101 USD 로드/시뮬레이션
  스텝까지 사용자 터미널에서 검증 완료 (`ai_layer/sim/smoke_test_so101.py`).
- BC/RL 코드는 이 세션에서 더미 배치로 단위 검증 완료 (forward/inference, SAC의 4개 loss + target
  network 업데이트 + `select_action`까지). **`ai_layer/envs/so101_seam_env.py` + `train_rl.py`도 사용자
  터미널에서 실제 end-to-end 실행 성공** (SO-101 64개 병렬 시뮬레이션, DifferentialIK 연동, SAC 학습
  루프, 체크포인트 저장까지 확인, 2026-09-21) — AI 추론 계층의 시뮬레이션 학습 파이프라인 자체는
  하드웨어와 무관하게 전부 동작 확인된 상태.
- ⚠️ **세션 제약**: Isaac Sim/IsaacLab 실행(헤드리스 시뮬레이션 포함)은 이 코딩 세션(Bash 도구 샌드박스)
  안에서는 CUDA P2P 검증 단계에서 멈춘다 (하드웨어/드라이버 문제 아님, 실제 터미널에서는 정상).
- ⚠️ **SO-101 leader 하드웨어 미해결**: 서보 버스가 모든 baud rate/ID/raw 시리얼 레벨에서 무응답이고
  실시간 USB disconnect 이벤트도 관측됨 — 소프트웨어(conda/lerobot 버전)가 원인이 아님을 여러 각도로
  확인함(권한, pyserial 버전, conda 미사용 시스템 파이썬, Python 3.12 공식 설치 절차 전부 동일 실패).
  사용자가 하드웨어를 별도로 재설치/점검 중 — 실물 데이터 수집(1단계)은 이게 해결돼야 재개 가능.
  따라서 GPU 시뮬레이션 실행은 항상 사용자 터미널에서 진행한다.

이 항목들은 실측 데이터(t_infer 분산, 초기 BC/RL 학습 결과)가 나오는 대로 확정하고 이 문서를 갱신할 것.
