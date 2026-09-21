"""SO-101 용접 궤적 BC용 ACTConfig 프리셋.

설계 문서: docs/AI_추론계층_프레임워크.md 3.3절 확정 사항.
  - chunk_size = 32 (dt_AI=33.3ms 기준 약 1067ms). dt는 송지수 제어 ActionChunk 기본 dt(30 Hz)에 맞춤.
    docs 4.4절의 20ms는 구 값 (2026-09-21 도윤 결정으로 33.3ms).
  - 자유도: 민제씨 PAC_Supermoon URDF(5관절+그리퍼)에 맞춤. IK가 5D(XYZ+roll/pitch)라
    yaw(월드 z 회전 증분)는 학습 타깃에서 0으로 고정 (ZERO_YAW).
  - use_vae = False (1차 버전은 CVAE 없이 결정적 chunk 예측)
  - 입력: 이미지 1대(wrist) + seam CV 특징(observation.environment_state) + proprioception(observation.state)
  - 출력: EEF-delta 7차원 (dx, dy, dz, drx, dry, drz, gripper). 그리퍼는 유지.

이 프리셋은 lerobot의 ACTPolicy/ACTConfig를 그대로 사용하고, 이 프로젝트의
관측/액션 스펙만 주입한다 (Transformer 자체를 새로 구현하지 않음).

정규화 (LeRobot 0.4.x): 정규화는 정책 안이 아니라 `make_act_pre_post_processors(cfg, stats)`가
만드는 전/후처리 파이프라인에서 수행된다. train_bc.py가 SO101BCDataset.compute_stats()로
통계를 만들어 넘긴다. normalization_mapping은 그 파이프라인이 읽는다.
"""

from __future__ import annotations

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# 스텝 간격: 송지수 제어 ActionChunk 기본 dt = 1/30 s (≈33.3 ms). docs 4.4절 20ms는 구 값.
DT_AI_SEC = 1.0 / 30.0
CHUNK_SIZE = 32  # 32 × 33.3 ms ≈ 1.07 s (ActionChunk 최대 64 스텝 이내)
# 민제씨 실로봇 IK는 5D(XYZ + roll/pitch). yaw 증분(drz)은 학습 타깃에서 0으로 고정.
ZERO_YAW = True

# seam_cv.SeamPoint에서 뽑는 로컬 특징 차원: (lookahead 상대위치 2, curvature 1, thickness 1, 점 밀도 1)
SEAM_FEATURE_DIM = 5
# proprioception: EEF pose 9 = 위치 3 (x,y,z) + 회전 6D (회전행렬 앞 두 열).
# 회전벡터(3)는 180° 근처에서 값이 불연속으로 튀어 관측 입력으로 부적합 → 연속적인 6D 표현 사용.
# (kinematics.pose_to_state 참고. 델타 계산은 별도로 회전행렬 기반이라 영향 없음.)
STATE_DIM = 9
# action: EEF-delta 6 + gripper 1. gripper는 LeRobot 녹화값(0~100, RANGE_0_100) 그대로.
ACTION_DIM = 7

# 이미지 키. lerobot-record가 만드는 키는 "observation.images.<카메라>" (images, 복수형).
# RL(configs/so101_sac.py)도 같은 키를 쓴다.
IMAGE_KEY = f"{OBS_IMAGES}.wrist"


def build_so101_act_config(camera_height: int = 240, camera_width: int = 320) -> ACTConfig:
    input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, camera_height, camera_width)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(SEAM_FEATURE_DIM,)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
    }

    return ACTConfig(
        n_obs_steps=1,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        vision_backbone="resnet18",
        use_vae=False,  # docs 3.3: 1차 버전은 CVAE 제외
    )
