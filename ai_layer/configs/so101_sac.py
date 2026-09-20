"""SO-101 용접 궤적 RL용 SACConfig 프리셋.

설계 문서: docs/AI_추론계층_프레임워크.md 3.4절 확정 사항.
lerobot의 SACPolicy/SACConfig(HIL-SERL 구현)를 그대로 사용 — 새로 구현하지 않음.

액션 공간: 연속 6dim(EEF-delta: dx,dy,dz,drx,dry,drz) + 이산 1개(그리퍼/펌프 0|1),
lerobot SACConfig의 `num_discrete_actions` 네이티브 지원을 사용 (그리퍼를 별도 로짓으로
분리 — 3.3/3.4절 초안에서 "연속 로짓 포함"으로 적었던 것보다 더 정확한 표현).
"""

from __future__ import annotations

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE

# docs 3.4 권장 초기값
DISCOUNT = 0.97
CRITIC_TARGET_UPDATE_WEIGHT = 0.005  # tau, docs 권장값과 동일

SEAM_FEATURE_DIM = 5  # ai_layer/perception/seam_cv.py 특징 차원과 동일
STATE_DIM = 6  # EEF pose 6dim
CONTINUOUS_ACTION_DIM = 6  # EEF-delta 6dim (그리퍼는 discrete로 분리)
NUM_DISCRETE_ACTIONS = 2  # 그리퍼/펌프 0|1

IMAGE_KEY = f"{OBS_IMAGE}.wrist"


def build_so101_sac_config(camera_height: int = 240, camera_width: int = 320) -> SACConfig:
    input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, camera_height, camera_width)),
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(SEAM_FEATURE_DIM,)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(CONTINUOUS_ACTION_DIM,)),
    }

    return SACConfig(
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        discount=DISCOUNT,
        critic_target_update_weight=CRITIC_TARGET_UPDATE_WEIGHT,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        num_critics=2,
        shared_encoder=True,
        device="cuda",
        storage_device="cuda",
    )
