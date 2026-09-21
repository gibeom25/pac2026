"""SO-101 용접 궤적 RL용 SACConfig 프리셋.

설계 문서: docs/AI_추론계층_프레임워크.md 3.4절 확정 사항.
lerobot의 SACPolicy/SACConfig(HIL-SERL 구현)를 그대로 사용 — 새로 구현하지 않음.

액션 공간: 연속 6dim(EEF-delta 정규화값 [-1,1]: dx,dy,dz,drx,dry,drz) + 이산 1개(그리퍼/펌프 0|1),
lerobot SACConfig의 `num_discrete_actions` 네이티브 지원을 사용.

BC(configs/so101_act_bc.py)와 맞춘 것:
  - 이미지 키 = observation.images.wrist (BC와 동일. 예전 "observation.image.wrist"는 BC와 달라 KeyError)
  - observation.state = 9 (xyz + rot6d). BC와 같은 표현이어야 BC teacher가 RL 관측을 읽을 수 있다.
  - seam 특징 5.

정규화 (LeRobot 0.4.x): SACConfig.dataset_stats 기본값은 다른 프로젝트용(키·차원 불일치)이라
여기서 우리 키에 맞는 통계를 직접 준다 (SAC_DATASET_STATS). 실측 데이터가 생기면 갱신할 것.
"""

from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ai_layer.configs.so101_act_bc import IMAGE_KEY, SEAM_FEATURE_DIM, STATE_DIM  # BC와 동일 키/차원

# docs 3.4 권장 초기값
DISCOUNT = 0.97
CRITIC_TARGET_UPDATE_WEIGHT = 0.005  # tau, docs 권장값과 동일

CONTINUOUS_ACTION_DIM = 6  # EEF-delta 6dim (그리퍼는 discrete로 분리)
NUM_DISCRETE_ACTIONS = 2  # 그리퍼/펌프 0|1

__all__ = [
    "IMAGE_KEY", "SEAM_FEATURE_DIM", "STATE_DIM", "CONTINUOUS_ACTION_DIM", "NUM_DISCRETE_ACTIONS",
    "build_so101_sac_config", "build_sac_dataset_stats",
]


def build_sac_dataset_stats() -> dict[str, dict[str, torch.Tensor]]:
    """우리 관측/액션 키에 맞는 정규화 통계 (초기 추정치. 실측 후 갱신).

    - 이미지: ImageNet 평균/표준편차 (3,1,1)
    - state(9): 위치는 절차적 경로 작업공간(so101_seam_env 기준) 중심/폭, rot6d는 [-1,1] 성분
    - env_state(5): [lookahead 상대 xyz(3), 곡률, 굵기]
    - action(6): [-1,1] → MIN_MAX 정규화가 항등이 되도록 min=-1, max=1
    """
    f = lambda *xs: torch.tensor(xs, dtype=torch.float32)  # noqa: E731
    return {
        IMAGE_KEY: {
            "mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1),
        },
        OBS_STATE: {
            "mean": f(0.25, 0.0, 0.10, 0, 0, 0, 0, 0, 0),
            "std": f(0.10, 0.10, 0.10, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6),
        },
        OBS_ENV_STATE: {
            "mean": f(0.0, 0.0, 0.0, 0.0, 1.0),
            "std": f(0.05, 0.05, 0.05, 1.0, 0.5),
        },
        ACTION: {
            "min": -torch.ones(CONTINUOUS_ACTION_DIM),
            "max": torch.ones(CONTINUOUS_ACTION_DIM),
        },
    }


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
        dataset_stats=None,  # 기본값(타 프로젝트용) 사용 안 함. build_sac_dataset_stats()를 processor에 주입.
        discount=DISCOUNT,
        critic_target_update_weight=CRITIC_TARGET_UPDATE_WEIGHT,
        num_discrete_actions=NUM_DISCRETE_ACTIONS,
        num_critics=2,
        shared_encoder=True,
        device="cuda",
        storage_device="cuda",
    )
