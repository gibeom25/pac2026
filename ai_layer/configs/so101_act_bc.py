"""SO-101 용접 궤적 BC용 ACTConfig 프리셋.

설계 문서: docs/AI_추론계층_프레임워크.md 3.3절 확정 사항.
  - chunk_size = 32 (dt_AI=20ms 기준 640ms, docs 4.4절과 연동)
  - use_vae = False (1차 버전은 CVAE 없이 결정적 chunk 예측)
  - 입력: 이미지 1대(wrist) + seam CV 특징(observation.environment_state) + proprioception(observation.state)
  - 출력: EEF-delta 7차원 (dx, dy, dz, drx, dry, drz, gripper_signal)

이 프리셋은 lerobot의 ACTPolicy/ACTConfig를 그대로 사용하고, 이 프로젝트의
관측/액션 스펙만 주입한다 (Transformer 자체를 새로 구현하지 않음).
"""

from __future__ import annotations

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig

# docs 4.4절 파라미터와 연동
DT_AI_SEC = 0.020
CHUNK_SIZE = 32

# seam_cv.SeamPoint에서 뽑는 로컬 특징 차원: (lookahead 상대위치 3, curvature 1, thickness 1)
SEAM_FEATURE_DIM = 5
# proprioception: EEF pose 6 (x,y,z,rx,ry,rz) — joint pos 대신 EEF 기준으로 통일 (docs 0절)
STATE_DIM = 6
# action: EEF-delta 6 + gripper/pump 신호 1 (docs 2장 인터페이스)
ACTION_DIM = 7

IMAGE_KEY = "observation.images.wrist"


def build_so101_act_config(camera_height: int = 240, camera_width: int = 320) -> ACTConfig:
    input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, camera_height, camera_width)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(SEAM_FEATURE_DIM,)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,)),
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
