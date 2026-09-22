"""BC(ACT) 체크포인트 로드 + 청크 예측 (학습 스크립트, RL teacher, ActionChunk 직렬화가 공용으로 사용).

체크포인트 폴더 구성 (train_bc.save_checkpoint):
  config.json, model.safetensors, policy_preprocessor.json(+safetensors), policy_postprocessor.json(+safetensors)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def load_bc_checkpoint(ckpt_dir: str | Path, device: str = "cuda"):
    """폴더 하나에서 정책 + 전/후처리(정규화 통계 포함)를 함께 복원한다."""
    ckpt_dir = Path(ckpt_dir)
    policy = ACTPolicy.from_pretrained(ckpt_dir)
    policy.to(device)
    policy.eval()
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        ckpt_dir,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": device}},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        ckpt_dir,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return policy, preprocessor, postprocessor


@torch.no_grad()
def predict_chunk(policy: ACTPolicy, preprocessor, postprocessor, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """관측(배치 or 단일) -> 비정규화된 액션 청크 (B, chunk, 7) [dxyz, drotvec, gripper], CPU 텐서.

    obs 키: observation.images.wrist (3,H,W float[0,1]), observation.state (9,), observation.environment_state (5,)
    """
    batch = preprocessor(dict(obs))
    chunk = policy.predict_action_chunk(batch)  # (B, chunk, 7) 정규화 공간
    b, t, d = chunk.shape
    flat = postprocessor(chunk.reshape(b * t, d))  # 후처리는 (N, action_dim) 텐서를 받는다
    return flat.reshape(b, t, d)


def predict_chunk_np(policy, preprocessor, postprocessor, obs_np: dict[str, np.ndarray]) -> np.ndarray:
    """numpy 단일 관측 -> (chunk, 7) numpy. 실시간 추론 루프용 편의 함수."""
    obs = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32) for k, v in obs_np.items()}
    return predict_chunk(policy, preprocessor, postprocessor, obs)[0].numpy()
