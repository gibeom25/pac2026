"""단일 프로세스 SAC 학습용 최소 리플레이 버퍼.

lerobot SAC은 원래 HIL-SERL의 분산 actor/learner 구조(gRPC)를 전제로 하지만, 이 프로젝트는
단일 GPU 워크스테이션에서 학습하므로 간단한 인메모리 버퍼로 충분하다 (train_rl.py에서 사용).

구현: 고정 크기 리스트 링버퍼 + numpy 랜덤 인덱스. (예전 deque + random.sample은 deque 인덱싱이
O(n)이라 용량 10만에서 샘플 한 번에 수만 번 순회했다.)

2026-10-09: image_uint8=True면 "observation.images.*" 키(0~1 float)를 uint8로 저장하고 꺼낼 때 다시 float로
바꾼다 — 실제 렌더 이미지를 쓰는 envs/piper_seam_env.py용(float 그대로면 메모리가 4배).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    observations: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_observations: dict[str, torch.Tensor]
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cuda", seed: int = 0, image_uint8: bool = False):
        self.capacity = int(capacity)
        self.image_uint8 = image_uint8
        self.buffer: list[Transition | None] = [None] * self.capacity
        self.pos = 0
        self.size = 0
        self.device = device
        self.rng = np.random.default_rng(seed)

    def push(self, obs: dict, action: torch.Tensor, reward: torch.Tensor, next_obs: dict, done: torch.Tensor) -> None:
        """obs/next_obs/action/reward/done은 모두 (num_envs, ...) 배치 텐서 — env별로 분해해서 저장."""
        num_envs = action.shape[0]
        obs_cpu = {k: self._pack(k, v.detach().cpu()) for k, v in obs.items()}
        next_cpu = {k: self._pack(k, v.detach().cpu()) for k, v in next_obs.items()}
        action_cpu, reward_cpu, done_cpu = action.detach().cpu(), reward.detach().cpu(), done.detach().cpu()
        for i in range(num_envs):
            self.buffer[self.pos] = Transition(
                observations={k: v[i].clone() for k, v in obs_cpu.items()},
                action=action_cpu[i].clone(),
                reward=reward_cpu[i].clone(),
                next_observations={k: v[i].clone() for k, v in next_cpu.items()},
                done=done_cpu[i].clone(),
            )
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def _is_image(self, key: str) -> bool:
        return self.image_uint8 and key.startswith("observation.images")

    def _pack(self, key: str, v: torch.Tensor) -> torch.Tensor:
        return v.mul(255).round().clamp(0, 255).to(torch.uint8) if self._is_image(key) else v

    def _unpack(self, key: str, v: torch.Tensor) -> torch.Tensor:
        return v.float().div(255) if self._is_image(key) else v

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.choice(self.size, size=batch_size, replace=False)
        batch = [self.buffer[int(i)] for i in idx]
        obs_keys = batch[0].observations.keys()

        def stack_obs(items: list[Transition], attr: str) -> dict[str, torch.Tensor]:
            return {k: self._unpack(k, torch.stack([getattr(t, attr)[k] for t in items]).to(self.device)) for k in obs_keys}

        return {
            "observations": stack_obs(batch, "observations"),
            "action": torch.stack([t.action for t in batch]).to(self.device),
            "reward": torch.stack([t.reward for t in batch]).to(self.device),
            "next_observations": stack_obs(batch, "next_observations"),
            "done": torch.stack([t.done for t in batch]).to(self.device),
        }
