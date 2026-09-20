"""단일 프로세스 SAC 학습용 최소 리플레이 버퍼.

lerobot SAC은 원래 HIL-SERL의 분산 actor/learner 구조(gRPC)를 전제로 하지만, 이 프로젝트는
단일 GPU 워크스테이션에서 학습하므로 간단한 인메모리 버퍼로 충분하다 (train_rl.py에서 사용).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Transition:
    observations: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_observations: dict[str, torch.Tensor]
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cuda"):
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.device = device

    def push(self, obs: dict, action: torch.Tensor, reward: torch.Tensor, next_obs: dict, done: torch.Tensor) -> None:
        """obs/next_obs/action/reward/done은 모두 (num_envs, ...) 배치 텐서 — env별로 분해해서 저장."""
        num_envs = action.shape[0]
        for i in range(num_envs):
            self.buffer.append(
                Transition(
                    observations={k: v[i].detach().cpu() for k, v in obs.items()},
                    action=action[i].detach().cpu(),
                    reward=reward[i].detach().cpu(),
                    next_observations={k: v[i].detach().cpu() for k, v in next_obs.items()},
                    done=done[i].detach().cpu(),
                )
            )

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int) -> dict:
        batch = random.sample(self.buffer, batch_size)
        obs_keys = batch[0].observations.keys()

        def stack_obs(items: list[Transition], attr: str) -> dict[str, torch.Tensor]:
            return {k: torch.stack([getattr(t, attr)[k] for t in items]).to(self.device) for k in obs_keys}

        return {
            "observations": stack_obs(batch, "observations"),
            "action": torch.stack([t.action for t in batch]).to(self.device),
            "reward": torch.stack([t.reward for t in batch]).to(self.device),
            "next_observations": stack_obs(batch, "next_observations"),
            "done": torch.stack([t.done for t in batch]).to(self.device),
        }
