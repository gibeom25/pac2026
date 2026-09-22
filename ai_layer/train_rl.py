#!/usr/bin/env python
"""RL(SAC) 학습 진입점. 설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.

2026-09-22: MuJoCo 환경(envs/so101_seam_env.py)으로 전환되어 IsaacLab 의존성 없음. 일반 파이썬 스크립트.
LeRobot 0.4.x: 정규화는 make_sac_pre_post_processors(cfg, stats) 파이프라인이 한다 (관측만 통과, 액션은 [-1,1] 그대로).
BC teacher 는 train_bc.py 폴더 체크포인트를 bc_inference.load_bc_checkpoint 로 읽는다.

사용 예 (env_lerobot):
  cd pac2026-team
  PYTHONPATH=. python ai_layer/train_rl.py --num-steps 200000
  PYTHONPATH=. python ai_layer/train_rl.py --num-steps 200000 --bc-checkpoint outputs/bc_act/last
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors

from ai_layer.configs.so101_sac import build_sac_dataset_stats, build_so101_sac_config
from ai_layer.envs.so101_seam_env import SO101SeamEnv, SO101SeamEnvCfg
from ai_layer.rl.replay_buffer import ReplayBuffer


def load_bc_reference(checkpoint_dir: str | None, device: str):
    """3.4절 R_imitation 계산용 BC(ACT) teacher. 없으면 (None, None, None)."""
    if checkpoint_dir is None:
        return None, None, None
    from ai_layer.bc_inference import load_bc_checkpoint

    return load_bc_checkpoint(checkpoint_dir, device=device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 seam-following SAC training (MuJoCo).")
    p.add_argument("--num-steps", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-dir", default="outputs/rl_sac")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--bc-checkpoint", default=None, help="train_bc.py 체크포인트 폴더 (R_imitation 용, 선택)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args_cli = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args_cli.seed)

    env = SO101SeamEnv(SO101SeamEnvCfg())
    env.set_total_env_steps(args_cli.num_steps)

    sac_cfg = build_so101_sac_config()
    sac_cfg.device = device
    sac_cfg.storage_device = device
    policy = SACPolicy(sac_cfg)
    policy.to(device)
    preprocessor, _postprocessor = make_sac_pre_post_processors(sac_cfg, dataset_stats=build_sac_dataset_stats())

    bc_policy, bc_pre, bc_post = load_bc_reference(args_cli.bc_checkpoint, device)
    env.set_bc_reference(bc_policy, bc_pre, bc_post)

    optim_params = policy.get_optim_params()
    optimizers = {
        "actor": torch.optim.Adam(optim_params["actor"], lr=sac_cfg.actor_lr),
        "critic": torch.optim.Adam(optim_params["critic"], lr=sac_cfg.critic_lr),
        "temperature": torch.optim.Adam([optim_params["temperature"]], lr=sac_cfg.temperature_lr),
    }
    if "discrete_critic" in optim_params:
        optimizers["discrete_critic"] = torch.optim.Adam(optim_params["discrete_critic"], lr=sac_cfg.critic_lr)

    buffer = ReplayBuffer(capacity=sac_cfg.online_buffer_capacity, device=device, seed=args_cli.seed)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def norm_obs(obs: dict) -> dict:
        """관측 dict(배치) -> 정규화 + device. 파이프라인 출력에서 관측 키만 남긴다 (None/스칼라 제거)."""
        out = preprocessor(dict(obs))
        return {k: v for k, v in out.items() if k.startswith("observation.") and torch.is_tensor(v)}

    obs, _ = env.reset(seed=args_cli.seed)
    step = 0
    critic_loss = torch.tensor(float("nan"))
    while step < args_cli.num_steps:
        with torch.no_grad():
            if step < sac_cfg.online_step_before_learning:
                action = torch.rand(env.num_envs, 7, device=device) * 2 - 1
                action[:, -1] = (action[:, -1] > 0).float()
            else:
                action = policy.select_action(norm_obs(obs))

        next_obs, reward, terminated, truncated, _ = env.step(action.cpu())
        done = torch.tensor([float(terminated or truncated)])
        reward_t = torch.tensor([reward])
        buffer.push(obs, action, reward_t, next_obs, done)
        obs = next_obs
        if truncated or terminated:
            obs, _ = env.reset()

        if len(buffer) >= args_cli.batch_size and step >= sac_cfg.online_step_before_learning:
            batch = buffer.sample(args_cli.batch_size)
            b_obs = norm_obs(batch["observations"])
            b_next = norm_obs(batch["next_observations"])

            critic_loss = policy.compute_loss_critic(
                observations=b_obs, actions=batch["action"], rewards=batch["reward"],
                next_observations=b_next, done=batch["done"],
            )
            optimizers["critic"].zero_grad()
            critic_loss.backward()
            optimizers["critic"].step()

            if "discrete_critic" in optimizers:
                discrete_critic_loss = policy.compute_loss_discrete_critic(
                    observations=b_obs, actions=batch["action"], rewards=batch["reward"],
                    next_observations=b_next, done=batch["done"],
                )
                optimizers["discrete_critic"].zero_grad()
                discrete_critic_loss.backward()
                optimizers["discrete_critic"].step()

            if step % sac_cfg.policy_update_freq == 0:
                actor_loss = policy.compute_loss_actor(observations=b_obs)
                optimizers["actor"].zero_grad()
                actor_loss.backward()
                optimizers["actor"].step()

                temp_loss = policy.compute_loss_temperature(observations=b_obs)
                optimizers["temperature"].zero_grad()
                temp_loss.backward()
                optimizers["temperature"].step()

            policy.update_target_networks()

            if step % args_cli.log_every == 0:
                print(f"step={step} critic_loss={critic_loss.item():.4f} reward={reward:.4f}")

        if step % args_cli.ckpt_every == 0 and step > 0:
            policy.save_pretrained(out_dir / f"sac_step{step:07d}")

        step += 1

    policy.save_pretrained(out_dir / "sac_final")
    print(f"done. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
