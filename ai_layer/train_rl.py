#!/usr/bin/env python
"""RL(SAC) 학습 진입점. 설계 문서: docs/AI_추론계층_프레임워크.md 3.4절.

⚠️ IsaacLab이 필요하므로 반드시 사용자 터미널에서 직접 실행할 것 (이 세션/Bash 도구 안에서는
CUDA P2P 검증 단계에서 멈춤 — ai_layer_welding_trajectory_design.md 메모리 참고).

사용 예:
  ./isaaclab.sh -p ai_layer/train_rl.py --headless --num-steps 200000
  ./isaaclab.sh -p ai_layer/train_rl.py --headless --bc-checkpoint outputs/bc_act/act_epoch0099.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 seam-following SAC training.")
parser.add_argument("--num-steps", type=int, default=200_000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--out-dir", default="outputs/rl_sac")
parser.add_argument("--log-every", type=int, default=200)
parser.add_argument("--ckpt-every", type=int, default=5000)
parser.add_argument(
    "--bc-checkpoint", default=None, help="train_bc.py가 만든 ACT 체크포인트 (R_imitation 항에 사용, 선택)"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""이 아래부터는 시뮬레이터 앱이 뜬 뒤에만 import 가능."""

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.policies.sac.modeling_sac import SACPolicy  # noqa: E402

from ai_layer.configs.so101_sac import build_so101_sac_config  # noqa: E402
from ai_layer.envs.so101_seam_env import SO101SeamEnv, SO101SeamEnvCfg  # noqa: E402
from ai_layer.rl.replay_buffer import ReplayBuffer  # noqa: E402


def load_bc_reference(checkpoint_path: str | None, device: str):
    """3.4절: R_imitation 계산용 BC(ACT) teacher. 없으면 None (모방항 0, 3.4 가중치 스케줄에서 w1도 낮게)."""
    if checkpoint_path is None:
        return None
    from ai_layer.configs.so101_act_bc import build_so101_act_config
    from lerobot.policies.act.modeling_act import ACTPolicy

    cfg = build_so101_act_config()
    policy = ACTPolicy(cfg)
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.to(device)
    policy.eval()
    return policy


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_cfg = SO101SeamEnvCfg()
    env = SO101SeamEnv(cfg=env_cfg)

    sac_cfg = build_so101_sac_config()
    sac_cfg.device = device
    sac_cfg.storage_device = device
    policy = SACPolicy(sac_cfg)
    policy.to(device)

    bc_ref = load_bc_reference(args_cli.bc_checkpoint, device)
    env.set_bc_reference(bc_ref)

    optim_params = policy.get_optim_params()
    optimizers = {
        "actor": torch.optim.Adam(optim_params["actor"], lr=sac_cfg.actor_lr),
        "critic": torch.optim.Adam(optim_params["critic"], lr=sac_cfg.critic_lr),
        "temperature": torch.optim.Adam([optim_params["temperature"]], lr=sac_cfg.temperature_lr),
    }
    if "discrete_critic" in optim_params:
        optimizers["discrete_critic"] = torch.optim.Adam(optim_params["discrete_critic"], lr=sac_cfg.critic_lr)

    buffer = ReplayBuffer(capacity=sac_cfg.online_buffer_capacity, device=device)

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    step = 0
    while step < args_cli.num_steps and simulation_app.is_running():
        with torch.no_grad():
            if step < sac_cfg.online_step_before_learning:
                action = torch.rand(env.num_envs, 7, device=device) * 2 - 1
                action[:, -1] = (action[:, -1] > 0).float()
            else:
                action = policy.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = (terminated | truncated).float()
        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        if len(buffer) >= args_cli.batch_size and step >= sac_cfg.online_step_before_learning:
            batch = buffer.sample(args_cli.batch_size)

            critic_loss = policy.compute_loss_critic(
                observations=batch["observations"],
                actions=batch["action"],
                rewards=batch["reward"],
                next_observations=batch["next_observations"],
                done=batch["done"],
            )
            optimizers["critic"].zero_grad()
            critic_loss.backward()
            optimizers["critic"].step()

            if "discrete_critic" in optimizers:
                discrete_critic_loss = policy.compute_loss_discrete_critic(
                    observations=batch["observations"],
                    actions=batch["action"],
                    rewards=batch["reward"],
                    next_observations=batch["next_observations"],
                    done=batch["done"],
                )
                optimizers["discrete_critic"].zero_grad()
                discrete_critic_loss.backward()
                optimizers["discrete_critic"].step()

            if step % sac_cfg.policy_update_freq == 0:
                actor_loss = policy.compute_loss_actor(observations=batch["observations"])
                optimizers["actor"].zero_grad()
                actor_loss.backward()
                optimizers["actor"].step()

                temp_loss = policy.compute_loss_temperature(observations=batch["observations"])
                optimizers["temperature"].zero_grad()
                temp_loss.backward()
                optimizers["temperature"].step()

            policy.update_target_networks()

            if step % args_cli.log_every == 0:
                print(
                    f"step={step} critic_loss={critic_loss.item():.4f} "
                    f"reward_mean={reward.mean().item():.4f}"
                )

        if step % args_cli.ckpt_every == 0 and step > 0:
            torch.save(policy.state_dict(), out_dir / f"sac_step{step:07d}.pt")

        step += env.num_envs

    torch.save(policy.state_dict(), out_dir / "sac_final.pt")
    print(f"done. checkpoints in {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
