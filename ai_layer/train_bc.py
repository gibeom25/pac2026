#!/usr/bin/env python
"""BC(ACT) 학습 진입점.

설계 문서: docs/AI_추론계층_프레임워크.md 3.3절.
lerobot의 ACTPolicy를 그대로 사용하고, 이 프로젝트의 데이터 변환(SO101BCDataset)과
관측/액션 스펙(configs/so101_act_bc.py)만 주입한다.

사용 예:
  python -m ai_layer.train_bc --repo-id <hf-user>/so101-weld-demo --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.policies.act.modeling_act import ACTPolicy

from ai_layer.configs.so101_act_bc import build_so101_act_config
from ai_layer.data.so101_bc_dataset import SO101BCDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="lerobot-record로 만든 데이터셋 repo id 또는 로컬 경로")
    p.add_argument("--root", default=None, help="로컬에 데이터셋이 있으면 그 경로 (repo_id는 이름표로만 사용)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="outputs/bc_act")
    p.add_argument("--log-every", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = build_so101_act_config()
    dataset = SO101BCDataset(repo_id=args.repo_id, root=args.root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    policy = ACTPolicy(cfg)
    policy.to(device)
    policy.train()

    optim_cfg = cfg.get_optimizer_preset()
    optimizer = optim_cfg.build(policy.parameters())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, loss_dict = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step} loss={loss.item():.4f} {loss_dict}")
            step += 1

        ckpt_path = out_dir / f"act_epoch{epoch:04d}.pt"
        torch.save(policy.state_dict(), ckpt_path)

    print(f"done. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
