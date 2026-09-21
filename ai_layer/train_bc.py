#!/usr/bin/env python
"""BC(ACT) 학습 진입점.

설계 문서: docs/AI_추론계층_프레임워크.md 3.3절.
lerobot의 ACTPolicy를 그대로 사용하고, 이 프로젝트의 데이터 변환(SO101BCDataset)과
관측/액션 스펙(configs/so101_act_bc.py)만 주입한다.

LeRobot 0.4.x 구조:
  - 정규화는 정책 안이 아니라 preprocessor/postprocessor 파이프라인이 한다.
    (make_act_pre_post_processors + SO101BCDataset.compute_stats)
  - 체크포인트는 policy.save_pretrained + processor.save_pretrained 로 한 폴더에
    config.json / model.safetensors / policy_preprocessor.json(+stats) / policy_postprocessor.json
    을 같이 저장한다. 추론 시 이 폴더 하나만 있으면 같은 정규화로 복원된다.

사용 예:
  cd pac2026-team
  PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/train_bc.py \
      --repo-id <hf-user>/so101-weld-demo --root <로컬 경로> --epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

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
    p.add_argument("--save-every", type=int, default=10, help="몇 epoch마다 체크포인트를 남길지")
    p.add_argument("--stats-max-samples", type=int, default=2000, help="정규화 통계 계산에 쓸 최대 샘플 수")
    p.add_argument("--no-precompute-seam", action="store_true", help="seam 특징 사전계산(캐시) 끄기")
    return p.parse_args()


def save_checkpoint(out_dir: Path, tag: str, policy: ACTPolicy, preprocessor, postprocessor, extra: dict) -> Path:
    ckpt_dir = out_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    preprocessor.save_pretrained(ckpt_dir)
    postprocessor.save_pretrained(ckpt_dir)
    (ckpt_dir / "train_info.json").write_text(json.dumps(extra, indent=2, ensure_ascii=False))
    return ckpt_dir


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = build_so101_act_config()
    cfg.device = str(device)

    dataset = SO101BCDataset(repo_id=args.repo_id, root=args.root, precompute_seam=not args.no_precompute_seam)
    print(f"dataset frames={len(dataset)} fps={dataset.raw.fps}")

    # 변환 후(EEF pose/delta/seam) 값으로 정규화 통계 계산 → 전/후처리 파이프라인에 주입
    stats = dataset.compute_stats(max_samples=args.stats_max_samples)
    preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=stats)

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
    last_loss = float("nan")
    for epoch in range(args.epochs):
        for batch in loader:
            batch = preprocessor(batch)  # 정규화 + device 이동 (action_is_pad 등은 그대로 통과)
            loss, loss_dict = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step} loss={last_loss:.4f} {loss_dict}")
            step += 1

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                out_dir, f"epoch{epoch:04d}", policy, preprocessor, postprocessor,
                {"epoch": epoch, "step": step, "loss": last_loss, "repo_id": args.repo_id},
            )

    final = save_checkpoint(
        out_dir, "last", policy, preprocessor, postprocessor,
        {"epoch": args.epochs - 1, "step": step, "loss": last_loss, "repo_id": args.repo_id},
    )
    print(f"done. checkpoints in {out_dir} (latest: {final})")


if __name__ == "__main__":
    main()
