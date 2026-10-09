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

2026-10-08: 시뮬 사전학습 -> 실물 co-training 흐름을 위해 세 가지를 추가했다.
  - 데이터셋 여러 개: --repo-id/--root에 여러 개를 주고 --weights로 섞는 비율을 정한다(크기와
    무관하게 데이터셋별 비율로 샘플링). 정규화 통계도 같은 비율로 합친다.
  - 고정(overview) 카메라 드롭아웃: configs/so101_act_bc.OVERVIEW_IMAGE_KEY 참고. 키가 없는
    데이터셋(시뮬)은 항상 0, 있는 데이터셋(실물)은 --overview-dropout 확률로 0.
    --no-overview면 예전처럼 손목 카메라만 쓴다.
  - --init-from: 시뮬로 사전학습한 체크포인트에서 가중치를 불러와 이어서 학습(파인튜닝).

2026-10-09: --holdout-every N(데이터셋별, 0이면 안 뺌) + --holdout-offset K — episode_index % N == K인
에피소드를 학습에서 빼 둔다. 체크포인트 train_info.json에 기록되고, tools/eval_bc_offline.py가 그
정보를 읽어 빼 둔 에피소드로 평가한다.

  # 1) 시뮬 사전학습
  PYTHONPATH=. python ai_layer/train_bc.py --repo-id me/sim --epochs 50 --out-dir outputs/bc_sim
  # 2) 실물 부스 여러 개 + 시뮬 co-training
  PYTHONPATH=. python ai_layer/train_bc.py --repo-id me/real-booth1 me/real-booth2 me/sim \
      --weights 1 1 1 --init-from outputs/bc_sim/last --epochs 50 --out-dir outputs/bc_cotrain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

from ai_layer.configs.so101_act_bc import build_so101_act_config
from ai_layer.data import detect_dataset_kind, load_bc_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", nargs="+", required=True, help="데이터셋 repo id (여러 개면 섞어서 학습)")
    p.add_argument("--root", nargs="+", default=None,
                   help="데이터셋 로컬 경로 — --repo-id와 같은 개수/순서 (없으면 datasets/<repo-id>)")
    p.add_argument("--weights", type=float, nargs="+", default=None,
                   help="데이터셋별 샘플링 비율 — --repo-id와 같은 개수 (기본: 전부 1, 즉 데이터셋끼리 균등)")
    p.add_argument("--no-overview", action="store_true", help="고정(overview) 카메라 입력 없이 손목 카메라만 사용")
    p.add_argument("--overview-dropout", type=float, default=0.3,
                   help="overview 키가 있는 데이터셋(실물)에서 학습 중 고정 카메라를 검은 화면으로 가릴 확률")
    p.add_argument("--init-from", default=None, help="이 체크포인트 가중치에서 시작 (시뮬 사전학습 -> 파인튜닝)")
    p.add_argument("--holdout-every", type=int, nargs="+", default=None,
                   help="데이터셋별 N — episode_index %% N == --holdout-offset인 에피소드를 평가용으로 뺀다 (0이면 안 뺌)")
    p.add_argument("--holdout-offset", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="outputs/bc_act")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=10, help="몇 epoch마다 체크포인트를 남길지")
    p.add_argument("--stats-max-samples", type=int, default=2000, help="정규화 통계 계산에 쓸 최대 샘플 수")
    p.add_argument("--no-precompute-seam", action="store_true", help="seam 특징 사전계산(캐시) 끄기")
    p.add_argument(
        "--xyz-only", action="store_true",
        help="학습 타깃에서 회전(roll/pitch/yaw)을 전부 0으로 마스킹 — 4DOF(xyz+그리퍼) MVP용. "
             "녹화 데이터 자체는 안 바뀌므로 나중에 이 플래그 없이 다시 돌리면 회전 포함 버전도 "
             "바로 학습 가능(재녹화 불필요).",
    )
    args = p.parse_args()
    n = len(args.repo_id)
    if args.root is not None and len(args.root) != n:
        p.error(f"--root는 --repo-id와 같은 개수({n})여야 한다.")
    if args.weights is not None and len(args.weights) != n:
        p.error(f"--weights는 --repo-id와 같은 개수({n})여야 한다.")
    if args.holdout_every is not None and len(args.holdout_every) != n:
        p.error(f"--holdout-every는 --repo-id와 같은 개수({n})여야 한다.")
    return args


def holdout_episodes(total_episodes: int, every: int, offset: int) -> list[int]:
    """episode_index % every == offset인 에피소드 목록 (every<=0이면 빈 목록). eval_bc_offline.py도 같은 규칙."""
    if every <= 0:
        return []
    return [e for e in range(total_episodes) if e % every == offset % every]


def merge_stats(stats_list: list[dict], weights: list[float]) -> dict:
    """데이터셋별 mean/std를 샘플링 비율대로 합친다 (혼합분포의 평균/분산)."""
    w = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    merged = {}
    for key in stats_list[0]:
        means = [s[key]["mean"].double() for s in stats_list]
        stds = [s[key]["std"].double() for s in stats_list]
        mean = sum(wi * m for wi, m in zip(w, means))
        var = sum(wi * (sd**2 + m**2) for wi, m, sd in zip(w, means, stds)) - mean**2
        merged[key] = {"mean": mean.float(), "std": var.clamp_min(0).sqrt().float()}
    return merged


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

    use_overview = not args.no_overview
    cfg = build_so101_act_config(use_overview=use_overview)
    cfg.device = str(device)

    roots = args.root or [None] * len(args.repo_id)
    weights = args.weights or [1.0] * len(args.repo_id)
    holdout_every = args.holdout_every or [0] * len(args.repo_id)
    datasets = []
    holdout = {}
    for repo_id, root, every in zip(args.repo_id, roots, holdout_every):
        kind = detect_dataset_kind(repo_id, root)
        extra = {"use_overview": use_overview, "overview_dropout": args.overview_dropout} if kind == "ee" else {}
        if every > 0:
            if kind != "ee":
                raise SystemExit(f"{repo_id}: --holdout-every는 EE 데이터셋만 지원한다.")
            from ai_layer.data import resolve_dataset_root
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            meta = LeRobotDatasetMetadata(repo_id, root=resolve_dataset_root(repo_id, root))
            holdout[repo_id] = holdout_episodes(meta.total_episodes, every, args.holdout_offset)
            extra["exclude_episodes"] = holdout[repo_id]
        if kind != "ee" and use_overview:
            raise SystemExit(f"{repo_id}: 관절공간(joint) 데이터셋은 overview 카메라를 지원하지 않는다 — --no-overview로 학습할 것.")
        ds = load_bc_dataset(
            repo_id, root=root, precompute_seam=not args.no_precompute_seam, xyz_only=args.xyz_only, **extra,
        )
        overview = getattr(ds, "has_overview", False)
        print(f"dataset {repo_id} kind={kind} frames={len(ds)} fps={ds.raw.fps} overview={'있음' if overview else '없음(0)'}"
              + (f" holdout={holdout[repo_id]}" if repo_id in holdout else ""))
        datasets.append(ds)
    print(f"xyz_only={args.xyz_only} use_overview={use_overview} overview_dropout={args.overview_dropout} weights={weights}")

    # 변환 후(EEF pose/delta/seam) 값으로 정규화 통계 계산 → 전/후처리 파이프라인에 주입.
    # 여러 데이터셋이면 샘플링 비율대로 합친다(학습 때 실제로 보는 분포와 맞춤).
    per_ds_samples = max(1, args.stats_max_samples // len(datasets))
    stats = merge_stats([ds.compute_stats(max_samples=per_ds_samples) for ds in datasets], weights)
    preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=stats)

    dataset = ConcatDataset(datasets)
    # 데이터셋 크기와 무관하게 --weights 비율로 뽑히도록 샘플 가중치 = 비율 / 데이터셋 길이
    sample_w = torch.cat([torch.full((len(ds),), w / len(ds), dtype=torch.double) for ds, w in zip(datasets, weights)])
    sampler = WeightedRandomSampler(sample_w, num_samples=len(dataset), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )

    if args.init_from:
        policy = ACTPolicy.from_pretrained(args.init_from)
        if set(policy.config.image_features) != set(cfg.image_features):
            raise SystemExit(
                f"--init-from 체크포인트의 카메라 입력 {sorted(policy.config.image_features)} 이 지금 설정 "
                f"{sorted(cfg.image_features)} 과 다르다 — 사전학습 때와 같은 --no-overview 설정으로 학습할 것."
            )
        policy.config.device = str(device)
        print(f"init from {args.init_from}")
    else:
        policy = ACTPolicy(cfg)
    policy.to(device)
    policy.train()

    optim_cfg = cfg.get_optimizer_preset()
    optimizer = optim_cfg.build(policy.parameters())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2026-10-06: train_rl.py의 metrics.jsonl과 같은 패턴 — PyQt GUI 학습 탭이 콘솔 출력을
    # regex로 긁는 대신 이 파일을 tail해서 loss 곡선을 그릴 수 있게.
    metrics_path = out_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a")

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
                metrics_file.write(json.dumps({"epoch": epoch, "step": step, "loss": last_loss}) + "\n")
                metrics_file.flush()
            step += 1

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            save_checkpoint(
                out_dir, f"epoch{epoch:04d}", policy, preprocessor, postprocessor,
                {"epoch": epoch, "step": step, "loss": last_loss, "repo_id": args.repo_id, "weights": weights, "xyz_only": args.xyz_only,
                 "use_overview": use_overview, "overview_dropout": args.overview_dropout, "init_from": args.init_from,
                 "holdout": holdout},
            )

    final = save_checkpoint(
        out_dir, "last", policy, preprocessor, postprocessor,
        {"epoch": args.epochs - 1, "step": step, "loss": last_loss, "repo_id": args.repo_id, "weights": weights, "xyz_only": args.xyz_only,
                 "use_overview": use_overview, "overview_dropout": args.overview_dropout, "init_from": args.init_from,
                 "holdout": holdout},
    )
    metrics_file.close()
    print(f"done. checkpoints in {out_dir} (latest: {final})")


if __name__ == "__main__":
    main()
