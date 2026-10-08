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
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from ai_layer.configs.so101_act_bc import build_so101_act_config
from ai_layer.data import detect_dataset_kind, load_bc_dataset


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
    p.add_argument(
        "--xyz-only", action="store_true",
        help="학습 타깃에서 회전(roll/pitch/yaw)을 전부 0으로 마스킹 — 4DOF(xyz+그리퍼) MVP용. "
             "녹화 데이터 자체는 안 바뀌므로 나중에 이 플래그 없이 다시 돌리면 회전 포함 버전도 "
             "바로 학습 가능(재녹화 불필요).",
    )
    p.add_argument(
        "--init-checkpoint", default=None,
        help="처음부터(ACTPolicy(cfg)) 새로 만드는 대신 이 체크포인트 폴더(예: outputs/bc_act/last)에서 "
             "가중치+전/후처리 정규화 통계를 그대로 불러와서 이어서 학습(fine-tuning)한다. 2026-10-08: "
             "실로봇 데이터로 fine-tuning하거나, RL로 다듬은 롤아웃(rl_rollout_to_dataset.py 참고)을 "
             "증류할 때 씀. 정규화 통계는 지금 데이터셋에서 새로 안 뽑고 체크포인트에 저장된 걸 그대로 "
             "쓴다(이미 학습된 가중치가 그 정규화 기준으로 특징을 배웠으므로 바꾸면 안 맞아짐).",
    )
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

    kind = detect_dataset_kind(args.repo_id, args.root)
    dataset = load_bc_dataset(
        args.repo_id, root=args.root, precompute_seam=not args.no_precompute_seam, xyz_only=args.xyz_only,
    )
    print(f"dataset kind={kind} xyz_only={args.xyz_only} frames={len(dataset)} fps={dataset.raw.fps}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    if args.init_checkpoint is not None:
        # fine-tuning: 가중치 + 정규화 통계를 체크포인트에서 그대로 가져온다 — 지금 데이터셋에서
        # 통계를 새로 뽑지 않는다(이미 학습된 가중치가 기존 정규화 기준으로 특징을 배웠으므로,
        # fine-tuning 데이터가 적거나 분포가 조금 달라도 정규화 기준 자체는 유지해야 함).
        print(f"init from checkpoint: {args.init_checkpoint}")
        # config=cfg를 넘겨야 한다 — 안 그러면 from_pretrained가 pretrained_name_or_path를
        # 로컬 경로로 보기 전에 먼저 PreTrainedConfig.from_pretrained()로 HF Hub repo_id인 것처럼
        # config.json을 내려받으려다("경로에 '/'가 있으니 네임스페이스/이름이겠지") 실패한다
        # (실측 확인된 버그 — HFValidationError). config를 이미 들고 있으니 그걸 바로 넘겨서
        # 그 단계 자체를 건너뛴다.
        policy = ACTPolicy.from_pretrained(args.init_checkpoint, config=cfg)
        # save_pretrained()이 실제로 쓰는 파일명은 "policy_preprocessor.json"(.json 포함)인데
        # POLICY_PREPROCESSOR_DEFAULT_NAME 상수엔 확장자가 없어서 from_pretrained가 그대로 찾다
        # FileNotFoundError 난다(실측 확인) — .json을 직접 붙여서 넘긴다.
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            args.init_checkpoint, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            args.init_checkpoint, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
        )
    else:
        # 변환 후(EEF pose/delta/seam) 값으로 정규화 통계 계산 → 전/후처리 파이프라인에 주입
        stats = dataset.compute_stats(max_samples=args.stats_max_samples)
        preprocessor, postprocessor = make_act_pre_post_processors(cfg, dataset_stats=stats)
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
                {"epoch": epoch, "step": step, "loss": last_loss, "repo_id": args.repo_id, "xyz_only": args.xyz_only, "init_checkpoint": args.init_checkpoint},
            )

    final = save_checkpoint(
        out_dir, "last", policy, preprocessor, postprocessor,
        {"epoch": args.epochs - 1, "step": step, "loss": last_loss, "repo_id": args.repo_id, "xyz_only": args.xyz_only, "init_checkpoint": args.init_checkpoint},
    )
    metrics_file.close()
    print(f"done. checkpoints in {out_dir} (latest: {final})")


if __name__ == "__main__":
    main()
