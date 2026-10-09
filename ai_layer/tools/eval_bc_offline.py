#!/usr/bin/env python
"""BC(ACT) 체크포인트 오프라인 평가 — 학습에서 빼 둔 실물 시연에서 "정책이 내놓는 1초 경로"가
사람이 실제로 움직인 경로와 얼마나 다른지 mm/도 단위로 잰다.

로봇을 돌리지 않고 체크포인트 여러 개(예: 고정캠+손목캠 vs 손목캠 전용)를 같은 프레임에서 비교하는
용도다. 각 평가 프레임 t에서 정책이 예측한 action chunk(32스텝 ≈ 1.07초)를 누적해 경로로 만들고,
같은 구간의 실제 action(=실측 말단 이동)을 누적한 경로와 비교한다.

지표:
  pos@0.33s, pos@1.07s  예측 경로 끝점과 실제 경로 끝점의 거리 [mm] (10스텝, 32스텝 누적)
  heading               1.07초 동안의 수평 이동 방향 차이 [도] — 실제로 5mm 이상 움직인 프레임만
  rot@1.07s             roll/pitch 누적 회전 차이 [도] (yaw는 학습 타깃에서 0이라 제외)
  trig                  트리거(7번째 값) 0.5 기준 일치율
  moving                위 지표를 "실제로 1초 동안 5mm 이상 움직인 프레임"만으로 다시 계산(멈춰 있는 구간이
                        많아서 전체 평균만 보면 차이가 묻힌다)
비교 기준(baseline)으로 "정지(0 예측)"와 "등속(직전 프레임 이동을 그대로 반복)"도 같이 출력한다. 정책이
등속 baseline보다 나아야 이미지/seam을 실제로 쓰고 있다는 뜻이다.

고정캠 입력이 있는 체크포인트는 두 번 평가한다: 고정캠을 보여 줄 때 / 검은 화면으로 가릴 때
(고정캠이 흔들리거나 빠졌을 때 얼마나 버티는지).

평가 에피소드는 기본적으로 첫 체크포인트의 train_info.json "holdout"(train_bc.py --holdout-every)에서
읽는다. 한계: 열린 루프(open-loop) 평가라 정책 자신의 실수가 누적되는 효과는 못 본다 — 실물 주행의
대리 지표일 뿐이다.

사용 예:
  PYTHONPATH=. .venv/bin/python ai_layer/tools/eval_bc_offline.py \\
      --checkpoint outputs/bc_piper_ho/last outputs/bc_piper_wrist_ho/last \\
      --repo-id ddyyuu/piper-user-20261008-flange-2cam-v1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ai_layer.bc_inference import load_bc_checkpoint, predict_chunk
from ai_layer.configs.so101_act_bc import IMAGE_KEY, OVERVIEW_IMAGE_KEY, ZERO_YAW
from ai_layer.data import require_local_dataset
from ai_layer.data.so101_ee_dataset import SO101EEDataset
from ai_layer.kinematics import YAW_INDEX

HORIZONS = (10, 32)  # 스텝 (30 Hz) — 0.33초, 1.07초
MOVING_MM = 5.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True, help="비교할 체크포인트 폴더들")
    p.add_argument("--repo-id", required=True, help="평가 데이터셋 (보통 실물)")
    p.add_argument("--root", default=None)
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="평가 에피소드 (없으면 첫 체크포인트 train_info.json의 holdout)")
    p.add_argument("--stride", type=int, default=3, help="몇 프레임마다 평가할지")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="결과 JSON 경로 (기본: 첫 체크포인트 폴더/eval_<데이터셋>.json)")
    return p.parse_args()


def eval_episodes(args) -> list[int]:
    if args.episodes:
        return args.episodes
    info_path = Path(args.checkpoint[0]) / "train_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    eps = info.get("holdout", {}).get(args.repo_id)
    if not eps:
        raise SystemExit(
            f"{info_path}에 {args.repo_id}의 holdout 기록이 없다 — train_bc.py --holdout-every로 학습했거나 "
            "--episodes를 직접 줄 것. (학습에 쓴 에피소드로 평가하면 외운 걸 재는 셈이라 의미가 없다.)"
        )
    for ck in args.checkpoint[1:]:
        other = json.loads((Path(ck) / "train_info.json").read_text()).get("holdout", {}).get(args.repo_id)
        if other != eps:
            print(f"[경고] {ck}의 holdout {other} 이 첫 체크포인트 {eps} 과 다르다 — 공정한 비교가 아님.")
    return eps


def scene_labels(root: Path) -> dict[int, str]:
    """실물 Piper 데이터셋의 meta/piper_capture.jsonl에 있는 장면 라벨 (없으면 빈 dict)."""
    path = root / "meta" / "piper_capture.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        d = json.loads(line)
        if "scene" in d:
            out[int(d["episode_index"])] = str(d["scene"])
    return out


def chunk_metrics(pred: np.ndarray, gt: np.ndarray, pad: np.ndarray) -> dict[str, np.ndarray]:
    """(B,T,7) 예측/정답, (B,T) pad -> 샘플별 지표 (유효하지 않으면 nan)."""
    out = {}
    cp, cg = np.cumsum(pred, axis=1), np.cumsum(gt, axis=1)
    for k in HORIZONS:
        valid = ~pad[:, k - 1]
        e = np.linalg.norm(cp[:, k - 1, :3] - cg[:, k - 1, :3], axis=1) * 1000.0
        out[f"pos@{k}"] = np.where(valid, e, np.nan)
    k = HORIZONS[-1]
    valid = ~pad[:, k - 1]
    gxy, pxy = cg[:, k - 1, :2], cp[:, k - 1, :2]
    moving = valid & (np.linalg.norm(cg[:, k - 1, :3], axis=1) * 1000.0 > MOVING_MM)
    cos = (gxy * pxy).sum(1) / (np.linalg.norm(gxy, axis=1) * np.linalg.norm(pxy, axis=1) + 1e-9)
    heading = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    out["heading"] = np.where(moving & (np.linalg.norm(gxy, axis=1) * 1000.0 > MOVING_MM), heading, np.nan)
    rot = np.degrees(np.linalg.norm(cp[:, k - 1, 3:5] - cg[:, k - 1, 3:5], axis=1))
    out["rot@32"] = np.where(valid, rot, np.nan)
    agree = ((pred[..., 6] > 0.5) == (gt[..., 6] > 0.5)).astype(np.float64)
    agree[pad] = np.nan
    with np.errstate(invalid="ignore"):
        out["trig"] = np.nanmean(agree, axis=1)
    out["moving"] = moving
    return out


def summarize(m: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, float]:
    sel = np.ones(len(m["moving"]), dtype=bool) if mask is None else mask
    res = {}
    for key in (f"pos@{HORIZONS[0]}", f"pos@{HORIZONS[1]}", "heading", "rot@32", "trig"):
        v = m[key][sel]
        res[key] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    res["n"] = int(sel.sum())
    return res


def main() -> None:
    args = parse_args()
    root = require_local_dataset(args.repo_id, args.root)
    episodes = eval_episodes(args)
    scenes = scene_labels(root)

    datasets = {}

    def dataset(dropout: float) -> SO101EEDataset:
        if dropout not in datasets:
            datasets[dropout] = SO101EEDataset(
                args.repo_id, root=root, only_episodes=episodes, use_overview=True,
                overview_dropout=dropout, augment=False,
            )
        return datasets[dropout]

    base = dataset(0.0)
    sample_ids = np.arange(0, len(base), args.stride)
    raw_rows = base._index[sample_ids]
    ep_of = base.episode_index[raw_rows]
    print(f"평가 데이터: {args.repo_id} 에피소드 {len(episodes)}개 {episodes}, 프레임 {len(sample_ids)}개 (stride {args.stride})")

    # 정답 chunk와 baseline은 체크포인트와 무관 — 한 번만 만든다
    gts, pads = [], []
    for r in raw_rows:
        a, pad = base._action_chunk(int(r))
        gts.append(a.numpy())
        pads.append(pad.numpy())
    gt, pad = np.stack(gts), np.stack(pads)
    cur = base._actions[raw_rows].copy()
    if ZERO_YAW:
        gt[..., YAW_INDEX] = 0.0
        cur[:, YAW_INDEX] = 0.0
    baselines = {
        "baseline: 정지": np.zeros_like(gt),
        "baseline: 등속": np.repeat(cur[:, None, :], gt.shape[1], axis=1),
    }

    rows: dict[str, dict[str, np.ndarray]] = {}
    for name, pred in baselines.items():
        rows[name] = chunk_metrics(pred, gt, pad)

    for ck in args.checkpoint:
        policy, pre, post = load_bc_checkpoint(ck, device=args.device)
        uses_overview = OVERVIEW_IMAGE_KEY in policy.config.image_features
        conds = [("고정캠+손목캠", 0.0), ("고정캠 가림", 1.0)] if uses_overview and base.has_overview else [("손목캠", 0.0)]
        keys = [IMAGE_KEY, OBS_STATE, OBS_ENV_STATE] + ([OVERVIEW_IMAGE_KEY] if uses_overview else [])
        for cname, dropout in conds:
            loader = DataLoader(Subset(dataset(dropout), sample_ids.tolist()), batch_size=args.batch_size,
                                num_workers=args.num_workers, shuffle=False)
            preds = []
            for batch in loader:
                obs = {k: batch[k].to(args.device) for k in keys}
                preds.append(predict_chunk(policy, pre, post, obs).numpy())
            pred = np.concatenate(preds)
            if ZERO_YAW:
                pred[..., YAW_INDEX] = 0.0
            label = f"{Path(ck).parent.name}/{Path(ck).name} [{cname}]"
            rows[label] = chunk_metrics(pred, gt, pad)
            print(f"  done: {label}")
        del policy
        torch.cuda.empty_cache()

    cols = [f"pos@{HORIZONS[0]}", f"pos@{HORIZONS[1]}", "heading", "rot@32", "trig"]
    head = ["pos@0.33s[mm]", "pos@1.07s[mm]", "heading[°]", "rot@1.07s[°]", "trig일치"]
    width = max(len(k) for k in rows) + 2
    result = {"repo_id": args.repo_id, "episodes": episodes, "stride": args.stride, "rows": {}}

    def table(title: str, mask_fn) -> None:
        print(f"\n== {title} ==")
        print("".ljust(width) + "".join(h.rjust(15) for h in head) + "n".rjust(8))
        for name, m in rows.items():
            s = summarize(m, mask_fn(m))
            result["rows"].setdefault(name, {})[title] = s
            print(name.ljust(width) + "".join(f"{s[c]:15.2f}" for c in cols) + f"{s['n']:8d}")

    table("전체 프레임", lambda m: None)
    table(f"움직이는 프레임 (1.07초에 >{MOVING_MM:g}mm)", lambda m: m["moving"])

    if scenes:
        names = sorted({scenes.get(int(e), "?") for e in ep_of})
        print(f"\n== 장면별 pos@1.07s [mm], 움직이는 프레임 ==")
        print("".ljust(width) + "".join(n[:12].rjust(13) for n in names))
        for name, m in rows.items():
            vals = []
            for sc in names:
                sel = m["moving"] & np.array([scenes.get(int(e), "?") == sc for e in ep_of])
                vals.append(summarize(m, sel)[f"pos@{HORIZONS[1]}"])
                result["rows"][name].setdefault("scene", {})[sc] = vals[-1]
            print(name.ljust(width) + "".join(f"{v:13.2f}" for v in vals))

    out = Path(args.out) if args.out else Path(args.checkpoint[0]) / f"eval_{args.repo_id.replace('/', '__')}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
