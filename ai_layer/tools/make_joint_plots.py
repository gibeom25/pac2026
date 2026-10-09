#!/usr/bin/env python
"""joint_dynamics_bench.py 결과 그래프 — docs/Experiment_Plan.md "조인트 동역학 검증" 섹션용.
처음부터 영어로 쓴다(이전 세션에서 matplotlib 기본 폰트가 한글 글리프를 못 그려 깨진 적 있음).

실행: PYTHONPATH=. python ai_layer/tools/make_joint_plots.py --out-dir docs/ablation_results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai_layer.tools.joint_dynamics_bench import AblationConfig, run_joint_benchmark

BASELINE_CFG = AblationConfig(anchor_level="none", buffer_baseline=True, trigger_baseline=True, use_corrector=False)
PROPOSED_CFG = AblationConfig(anchor_level="commit_refine", buffer_baseline=False, trigger_baseline=False, use_corrector=True)
TORQUE_LIMIT_NM = 3.35  # so101_new_calib.xml actuator forcerange


def _arr(log: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in log])


def plot_progress(baseline: dict, proposed: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, res, color in [("baseline", baseline, "tab:red"), ("proposed", proposed, "tab:blue")]:
        log = res["log"]
        t = (_arr(log, "t_ns") - log[0]["t_ns"]) * 1e-9
        prog = _arr(log, "progress")
        ax.plot(t, prog, label=name, color=color, linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("path progress (arc length) [m]")
    ax.set_title("Joint-dynamics plant — progress vs time (backward jumps = problem #8 oscillation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "10_joint_progress.png", dpi=130)
    plt.close(fig)


def plot_ik_tracking_error(baseline: dict, proposed: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, res, color in [("baseline", baseline, "tab:red"), ("proposed", proposed, "tab:blue")]:
        log = res["log"]
        t = (_arr(log, "t_ns") - log[0]["t_ns"]) * 1e-9
        err = _arr(log, "ik_tracking_err_m") * 1e3
        ax.plot(t, err, label=name, color=color, linewidth=0.9)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|true EE pos - commanded pos| [mm]")
    ax.set_title("Physical tracking lag — how far the real arm falls behind the commanded target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "11_ik_tracking_error.png", dpi=130)
    plt.close(fig)


def plot_torque(baseline: dict, proposed: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True)
    for ax, (name, res) in zip(axes, [("baseline", baseline), ("proposed", proposed)]):
        log = res["log"]
        t = (_arr(log, "t_ns") - log[0]["t_ns"]) * 1e-9
        sat = _arr(log, "torque_saturated").astype(float)
        ax.fill_between(t, 0, sat, step="pre", color="tab:red" if name == "baseline" else "tab:blue", alpha=0.6)
        n_sat = int(sat.sum())
        ax.set_title(f"{name}: torque-saturated ticks = {n_sat}/{len(log)} ({100*n_sat/len(log):.1f}%)")
        ax.set_ylabel("saturated (0/1)")
        ax.set_ylim(-0.1, 1.1)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Joint torque saturation (any of 5 joints at |forcerange| limit = {TORQUE_LIMIT_NM}Nm)")
    fig.tight_layout()
    fig.savefig(out / "12_torque_saturation.png", dpi=130)
    plt.close(fig)


def plot_ablation_bars(sweep_csv: Path, out: Path) -> None:
    with open(sweep_csv) as f:
        rows = list(csv.DictReader(f))
    names = [r["config"] for r in rows]
    metrics = ["reversal_events", "max_backstep_m", "ik_tracking_err_rms_m", "torque_saturated_ticks"]
    titles = ["reversal events (#8)", "max backstep [m]", "IK tracking error RMS [m]", "torque-saturated ticks"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [float(r[metric]) for r in rows]
        colors = ["tab:red" if n == "all_baseline" else ("tab:blue" if "proposed" in n else "tab:gray") for n in names]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(title)
    fig.suptitle("Joint-dynamics plant — result metrics by module ablation")
    fig.tight_layout()
    fig.savefig(out / "13_joint_ablation_bars.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="docs/ablation_results")
    ap.add_argument("--sweep-csv", default="docs/ablation_results/joint_sweep.csv")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("running baseline...")
    baseline = run_joint_benchmark(BASELINE_CFG, duration=args.duration, seed=args.seed, verbose=False)
    print("running proposed...")
    proposed = run_joint_benchmark(PROPOSED_CFG, duration=args.duration, seed=args.seed, verbose=False)

    plot_progress(baseline, proposed, out)
    plot_ik_tracking_error(baseline, proposed, out)
    plot_torque(baseline, proposed, out)
    if Path(args.sweep_csv).exists():
        plot_ablation_bars(Path(args.sweep_csv), out)
    print(f"saved plots to {out}")

    import os
    import sys

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
