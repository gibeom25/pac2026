#!/usr/bin/env python
"""docs/Experiment_Plan.md "결과" 섹션용 그래프 생성 — baseline/proposed를 직접 돌려서(CSV 재로드
없이, raw_path 같은 복합 필드를 그대로 메모리에서 씀) 한 번에 전부 그린다.

실행: PYTHONPATH=. python ai2ctrl_layer/make_plots.py --out-dir docs/ablation_results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai2ctrl_layer.interface_benchmark import AblationConfig, bent_polyline, run_benchmark

BASELINE_CFG = AblationConfig(anchor_level="none", buffer_baseline=True, trigger_baseline=True, use_corrector=False)
PROPOSED_CFG = AblationConfig(anchor_level="commit_refine", buffer_baseline=False, trigger_baseline=False, use_corrector=True)


def _arr(log: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in log])


def plot_input(baseline: dict, proposed: dict, out: Path) -> None:
    """입력 그래프: AI chunk 도착 시각/추론 지연 — Problem.md 핵심 전제(추론 주기 불규칙)를 실측으로."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    for ax, (name, res) in zip(axes, [("baseline", baseline), ("proposed", proposed)]):
        clog = res["chunk_log"]
        t0 = clog[0]["t_arrival_ns"]
        t = np.array([(c["t_arrival_ns"] - t0) * 1e-9 for c in clog])
        delay = np.array([c["infer_latency_s"] * 1e3 for c in clog])
        ax.stem(t, delay, basefmt=" ")
        ax.axhline(42.5, color="gray", linestyle="--", linewidth=1, label="measured p50 (42.5ms)")
        ax.set_ylabel("inference latency [ms]")
        ax.set_title(f"Input (AI chunk arrivals) — {name}: mean interval {np.mean(np.diff(t))*1e3:.1f}ms, mean delay {delay.mean():.1f}ms")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out / "01_input_chunk_timing.png", dpi=130)
    plt.close(fig)


def plot_output(baseline: dict, proposed: dict, out: Path) -> None:
    """출력 그래프: 실행된 progress(t) — baseline의 역행(8번)과 proposed의 깨끗한 단조증가 대비."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, res, color in [("baseline", baseline, "tab:red"), ("proposed", proposed, "tab:blue")]:
        log = res["log"]
        t = (_arr(log, "t_ns") - log[0]["t_ns"]) * 1e-9
        prog = _arr(log, "progress")
        ax.plot(t, prog, label=name, color=color, linewidth=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("path progress (arc length) [m]")
    ax.set_title("Output trajectory — progress vs time (backward jumps = problem #8 oscillation)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "02_output_progress.png", dpi=130)
    plt.close(fig)


def plot_axis_pos_vel(baseline: dict, proposed: dict, out: Path) -> None:
    """차원별(X/Y/Z) 위치 + 속도 — baseline vs proposed."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for col, axis in enumerate(["x", "y", "z"]):
        for name, res, color in [("baseline", baseline, "tab:red"), ("proposed", proposed, "tab:blue")]:
            log = res["log"]
            t = (_arr(log, "t_ns") - log[0]["t_ns"]) * 1e-9
            pos = _arr(log, f"pos_{axis}")
            vel = np.gradient(pos, t)
            axes[0, col].plot(t, pos, label=name, color=color, linewidth=0.8)
            axes[1, col].plot(t, vel, label=name, color=color, linewidth=0.8)
        axes[0, col].set_title(f"Position {axis.upper()}(t)")
        axes[1, col].set_title(f"Velocity {axis.upper()}(t)")
        axes[1, col].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("position [m]")
    axes[1, 0].set_ylabel("velocity [m/s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per-axis (X/Y/Z) position & velocity — baseline vs proposed")
    fig.tight_layout()
    fig.savefig(out / "03_axis_pos_vel.png", dpi=130)
    plt.close(fig)


def plot_3d_trajectory(baseline: dict, proposed: dict, polyline: np.ndarray, out: Path) -> None:
    """3D 궤적: REF(참조 경로) + 출력(실제 실행) + 입력(각 chunk가 resync 전 제안했던 경로) 겹쳐그림."""
    fig = plt.figure(figsize=(13, 6))
    for i, (name, res) in enumerate([("baseline", baseline), ("proposed", proposed)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot(polyline[:, 0], polyline[:, 1], polyline[:, 2], color="black", linewidth=2, label="REF (reference path)")
        log = res["log"]
        ax.plot(_arr(log, "pos_x"), _arr(log, "pos_y"), _arr(log, "pos_z"), color="tab:blue", linewidth=1.2, label="output (executed trajectory)")
        for j, c in enumerate(res["chunk_log"]):
            xs, ys, zs = c["raw_path_x"], c["raw_path_y"], c["raw_path_z"]
            ax.plot(xs, ys, zs, color="tab:orange", linewidth=0.5, alpha=0.5, label="input (chunk proposed path)" if j == 0 else None)
        ax.set_title(name)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        # 2026-10-09: z축을 auto-scale에 맡겼더니 센서노이즈(표준편차 0.5mm) 수준으로 범위가
        # 잡혀서, 실제로는 미세한 노이즈일 뿐인데 거대한 지그재그 기둥처럼 보이는 착시가
        # 생겼다(실측 확인 — 데이터 자체는 정상, 그래프만 오해를 일으킴) — z축 범위를 x/y
        # 범위 대비 적당히 작은 고정폭(±2cm)으로 맞춰서 "이 작업은 거의 평면 위에서 일어난다"는
        # 사실이 그대로 보이게 한다.
        ax.set_zlim(-0.02, 0.02)
        ax.legend(fontsize=7)
    fig.suptitle("3D trajectory — REF vs output vs input (each chunk's pre-resync proposed path)")
    fig.tight_layout()
    fig.savefig(out / "04_trajectory_3d.png", dpi=130)
    plt.close(fig)


def plot_ablation_bars(sweep_csv: Path, out: Path) -> None:
    """결과 지표 그래프: ablation sweep 요약을 막대그래프로."""
    with open(sweep_csv) as f:
        rows = list(csv.DictReader(f))
    module_rows = [r for r in rows if not r["config"].startswith("corner_deg")]
    names = [r["config"] for r in module_rows]
    metrics = ["reversal_events", "max_backstep_m", "pos_error_rms_m", "trigger_toggles"]
    titles = ["reversal events (#8)", "max backstep [m]", "position error RMS [m]", "trigger chattering count"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [float(r[metric]) for r in module_rows]
        colors = ["tab:red" if n == "all_baseline" else ("tab:blue" if "proposed" in n else "tab:gray") for n in names]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(title)
    fig.suptitle("Result metrics — per-module ablation comparison")
    fig.tight_layout()
    fig.savefig(out / "05_ablation_bars.png", dpi=130)
    plt.close(fig)

    # 일반화 스윕(코너 각도 / 노이즈 / 속도 / 경로 길이) — "다양한 곳/조건"에서 proposed가
    # 그대로 버티는지(reversal_events=0) + corrector가 노이즈/속도에 맞춰 반응하는지(pos_error_rms)
    sweeps = [
        ("corner_deg", "corner angle [deg]", "corner difficulty", "06_corner_sweep.png"),
        ("noise_std", "sensor noise std [m]", "sensor noise level", "07_noise_sweep.png"),
        ("speed", "tracking speed [m/s]", "tracking speed", "08_speed_sweep.png"),
        ("seg_len", "segment length [m]", "weld seam length", "09_seglen_sweep.png"),
    ]
    for prefix, xlabel, label, fname in sweeps:
        srows = [r for r in rows if r["config"].startswith(prefix + "=")]
        if not srows:
            continue
        xs = [float(r["config"].split("=")[1]) for r in srows]
        rev = [float(r["reversal_events"]) for r in srows]
        err = [float(r["pos_error_rms_m"]) for r in srows]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(xs, rev, "o-", color="tab:blue")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("reversal events")
        ax1.set_title(f"Reversal events vs {label}")
        ax1.set_ylim(-0.5, max(5, max(rev) + 1))

        ax2.plot(xs, err, "o-", color="tab:orange")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("position error RMS [m]")
        ax2.set_title(f"Position error vs {label}")

        fig.suptitle(f"Generalization sweep — {label} (proposed, all configs)")
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", default="docs/ablation_results")
    ap.add_argument("--sweep-csv", default="docs/ablation_results/sweep.csv")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    polyline = bent_polyline()
    print("running baseline...")
    baseline = run_benchmark(BASELINE_CFG, duration=args.duration, seed=args.seed, verbose=False)
    print("running proposed...")
    proposed = run_benchmark(PROPOSED_CFG, duration=args.duration, seed=args.seed, verbose=False)

    plot_input(baseline, proposed, out)
    plot_output(baseline, proposed, out)
    plot_axis_pos_vel(baseline, proposed, out)
    plot_3d_trajectory(baseline, proposed, polyline, out)
    if Path(args.sweep_csv).exists():
        plot_ablation_bars(Path(args.sweep_csv), out)
    print(f"saved plots to {out}")

    import os
    import sys

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
