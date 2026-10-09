#!/usr/bin/env python
"""joint_dynamics_bench.py 버전 ablation 스윕 — run_ablation_sweep.py(point-mass)와 똑같은
8개 설정(all_baseline/all_proposed/모듈별 단독 baseline/AnchorResync 3단계)을 실제 관절
동역학 plant 위에서 재측정한다. point-mass 결과와 나란히 비교하기 위해 설정 이름을 그대로
맞췄다.

실행:
    PYTHONPATH=. python ai_layer/tools/run_joint_ablation_sweep.py --duration 4 \
        --out docs/ablation_results/joint_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

from ai_layer.tools.joint_dynamics_bench import AblationConfig, run_joint_benchmark

PROPOSED = dict(anchor_level="commit_refine", buffer_baseline=False, trigger_baseline=False, use_corrector=True)


def _run(name: str, overrides: dict, **kwargs) -> dict:
    cfg = AblationConfig(**{**PROPOSED, **overrides})
    print(f"\n=== {name} ===")
    r = run_joint_benchmark(cfg, verbose=True, **kwargs)
    r.pop("log", None)
    r.pop("chunk_log", None)
    r["config"] = name
    return r


def module_ablation(duration: float, seed: int) -> list[dict]:
    rows = []
    rows.append(_run("all_baseline", dict(anchor_level="none", buffer_baseline=True, trigger_baseline=True, use_corrector=False), duration=duration, seed=seed))
    rows.append(_run("all_proposed", {}, duration=duration, seed=seed))
    rows.append(_run("only_buffer_baseline", dict(buffer_baseline=True), duration=duration, seed=seed))
    rows.append(_run("only_trigger_baseline", dict(trigger_baseline=True), duration=duration, seed=seed))
    rows.append(_run("only_corrector_off", dict(use_corrector=False), duration=duration, seed=seed))
    rows.append(_run("anchor_none", dict(anchor_level="none"), duration=duration, seed=seed))
    rows.append(_run("anchor_commit_only", dict(anchor_level="commit_only"), duration=duration, seed=seed))
    rows.append(_run("anchor_commit_refine(=proposed)", dict(anchor_level="commit_refine"), duration=duration, seed=seed))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = module_ablation(args.duration, args.seed)

    print("\n\n" + "=" * 150)
    print(f"{'config':<32} {'ticks':>6} {'jit_std_ms':>10} {'reversal':>8} {'backstep_mm':>11} "
          f"{'ik_err_rms_mm':>13} {'ik_err_max_mm':>13} {'torque_sat':>10} {'trig_tog':>8}")
    for r in rows:
        print(f"{r['config']:<32} {r['ticks']:>6} {r['tick_jitter_std_ms']:>10.3f} {r['reversal_events']:>8} "
              f"{r['max_backstep_m']*1e3:>11.3f} {r['ik_tracking_err_rms_m']*1e3:>13.3f} "
              f"{r['ik_tracking_err_max_m']*1e3:>13.3f} {r['torque_saturated_ticks']:>10} {r['trigger_toggles']:>8}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[joint-sweep] saved {args.out}")

    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
