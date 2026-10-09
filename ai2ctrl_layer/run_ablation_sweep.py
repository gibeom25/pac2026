#!/usr/bin/env python
"""전체 ablation + 시나리오 스윕 — docs/Experiment_Plan.md "모듈 ablation 매트릭스" 구현.

1) 모듈별 독립 토글(한 번에 하나씩만 baseline으로 내려서 순수 기여도 측정)
2) AnchorResync 3단계 비교(none / commit_only / commit_refine)
3) 코너 각도 스윕(완만~급격)으로 난이도별 일반화 확인

실행:
    PYTHONPATH=. python ai2ctrl_layer/run_ablation_sweep.py --duration 8 --out docs/ablation_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

from ai2ctrl_layer.interface_benchmark import AblationConfig, run_benchmark

PROPOSED = dict(anchor_level="commit_refine", buffer_baseline=False, trigger_baseline=False, use_corrector=True)


def _run(name: str, overrides: dict, **kwargs) -> dict:
    cfg = AblationConfig(**{**PROPOSED, **overrides})
    print(f"\n=== {name} ===")
    r = run_benchmark(cfg, verbose=True, **kwargs)
    r.pop("log", None)
    r["config"] = name
    return r


def module_ablation(duration: float, seed: int) -> list[dict]:
    """각 모듈을 하나씩만 baseline으로 내려서(나머지는 proposed) 순수 기여도 측정."""
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


def scenario_sweep(duration: float, seed: int) -> list[dict]:
    """코너 각도(완만~급격)별로 proposed 설정이 얼마나 일반화되는지."""
    rows = []
    for corner_deg in [30.0, 60.0, 90.0, 120.0, 150.0]:
        rows.append(_run(f"corner_deg={corner_deg}", {}, duration=duration, seed=seed, corner_deg=corner_deg))
    return rows


def noise_sweep(duration: float, seed: int) -> list[dict]:
    """센서 노이즈 수준(0~4mm 표준편차)별로 proposed가 얼마나 버티는지 — corrector(KF)가
    노이즈 커질수록 더 중요해진다는 걸 pos_error_rms로 보여주는 게 목적."""
    rows = []
    for noise_std in [0.0, 0.0005, 0.001, 0.002, 0.004]:
        rows.append(_run(f"noise_std={noise_std}", {}, duration=duration, seed=seed, noise_std=noise_std))
    return rows


def speed_sweep(duration: float, seed: int) -> list[dict]:
    """추종 속도(느림~빠름)별로 proposed가 얼마나 일반화되는지 — 빠를수록 chunk 경계 간극(2/8번
    문제)이 더 커질 것이므로 anchor_resync/corrector의 기여가 속도에 비례하는지 확인."""
    rows = []
    for speed in [0.01, 0.02, 0.04, 0.08, 0.16]:
        rows.append(_run(f"speed={speed}", {}, duration=duration, seed=seed, speed=speed))
    return rows


def seglen_sweep(duration: float, seed: int) -> list[dict]:
    """경로 세그먼트 길이(작업 범위 크기)별로 proposed가 얼마나 일반화되는지 — "다양한 곳"에서
    용접하는 상황(짧은 이음매 ~ 긴 이음매)을 흉내."""
    rows = []
    for seg_len in [0.1, 0.3, 0.5, 1.0, 2.0]:
        rows.append(_run(f"seg_len={seg_len}", {}, duration=duration, seed=seed, seg_len=seg_len))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=8.0, help="설정당 실행 시간 [s]")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="결과 CSV 저장 경로")
    args = ap.parse_args()

    rows = (
        module_ablation(args.duration, args.seed)
        + scenario_sweep(args.duration, args.seed)
        + noise_sweep(args.duration, args.seed)
        + speed_sweep(args.duration, args.seed)
        + seglen_sweep(args.duration, args.seed)
    )

    print("\n\n" + "=" * 130)
    print(f"{'config':<32} {'ticks':>6} {'jit_std_ms':>10} {'jit_max_ms':>10} {'reversal':>8} {'backstep_m':>10} "
          f"{'pos_err_m':>10} {'trig_tog':>8} {'drain':>6} {'hold':>5}")
    for r in rows:
        print(f"{r['config']:<32} {r['ticks']:>6} {r['tick_jitter_std_ms']:>10.3f} {r['tick_jitter_max_ms']:>10.3f} "
              f"{r['reversal_events']:>8} {r['max_backstep_m']:>10.5f} {r['pos_error_rms_m']:>10.5f} "
              f"{r['trigger_toggles']:>8} {r['draining_ticks']:>6} {r['holding_ticks']:>5}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[sweep] saved {args.out}")

    sys.stdout.flush()
    sys.stderr.flush()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
