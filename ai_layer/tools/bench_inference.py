"""추론 지연 벤치마크. 어느 PC 에서든 같은 방법으로 잰다 (5090 등 시연 PC 비교용).

잰다: seam 특징(CPU) / 전처리 / ACT forward(청크 32) / 후처리 / ActionChunk 인코딩 / 끝-끝(ai_node.step).
체크포인트가 없으면 --synthetic 로 가짜 데이터 체크포인트를 임시 학습해서 잰다 (모델 크기가 같아 시간은 실제와 동일).

실행:
  cd pac2026-team
  PYTHONPATH=. python ai_layer/tools/bench_inference.py --checkpoint outputs/bc_act/last
  PYTHONPATH=. python ai_layer/tools/bench_inference.py --synthetic          # 체크포인트 없을 때
옵션: --device cuda|cpu  --n 200  --json out.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ai_layer.bc_inference import load_bc_checkpoint
from ai_layer.configs.so101_act_bc import ACTION_DIM, CHUNK_SIZE, DT_AI_SEC, IMAGE_KEY
from ai_layer.control_bridge.ai_node import ACTPredictor, AiNode, BlankImageSource, DryRunSink, FakeSnapshotSource
from ai_layer.control_bridge.chunk_builder import build_action_chunk
from ai_layer.control_bridge.protocol import encode_chunk
from ai_layer.perception.seam_cv import SeamGrooveDetector
from ai_layer.perception.seam_features import seam_features_from_rgb


def make_synthetic_checkpoint() -> str:
    """가짜 데이터로 6스텝만 학습한 체크포인트 (bc_synthetic_test 재사용)."""
    d = Path(tempfile.mkdtemp(prefix="bench_ckpt_"))
    subprocess.run([sys.executable, "ai_layer/tools/bc_synthetic_test.py", "--keep-dir", str(d)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(d / "out" / "last")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if not args.checkpoint:
        if not args.synthetic:
            print("--checkpoint 또는 --synthetic 필요"); return 1
        print("[bench] 가짜 데이터 체크포인트 만드는 중 (1~2분)...")
        args.checkpoint = make_synthetic_checkpoint()

    dev = args.device
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "-"
    print(f"=== 환경: {platform.node()} | {platform.processor() or platform.machine()} | GPU {gpu} | torch {torch.__version__} ===")
    print(f"청크 {CHUNK_SIZE}걸음 × {DT_AI_SEC*1000:.2f} ms = {CHUNK_SIZE*DT_AI_SEC*1000:.0f} ms 분량, 행동 {ACTION_DIM}차원")

    pol, pre, post = load_bc_checkpoint(args.checkpoint, device=dev)
    img = np.full((240, 320, 3), 230, np.uint8)
    cv2.line(img, (40, 20), (240, 210), (20, 20, 20), 4)
    det = SeamGrooveDetector()
    state = np.array([0.25, 0, 0.10, 1, 0, 0, 0, 1, 0], np.float32)

    def sync():
        if dev == "cuda":
            torch.cuda.synchronize()

    def timeit(fn, n=args.n, warm=20):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(n):
            sync(); t = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t) * 1e3)
        return np.array(ts)

    results = {}

    def row(name, a):
        results[name] = {"p50": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
        print(f"  {name:30s} p50 {np.median(a):7.2f} ms   p95 {np.percentile(a, 95):7.2f} ms   max {a.max():7.2f} ms")

    print(f"\n=== 단계별 ({args.n}회, 워밍업 20회 제외) ===")
    seam = seam_features_from_rgb(det, img)
    row("① seam 특징 (CPU)", timeit(lambda: seam_features_from_rgb(det, img)))
    obs = {IMAGE_KEY: torch.from_numpy(np.transpose(img.astype(np.float32) / 255, (2, 0, 1))),
           "observation.state": torch.from_numpy(state), "observation.environment_state": torch.from_numpy(seam)}
    row("② 전처리(정규화+device)", timeit(lambda: pre(dict(obs))))
    batch = pre(dict(obs))
    with torch.no_grad():
        row("③ ACT forward (청크 예측)", timeit(lambda: pol.predict_action_chunk(batch)))
        chunk = pol.predict_action_chunk(batch)
    flat = chunk.reshape(-1, ACTION_DIM)
    row("④ 후처리(비정규화+CPU)", timeit(lambda: post(flat)))
    mc = post(flat).reshape(CHUNK_SIZE, ACTION_DIM).numpy()
    row("⑤ ActionChunk 인코딩", timeit(lambda: encode_chunk(build_action_chunk(mc, seq_id=1, snap_id=1, t_obs_ns=0))))

    print(f"\n=== 끝-끝 ai_node.step ({args.n}회) ===")
    node = AiNode(ACTPredictor(args.checkpoint, dev), FakeSnapshotSource(), BlankImageSource(), DryRunSink(), verbose=False)
    node.images.img = img
    t = time.perf_counter(); node.step(); first = (time.perf_counter() - t) * 1e3
    a = timeit(node.step)
    row("전체 step", a)
    print(f"  첫 추론(워밍업) {first:.0f} ms | 제어 max_age 300 ms 대비 여유 {300 - np.percentile(a, 95):.0f} ms | 33 ms 주기 시 점유 {np.median(a) / 33.33 * 100:.0f}%")
    if args.json:
        Path(args.json).write_text(json.dumps({"host": platform.node(), "gpu": gpu, "torch": torch.__version__, "first_ms": first, **results}, indent=2, ensure_ascii=False))
        print(f"  -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
