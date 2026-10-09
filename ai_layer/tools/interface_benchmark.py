#!/usr/bin/env python
"""AI↔제어 인터페이스 벤치마크 — docs/Experiment_Plan.md 구현.

로봇/MuJoCo 없이 점(point-mass)이 꺾인 직선(polyline)을 따라가는 걸로 AI↔제어 인터페이스의
효과를 측정한다. 핵심 설계 원칙: ai_layer/control_bridge/ai_node.py의 AiNode는 **한 글자도
안 고친다** — 실제 배포 코드가 쓰는 ActionChunk/StateSnapshot/ChunkSink/SnapshotSource/
ChunkPredictor 프로토콜을 그대로 통과시키고, PointMass 쪽(이 파일)만 새로 만든다. 그래서 여기서
만든 chunk_buffer.py/anchor_resync.py/trigger_logic.py/open_loop_corrector.py는 나중에 진짜
제어 계층에 옮길 때 점-물리 부분만 떼어내면 된다.

사용 예:
    PYTHONPATH=. python ai_layer/tools/interface_benchmark.py --out /tmp/bench_proposed.csv
    PYTHONPATH=. python ai_layer/tools/interface_benchmark.py --anchor none --buffer-baseline --trigger-baseline \
        --no-corrector --out /tmp/bench_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ai_layer.control_bridge.ai_node import (
    OBS_ENV_STATE,
    OBS_STATE,
    AiNode,
    BlankImageSource,
    ChunkPredictor,
    ChunkSink,
    SnapshotSource,
)
from ai_layer.control_bridge.chunk_builder import EefMode
from ai_layer.control_bridge.protocol import (
    AnchorMode,
    CommitTrajectory,
    ControlMode,
    StateSnapshot,
    decode_chunk,
)
from ai_layer.configs.so101_act_bc import IMAGE_KEY, SEAM_FEATURE_DIM
from ai_layer.rl.reward import _point_to_polyline
from ai_layer.control_bridge.anchor_resync import resync_start_index
from ai_layer.control_bridge.chunk_buffer import ChunkBuffer
from ai_layer.control_bridge.open_loop_corrector import ConstantVelocityKF
from ai_layer.control_bridge.trigger_logic import TriggerLogic


# --------------------------------------------------------------------------- 참조 경로
def bent_polyline(corner_deg: float = 90.0, seg_len: float = 0.3, n_points: int = 200) -> np.ndarray:
    """원점에서 +x로 seg_len, 거기서 corner_deg만큼 꺾어 다시 seg_len — 2세그먼트 꺾인 직선을
    n_points개로 촘촘히 재샘플(anchor_resync가 쓰는 호 길이 투영의 분해능을 위해)."""
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = p0 + np.array([seg_len, 0.0, 0.0])
    theta = np.deg2rad(180.0 - corner_deg)
    p2 = p1 + seg_len * np.array([np.cos(theta), np.sin(theta), 0.0])
    half = n_points // 2
    seg1 = np.linspace(p0, p1, half, endpoint=False)
    seg2 = np.linspace(p1, p2, n_points - half)
    return np.concatenate([seg1, seg2], axis=0).astype(np.float64)


def _progress_at(polyline: np.ndarray, pos: np.ndarray) -> float:
    pt = torch.from_numpy(pos).float().unsqueeze(0)
    poly = torch.from_numpy(polyline).float().unsqueeze(0)
    _, progress, _ = _point_to_polyline(pt, poly)
    return float(progress.item())


def _point_at_progress(polyline: np.ndarray, progress: float) -> np.ndarray:
    seg = np.diff(polyline, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    progress = float(np.clip(progress, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, progress, side="right") - 1)
    i = min(max(i, 0), len(seg) - 1)
    t = (progress - cum[i]) / max(seg_len[i], 1e-9)
    return polyline[i] + t * seg[i]


# --------------------------------------------------------------------------- Mock AI
def real_measured_delay_sampler(rng: np.random.Generator) -> float:
    """이번 세션에 ai_node.py로 실측한 ACT 추론시간 분포(p50≈42.5ms, p95≈466.5ms)에 맞춘
    lognormal 근사. 가정한 숫자가 아니라 실측값 기반 — Problem.md/Experiment_Plan.md 참고."""
    mu = np.log(0.0425)
    sigma = np.log(466.5 / 42.5) / 1.645
    return float(np.clip(rng.lognormal(mu, sigma), 0.001, 2.0))


class MockGroundTruthPredictor(ChunkPredictor):
    """ChunkPredictor 인터페이스 그대로 구현 — 나중에 ACTPredictor로 교체해도 AiNode/다른 코드는
    무수정. 실제 vision 모델 대신 참조 경로(polyline)를 알고 있는 "정답 컨트롤러"가 매 호출마다
    추론 지연을 샘플링해 실제로 sleep한 뒤 chunk를 만든다 — AiNode의 infer_ms 측정이 그대로
    실측값이 되고, 8번 문제(관측-적용 시점 불일치)가 자연스럽게 재현된다."""

    def __init__(
        self, polyline: np.ndarray, chunk_size: int = 32, dt_s: float = 1 / 30,
        speed: float = 0.02, delay_sampler=None, seed: int = 0,
    ) -> None:
        self.polyline = polyline
        self.chunk_size = chunk_size
        self.dt_s = dt_s
        self.speed = speed
        self.delay_sampler = delay_sampler or real_measured_delay_sampler
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        delay_s = self.delay_sampler(self.rng)
        time.sleep(delay_s)  # 추론 지연 실제로 재현 (여기서 걸리는 시간이 AiNode의 infer_ms로 측정됨)

        pos0 = np.asarray(obs[OBS_STATE][:3], dtype=float)
        progress0 = _progress_at(self.polyline, pos0)
        step_dist = self.speed * self.dt_s

        chunk = np.zeros((self.chunk_size, 7), dtype=np.float32)
        pos = pos0.copy()
        progress = progress0
        for i in range(self.chunk_size):
            progress += step_dist
            target = _point_at_progress(self.polyline, progress)
            chunk[i, :3] = target - pos
            pos = target
            chunk[i, 6] = 1.0  # on-line 추종이므로 트리거 요청은 항상 ON (실시간 재평가는 TriggerLogic 몫)
        return chunk


# --------------------------------------------------------------------------- PointMass 스탠드인
PRIV_SEAM_FEATURES = np.zeros(SEAM_FEATURE_DIM, dtype=np.float32)  # Mock에선 CV 특징 불필요


@dataclass
class AblationConfig:
    anchor_level: str = "commit_refine"  # "none" | "commit_only" | "commit_refine"
    buffer_baseline: bool = False  # True면 ChunkBuffer를 "즉시 정지"로 강등(safe_zone=1.0, max_drain≈0)
    trigger_baseline: bool = False  # True면 히스테리시스 없이 단일 임계값
    use_corrector: bool = True  # False면 OpenLoopCorrector(KF) 미사용 — 순수 feedforward


@dataclass
class PointMassWorld:
    """공유 상태 — PointMassSnapshotSource/PointMassChunkSink/control-tick 루프가 같이 본다."""

    polyline: np.ndarray
    cfg: AblationConfig
    buffer: ChunkBuffer = field(default_factory=ChunkBuffer)
    corrector: ConstantVelocityKF = field(default_factory=ConstantVelocityKF)
    trigger: TriggerLogic | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    snap_id: int = 0
    current_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    eef_active: bool = False
    log: list[dict] = field(default_factory=list)
    chunk_log: list[dict] = field(default_factory=list)  # "입력" 그래프용 — chunk 도착 시각/지연

    def __post_init__(self) -> None:
        if self.cfg.buffer_baseline:
            self.buffer.safe_zone_frac = 1.0
            self.buffer.max_drain_ns = 1_000_000.0  # 사실상 즉시 HOLD
        on_tol = off_tol = 0.002 if self.cfg.trigger_baseline else None
        self.trigger = TriggerLogic() if on_tol is None else TriggerLogic(on_tol=on_tol, off_tol=off_tol)
        self.corrector.reset(self.current_pos.copy())
        self.buffer.reset(self.current_pos.copy())  # IDLE 기본 위치를 실제 시작 위치로(chunk_buffer.py 참고)


class PointMassSnapshotSource(SnapshotSource):
    """snap_id마다 "AI가 실제로 anchor로 쓸 두 후보 위치"(measured=OBS_POSE용,
    commit_end=COMMIT_END용)를 기록해둔다 — PointMassChunkSink가 chunk.anchor_mode/snap_id로
    "AI가 그 chunk를 만들 때 가정했던 위치"를 정확히 복원하기 위해(8번 문제 재현에 필수:
    여기서 실제 현재 위치를 쓰면 문제 자체가 재현이 안 됨, 실측 확인된 버그였음)."""

    def __init__(self, world: PointMassWorld, history_size: int = 256) -> None:
        self.world = world
        self.history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._history_size = history_size

    def latest(self) -> StateSnapshot:
        w = self.world
        with w.lock:
            w.snap_id += 1
            now = time.monotonic_ns()
            pos = w.current_pos.copy()
            commit = w.buffer.remaining_commit(now)
            mode = w.buffer.mode
            snap_id = w.snap_id
        commit_end_pos = commit.poses[-1, :3].copy() if commit.n > 0 else pos.copy()
        self.history[snap_id] = (pos.copy(), commit_end_pos)
        if len(self.history) > self._history_size:
            self.history.pop(min(self.history))
        pose7 = np.concatenate([pos, [0.0, 0.0, 0.0, 1.0]])
        return StateSnapshot(snap_id=snap_id, t_meas_ns=now, measured=pose7, commit=commit, mode=mode)


class PointMassChunkSink(ChunkSink):
    def __init__(self, world: PointMassWorld, snapshot_source: PointMassSnapshotSource) -> None:
        self.world = world
        self.snapshot_source = snapshot_source

    def send(self, payload: bytes) -> None:
        w = self.world
        chunk = decode_chunk(payload)
        now = time.monotonic_ns()
        measured_pos, commit_end_pos = self.snapshot_source.history.get(
            chunk.snap_id, (w.current_pos.copy(), w.current_pos.copy())
        )
        ai_assumed_pos = commit_end_pos if chunk.anchor_mode == AnchorMode.COMMIT_END else measured_pos
        # "입력 궤적"(3D 플롯용): resync 전, AI가 chunk를 만들 때 가정한 위치를 기준으로 그
        # chunk가 그대로 제안하는 절대 경로 — resync_start_index가 실제로 뭘 고쳐주는지 눈으로
        # 보여주려는 것(출력/참조 궤적과 겹쳐 그림).
        raw_cum = ai_assumed_pos[None, :] + np.cumsum(np.asarray(chunk.steps[:, :3], dtype=float), axis=0)
        w.chunk_log.append({
            "t_arrival_ns": now, "seq_id": chunk.seq_id, "snap_id": chunk.snap_id,
            "t_obs_ns": chunk.t_obs_ns, "anchor_mode": chunk.anchor_mode.name,
            "infer_latency_s": (now - chunk.t_obs_ns) * 1e-9,  # "입력" 그래프: AI가 실제로 걸린 시간
            "raw_path_x": raw_cum[:, 0].tolist(), "raw_path_y": raw_cum[:, 1].tolist(), "raw_path_z": raw_cum[:, 2].tolist(),
        })
        with w.lock:
            if w.cfg.anchor_level == "commit_refine":
                actual_pos = w.buffer.peek_pos(now) if w.buffer.n > 0 else w.current_pos.copy()
                start_idx, new_progress, anchor_pos = resync_start_index(
                    chunk, actual_pos, w.polyline, w.buffer.last_progress,
                )
            else:  # "none" / "commit_only" — AI가 chunk를 만들 때 가정했던 위치를 그대로 신뢰
                start_idx, anchor_pos = 0, ai_assumed_pos
                new_progress = _progress_at(w.polyline, ai_assumed_pos)
            w.buffer.load(chunk, start_idx, anchor_pos, now)
            w.buffer.last_progress = new_progress


def control_tick_loop(
    world: PointMassWorld, tick_dt_s: float, duration_s: float, stop: threading.Event,
    noise_std: float = 0.0005, seed: int = 0,
) -> None:
    """실제 wall-clock 기준 tick_dt_s 간격으로 ChunkBuffer를 소비 — 이 루프의 실제 간격 자체가
    "제어 틱 지터" 측정값이다 (일반 Linux/Python이라 RTOS 보장은 아님, Experiment_Plan.md 참고).

    noise_std: ChunkBuffer가 내놓는 "진짜" 위치에 센서 노이즈를 흉내 내서 더한 걸 "측정값"으로
    쓴다 — 이게 없으면 OpenLoopCorrector(KF)가 거를 노이즈 자체가 없어서 아무 효과도 안 보인다
    (실측 확인: 처음엔 노이즈를 아예 안 넣어서 corrector on/off 지표가 구분이 안 됐음).
    trigger 판단(dist_to_line)도 이 노이즈 섞인 값으로 해야 TriggerLogic의 히스테리시스
    효과(채터링 억제)도 눈에 보인다."""
    w = world
    rng = np.random.default_rng(seed)
    t_end = time.monotonic() + duration_s
    last_tick = time.monotonic()
    prev_active: bool | None = None
    toggles = 0
    while time.monotonic() < t_end and not stop.is_set():
        now_ns = time.monotonic_ns()
        now = time.monotonic()
        actual_dt = now - last_tick
        last_tick = now
        with w.lock:
            true_pos, eef_bit, mode = w.buffer.next_step(now_ns)
            measured_pos = true_pos + rng.normal(0.0, noise_std, 3)
            if w.cfg.use_corrector:
                w.corrector.predict(int(tick_dt_s * 1e9))
                pos = w.corrector.update(measured_pos)
            else:
                pos = measured_pos
            dist_t, progress_t, _ = _point_to_polyline(
                torch.from_numpy(pos).float().unsqueeze(0), torch.from_numpy(w.polyline).float().unsqueeze(0),
            )
            dist_to_line = float(dist_t.item())
            progress = float(progress_t.item())
            active = w.trigger.decide(eef_bit, dist_to_line)
            if prev_active is not None and active != prev_active:
                toggles += 1
            prev_active = active
            w.current_pos = pos
            w.eef_active = active
            w.log.append({
                "t_ns": now_ns, "tick_dt_s": actual_dt, "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
                "true_x": true_pos[0], "true_y": true_pos[1], "true_z": true_pos[2],
                "dist_to_line": dist_to_line, "progress": progress, "mode": mode.name,
                "eef_active": active, "trigger_toggles_so_far": toggles,
            })
        time.sleep(max(0.0, tick_dt_s - (time.monotonic() - now)))


def run_benchmark(
    cfg: AblationConfig, *, duration: float = 15.0, tick_dt_ms: float = 2.0,
    corner_deg: float = 90.0, seg_len: float = 0.3, speed: float = 0.02, seed: int = 0,
    noise_std: float = 0.0005, verbose: bool = True,
) -> dict:
    """한 설정으로 한 번 돌려서 지표 dict + 원본 log를 반환 — CLI(main)와 sweep 스크립트가 공유.

    주의: ai_thread는 daemon이라 이 함수가 리턴해도 백그라운드에서 계속 돈다(짧은 sleep 중일 수
    있음) — 여러 번 연달아 호출해도 프로세스 차원에서는 문제없다(스레드가 쌓였다가 실제 프로세스
    종료 시 한꺼번에 정리됨). 다만 PyTorch + daemon 스레드 조합이 **인터프리터 종료** 시 가끔
    core dump를 내는 걸 실측했으므로(그 자체는 결과에 영향 없음), 전체 스윕이 다 끝난 뒤
    호출부가 마지막에 한 번만 os._exit(0)로 깔끔히 종료할 것(main()이 그렇게 함)."""
    polyline = bent_polyline(corner_deg, seg_len)
    world = PointMassWorld(polyline=polyline, cfg=cfg)

    predictor = MockGroundTruthPredictor(polyline, speed=speed, seed=seed)
    snapshots = PointMassSnapshotSource(world)
    sink = PointMassChunkSink(world, snapshots)
    node = AiNode(
        predictor, snapshots, BlankImageSource(), sink,
        anchor_prefer=AnchorMode.OBS_POSE if cfg.anchor_level == "none" else AnchorMode.COMMIT_END,
        eef_mode=EefMode.FROM_CHANNEL, verbose=False,
    )

    stop = threading.Event()

    def _ai_loop() -> None:
        """AiNode.run(iterations=0)은 무한루프라 멈출 방법이 없다 — 실측 확인된 문제(스윕에서
        여러 번 연달아 돌리면 이전 실행의 AI 스레드가 안 죽고 계속 CPU를 먹어서 뒤로 갈수록
        지터가 누적돼 늘어났다). node.step()을 직접 불러서 stop을 보게 한다."""
        try:
            while not stop.is_set():
                out = node.step()
                if out is None:
                    time.sleep(0.01)
        finally:
            node.close()

    ai_thread = threading.Thread(target=_ai_loop, daemon=True)
    ctrl_thread = threading.Thread(
        target=control_tick_loop, args=(world, tick_dt_ms / 1000.0, duration, stop, noise_std, seed),
    )
    if verbose:
        print(f"[bench] anchor={cfg.anchor_level} buffer_baseline={cfg.buffer_baseline} "
              f"trigger_baseline={cfg.trigger_baseline} corrector={cfg.use_corrector} "
              f"corner_deg={corner_deg}")
    ai_thread.start()
    ctrl_thread.start()
    ctrl_thread.join()
    stop.set()
    ai_thread.join(timeout=2.0)  # step() 중간일 수 있으니 짧게 대기 — 다음 run의 CPU 경합을 막는 핵심

    log = world.log
    if not log:
        return {"ticks": 0}
    tick_dts = np.array([r["tick_dt_s"] for r in log][1:])
    dists = np.array([r["dist_to_line"] for r in log])
    progresses = np.array([r["progress"] for r in log])
    pos_err = np.array([
        [r["pos_x"] - r["true_x"], r["pos_y"] - r["true_y"], r["pos_z"] - r["true_z"]] for r in log
    ])
    draining = sum(1 for r in log if r["mode"] == "DRAINING")
    holding = sum(1 for r in log if r["mode"] == "HOLDING")
    # 2026-10-09: noise_std(센서 노이즈) 도입 후 1e-6 임계값으로 재실행했더니 전부 ~1000회로
    # 뭉개졌다 — 노이즈가 progress 계산에도 섞여서 수백 μm 수준의 미세 역행을 전부 "역행"으로
    # 잡은 것(실측 확인). 진짜 앵커링 버그(baseline에서 최대 18mm 역행)와 노이즈 지터(proposed도
    # 0.66mm 정도는 남음)를 구분하려고, 노이즈 수준보다 뚜렷하게 큰 1mm를 임계값으로 쓴다.
    backward = np.diff(progresses) < -0.001
    n_reversals = int(np.sum(np.diff(backward.astype(int)) == 1))
    max_backstep = float(-np.diff(progresses)[backward].min()) if backward.any() else 0.0
    metrics = {
        "ticks": len(log),
        "tick_jitter_std_ms": float(tick_dts.std() * 1e3),
        "tick_jitter_max_ms": float(tick_dts.max() * 1e3),
        "dist_rms_m": float(np.sqrt((dists**2).mean())),
        "dist_max_m": float(dists.max()),
        "pos_error_rms_m": float(np.sqrt((pos_err**2).sum(axis=1)).mean()),  # OpenLoopCorrector 효과(노이즈 대비 추정 오차)
        "trigger_toggles": int(log[-1]["trigger_toggles_so_far"]),  # TriggerLogic 채터링 지표
        "reversal_events": n_reversals,
        "max_backstep_m": max_backstep,
        "draining_ticks": draining,
        "holding_ticks": holding,
    }
    if verbose:
        print(f"[bench] ticks={metrics['ticks']} tick_jitter_std={metrics['tick_jitter_std_ms']:.3f}ms "
              f"tick_jitter_max={metrics['tick_jitter_max_ms']:.3f}ms")
        print(f"[bench] dist_to_line rms={metrics['dist_rms_m']:.5f}m max={metrics['dist_max_m']:.5f}m "
              f"pos_error_rms(vs true)={metrics['pos_error_rms_m']:.5f}m")
        print(f"[bench] 진동(8번): reversal_events={metrics['reversal_events']} max_backstep={metrics['max_backstep_m']:.5f}m")
        print(f"[bench] trigger_toggles={metrics['trigger_toggles']} DRAINING ticks={draining} HOLDING ticks={holding}")
    return {**metrics, "log": log, "chunk_log": world.chunk_log}


def main() -> None:
    ap = argparse.ArgumentParser(description="AI↔제어 인터페이스 point-mass 벤치마크")
    ap.add_argument("--duration", type=float, default=15.0, help="실행 시간 [s]")
    ap.add_argument("--tick-dt-ms", type=float, default=2.0, help="제어 틱 간격 목표 [ms]")
    ap.add_argument("--corner-deg", type=float, default=90.0)
    ap.add_argument("--seg-len", type=float, default=0.3)
    ap.add_argument("--speed", type=float, default=0.02, help="추종 목표 속도 [m/s]")
    ap.add_argument("--anchor", choices=["none", "commit_only", "commit_refine"], default="commit_refine")
    ap.add_argument("--buffer-baseline", action="store_true")
    ap.add_argument("--trigger-baseline", action="store_true")
    ap.add_argument("--no-corrector", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-std", type=float, default=0.0005, help="센서 노이즈 표준편차 [m] — corrector 효과를 보려면 0보다 커야 함")
    ap.add_argument("--out", default=None, help="CSV 저장 경로 (선택)")
    args = ap.parse_args()

    cfg = AblationConfig(
        anchor_level=args.anchor, buffer_baseline=args.buffer_baseline,
        trigger_baseline=args.trigger_baseline, use_corrector=not args.no_corrector,
    )
    result = run_benchmark(
        cfg, duration=args.duration, tick_dt_ms=args.tick_dt_ms, corner_deg=args.corner_deg,
        seg_len=args.seg_len, speed=args.speed, seed=args.seed, noise_std=args.noise_std,
    )
    if args.out and result.get("log"):
        log = result["log"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
        print(f"[bench] saved {args.out}")
        chunk_log = result.get("chunk_log") or []
        if chunk_log:
            chunk_out = args.out.rsplit(".", 1)[0] + "_chunks.csv"
            with open(chunk_out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(chunk_log[0].keys()))
                w.writeheader()
                w.writerows(chunk_log)
            print(f"[bench] saved {chunk_out}")


if __name__ == "__main__":
    main()
    # ai_thread는 daemon이라 node.run()이 여전히 time.sleep() 안에서 돌고 있을 수 있다 —
    # 결과는 이미 다 출력/저장됐으니, 인터프리터의 정상 종료 경로(스레드+torch 조합에서 가끔
    # core dump를 내는 것 실측 확인) 대신 즉시 종료한다. os._exit()는 버퍼를 안 비우므로
    # 먼저 flush 필수(실측 확인 — 안 하면 위 print가 전부 사라짐).
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
