#!/usr/bin/env python
"""AI 추론 노드: 스냅샷 수신 -> 이미지 -> ACT 추론 -> ActionChunk 송신.

송지수 제어 계층(tools/run_live.py --ai external)과 이렇게 연결된다.
    AI  --PUB-->  ipc:///tmp/pac_action    (ActionChunk, 3177 byte 고정)   제어가 SUB(connect)
    제어 --PUB--> ipc:///tmp/pac_snapshot  (StateSnapshot, 3695 byte 고정) AI 가 SUB(connect)
둘 다 ZeroMQ PUB/SUB + CONFLATE ("최신 것 하나만"). AI 쪽이 action 을 bind 한다 (fake_ai_node 와 동일).

한 번에 추론 하나만 돌린다 (control/README "튜닝하면서 배운 것 3"). 추론이 끝나면 곧바로 다음 스냅샷으로.

실행 예
    # 배관 점검 (모델·로봇·카메라 없이): 0 청크를 만들어 인코딩까지만
    PYTHONPATH=. python ai_layer/control_bridge/ai_node.py --dry-run --fake-snapshot --iterations 3
    # 체크포인트 + 가짜 스냅샷 + 빈 이미지 (모델 경로 점검)
    PYTHONPATH=. python ai_layer/control_bridge/ai_node.py --checkpoint outputs/bc_act/last --dry-run --fake-snapshot
    # 실제 연결 (제어 계층이 --ai external 로 떠 있을 때)
    PYTHONPATH=. python ai_layer/control_bridge/ai_node.py --checkpoint outputs/bc_act/last --camera realsense

7번째 채널(gripper) -> eef 매핑은 --eef-mode 로 고른다 (회의 전 기본 off = 펌프 OFF).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.configs.so101_act_bc import CHUNK_SIZE, DT_AI_SEC, IMAGE_KEY  # noqa: E402
from ai_layer.control_bridge.chunk_builder import (  # noqa: E402
    BuildStats,
    ChunkLimits,
    EefMode,
    build_action_chunk,
)
from ai_layer.control_bridge.protocol import (  # noqa: E402
    AnchorMode,
    CommitTrajectory,
    ControlMode,
    StateSnapshot,
    decode_snapshot,
    encode_chunk,
)
from ai_layer.control_bridge.snapshot_adapter import choose_anchor  # noqa: E402
from ai_layer.perception.seam_cv import SeamGrooveDetector  # noqa: E402
from ai_layer.perception.seam_features import seam_features_from_rgb  # noqa: E402

OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
NS = 1e-9


# ----------------------------------------------------------------------------- 이미지 소스
class ImageSource:
    """get() -> (rgb uint8 (H,W,3), t_capture_ns). 닫을 땐 close()."""

    def get(self) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class BlankImageSource(ImageSource):
    def __init__(self, h: int = 240, w: int = 320) -> None:
        self.img = np.full((h, w, 3), 230, dtype=np.uint8)

    def get(self) -> tuple[np.ndarray, int]:
        return self.img.copy(), time.monotonic_ns()


class RealSenseImageSource(ImageSource):
    """Intel RealSense (D405 손목 카메라) 컬러 프레임. 미검증 — 실카메라로 처음 쓸 때 확인할 것.

    t_capture_ns: RealSense 프레임 타임스탬프는 자체 시계라서, 여기서는 도착 시각(monotonic)을 쓴다.
    카메라 촬영 <-> pose 측정 시각 오프셋 캘리브레이션은 후속 작업 (control/README 남은 작업).
    """

    def __init__(self, w: int = 320, h: int = 240, fps: int = 30, serial: str | None = None) -> None:
        import pyrealsense2 as rs  # 지연 import: 카메라 없는 환경에서도 모듈은 로드되게

        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        self.pipe.start(cfg)

    def get(self) -> tuple[np.ndarray, int]:
        frames = self.pipe.wait_for_frames()
        t = time.monotonic_ns()
        color = frames.get_color_frame()
        return np.asanyarray(color.get_data()).copy(), t

    def close(self) -> None:
        self.pipe.stop()


# ----------------------------------------------------------------------------- 스냅샷 소스
class SnapshotSource:
    def latest(self) -> StateSnapshot | None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeSnapshotSource(SnapshotSource):
    """제어 계층 없이 배관 점검용. 측정 pose 고정, commit 은 비움 (OBS_POSE 로 강등됨)."""

    def __init__(self, pose7=(0.25, 0.0, 0.10, 0.0, 0.0, 0.0, 1.0)) -> None:
        self._id = 0
        self.pose7 = np.asarray(pose7, dtype=float)

    def latest(self) -> StateSnapshot:
        self._id += 1
        return StateSnapshot(
            snap_id=self._id, t_meas_ns=time.monotonic_ns(), measured=self.pose7.copy(),
            commit=CommitTrajectory(), mode=ControlMode.TRACKING,
        )


class ZmqSnapshotSource(SnapshotSource):
    def __init__(self, endpoint: str) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.RCVTIMEO, 100)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)
        self._latest: StateSnapshot | None = None
        self._lock = threading.Lock()
        self._run = threading.Event()
        self._run.set()
        self._th = threading.Thread(target=self._loop, daemon=True, name="snapshot-rx")
        self._th.start()

    def _loop(self) -> None:
        import zmq

        while self._run.is_set():
            try:
                payload = self._sock.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            try:
                snap = decode_snapshot(payload)
            except ValueError:
                continue
            with self._lock:
                self._latest = snap

    def latest(self) -> StateSnapshot | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._run.clear()
        self._th.join(timeout=1.0)
        self._sock.close(0)


# ----------------------------------------------------------------------------- 청크 송신
class ChunkSink:
    def send(self, payload: bytes) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DryRunSink(ChunkSink):
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)


class ZmqChunkSink(ChunkSink):
    def __init__(self, endpoint: str, bind: bool = True) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(endpoint) if bind else self._sock.connect(endpoint)
        time.sleep(0.2)  # PUB slow-joiner

    def send(self, payload: bytes) -> None:
        self._sock.send(payload, copy=False)

    def close(self) -> None:
        self._sock.close(0)


# ----------------------------------------------------------------------------- 정책 (모델 or 0)
class ChunkPredictor:
    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """obs -> (T, 7)"""
        raise NotImplementedError


class ZeroPredictor(ChunkPredictor):
    """모델 없이 배관 점검: 제자리 청크 (증분 0, gripper 0)."""

    def __init__(self, n_steps: int = CHUNK_SIZE) -> None:
        self.n = n_steps

    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros((self.n, 7), dtype=np.float32)


class ACTPredictor(ChunkPredictor):
    def __init__(self, ckpt_dir: str, device: str) -> None:
        from ai_layer.bc_inference import load_bc_checkpoint, predict_chunk_np

        self.policy, self.pre, self.post = load_bc_checkpoint(ckpt_dir, device=device)
        self._predict = predict_chunk_np

    def predict(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self._predict(self.policy, self.pre, self.post, obs)


# ----------------------------------------------------------------------------- 노드
class AiNode:
    def __init__(
        self,
        predictor: ChunkPredictor,
        snapshots: SnapshotSource,
        images: ImageSource,
        sink: ChunkSink,
        *,
        anchor_prefer: AnchorMode = AnchorMode.COMMIT_END,
        policy_id: int = 1,
        dt_ns: int = int(round(DT_AI_SEC * 1e9)),
        eef_mode: EefMode = EefMode.OFF,
        eef_threshold: float = 50.0,
        n_steps: int | None = None,
        limits: ChunkLimits | None = None,
        verbose: bool = True,
    ) -> None:
        self.predictor = predictor
        self.snapshots = snapshots
        self.images = images
        self.sink = sink
        self.anchor_prefer = anchor_prefer
        self.policy_id = policy_id
        self.dt_ns = dt_ns
        self.eef_mode = eef_mode
        self.eef_threshold = eef_threshold
        self.n_steps = n_steps
        self.limits = limits or ChunkLimits()
        self.verbose = verbose
        self.seam = SeamGrooveDetector()
        self.seq_id = 0
        self.stats = BuildStats()
        self.infer_ms_hist: list[float] = []

    def step(self) -> bytes | None:
        """스냅샷 하나 -> 청크 하나. 스냅샷이 없으면 None."""
        snap = self.snapshots.latest()
        if snap is None:
            return None
        rgb, t_obs_ns = self.images.get()
        choice = choose_anchor(snap, self.anchor_prefer)

        t0 = time.monotonic()
        seam = seam_features_from_rgb(self.seam, rgb)
        obs = {
            IMAGE_KEY: np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)),  # (3,H,W) [0,1]
            OBS_STATE: choice.state9,
            OBS_ENV_STATE: seam,
        }
        model_chunk = self.predictor.predict(obs)
        infer_ms = (time.monotonic() - t0) * 1e3
        self.infer_ms_hist.append(infer_ms)

        self.seq_id += 1
        chunk = build_action_chunk(
            model_chunk,
            seq_id=self.seq_id,
            snap_id=choice.snap_id,
            t_obs_ns=t_obs_ns,
            anchor_mode=choice.anchor_mode,
            policy_id=self.policy_id,
            dt_ns=self.dt_ns,
            eef_mode=self.eef_mode,
            eef_threshold=self.eef_threshold,
            limits=self.limits,
            n_steps=self.n_steps,
            stats=self.stats,
        )
        payload = encode_chunk(chunk)
        self.sink.send(payload)
        if self.verbose:
            print(
                f"[ai-node] seq={chunk.seq_id} snap={chunk.snap_id} anchor={chunk.anchor_mode.name} "
                f"n={chunk.n_steps} infer={infer_ms:.1f}ms |dp|max={np.linalg.norm(chunk.steps[:, :3], axis=1).max():.4f} "
                f"Σdp={np.round(chunk.steps[:, :3].sum(axis=0), 3).tolist()} "
                f"eef={int(chunk.eef.sum())}/{chunk.n_steps} clipped(pos/rot)={self.stats.pos_clipped_steps}/{self.stats.rot_clipped_steps}"
            )
        return payload

    def run(self, iterations: int = 0, min_period_s: float = 0.0) -> None:
        i = 0
        try:
            while iterations <= 0 or i < iterations:
                t0 = time.monotonic()
                out = self.step()
                if out is None:
                    time.sleep(0.01)
                    continue
                i += 1
                rest = min_period_s - (time.monotonic() - t0)
                if rest > 0:
                    time.sleep(rest)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        self.snapshots.close()
        self.images.close()
        self.sink.close()
        if self.infer_ms_hist:
            a = np.array(self.infer_ms_hist)
            print(f"[ai-node] done. chunks={self.stats.chunks} infer p50={np.median(a):.1f}ms p95={np.percentile(a, 95):.1f}ms "
                  f"max_step_seen pos={self.stats.max_pos_step_seen:.4f}m rot={self.stats.max_rot_step_seen:.4f}rad")


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="train_bc 체크포인트 폴더. 없으면 0 청크(배관 점검)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--action-endpoint", default="ipc:///tmp/pac_action")
    ap.add_argument("--snapshot-endpoint", default="ipc:///tmp/pac_snapshot")
    ap.add_argument("--dry-run", action="store_true", help="ZeroMQ 로 보내지 않고 인코딩까지만")
    ap.add_argument("--fake-snapshot", action="store_true", help="제어 계층 없이 가짜 스냅샷 사용")
    ap.add_argument("--camera", choices=["blank", "realsense"], default="blank")
    ap.add_argument("--anchor", choices=["obs", "commit"], default="commit")
    ap.add_argument("--policy-id", type=int, default=1, help="chunk_policy.json 의 policy_id (1 = seam-welding)")
    ap.add_argument("--eef-mode", choices=[m.value for m in EefMode], default=EefMode.OFF.value)
    ap.add_argument("--eef-threshold", type=float, default=50.0)
    ap.add_argument("--n-steps", type=int, default=None, help="청크 앞 n 스텝만 송신 (기본 전체 32)")
    ap.add_argument("--iterations", type=int, default=0, help="0 이면 Ctrl-C 까지")
    ap.add_argument("--min-period", type=float, default=0.0, help="청크 사이 최소 간격 [s]")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    predictor = ACTPredictor(args.checkpoint, args.device) if args.checkpoint else ZeroPredictor()
    snapshots = FakeSnapshotSource() if args.fake_snapshot else ZmqSnapshotSource(args.snapshot_endpoint)
    images = RealSenseImageSource() if args.camera == "realsense" else BlankImageSource()
    sink = DryRunSink() if args.dry_run else ZmqChunkSink(args.action_endpoint, bind=True)

    node = AiNode(
        predictor, snapshots, images, sink,
        anchor_prefer=AnchorMode.COMMIT_END if args.anchor == "commit" else AnchorMode.OBS_POSE,
        policy_id=args.policy_id, eef_mode=EefMode(args.eef_mode), eef_threshold=args.eef_threshold,
        n_steps=args.n_steps, verbose=not args.quiet,
    )
    print(f"[ai-node] predictor={'ACT:' + args.checkpoint if args.checkpoint else 'zero'} "
          f"snapshots={'fake' if args.fake_snapshot else args.snapshot_endpoint} "
          f"sink={'dry-run' if args.dry_run else args.action_endpoint} eef_mode={args.eef_mode}")
    node.run(iterations=args.iterations, min_period_s=args.min_period)


if __name__ == "__main__":
    main()
