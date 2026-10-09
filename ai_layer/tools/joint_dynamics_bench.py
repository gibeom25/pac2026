#!/usr/bin/env python
"""MuJoCo 조인트 동역학 검증 — interface_benchmark.py의 point-mass "plant"를 실제 관절 구동
SO-101 팔(assets/so101/scene.xml = so101_new_calib.xml, 5관절 + position PD actuator + 토크
한계 ±3.35Nm)로 바꿔서, 같은 인터페이스 레이어(ChunkBuffer/AnchorResync/TriggerLogic/
OpenLoopCorrector)가 실제 관성·토크 제한이 있는 plant에서도 8번(진동) 문제를 그대로 막아주는지
검증한다 — "인터페이스 설계는 유효한데 point-mass라 로봇에도 통할지는 모른다"는 한계를
메우기 위한 2단계 검증.

핵심 재사용: point-mass 벤치마크의 `PointMassWorld`/`PointMassSnapshotSource`/
`PointMassChunkSink`는 plant가 뭔지 전혀 몰라도 되게 짜여 있었다(필드: cfg/buffer/polyline/
lock/snap_id/current_pos) — 그래서 **그대로 재사용**하고, 바뀌는 건 control-tick 루프뿐이다.
point-mass 버전은 `buffer.next_step()`이 내놓는 목표를 그대로 "진짜 위치"로 썼는데, 여기서는
그 목표를 (1) 이 MJCF 자체의 Jacobian IK로 관절각 목표로 바꾸고 (2) position actuator에 넣고
(3) mj_step으로 물리를 전진시킨 뒤 (4) 실제 도달한 EE 위치를 읽어 그걸 "진짜 위치"로 쓴다 —
그래서 토크/속도 한계를 못 버티면 그게 고스란히 진동/역행 지표에 반영된다.

한계 (정직하게 명시):
  - 회전은 point-mass 벤치마크와 동일하게 미제어 — IK는 초기 자세의 방향을 그대로
    유지하도록(위치만 3D로) 풀이한다. 실제 용접 궤적의 회전 성분은 검증 범위 밖.
  - IK는 이 MJCF 자체의 site Jacobian으로 풀어서(piper_sim.py의 감쇠최소제곱법과 동일
    방식) 외부 URDF와의 기구학 불일치가 없다 — 대신 ai_layer/kinematics.py가 쓰는 실로봇
    URDF와는 별개의 경로다(그쪽은 실로봇 연동용, 여긴 동역학 검증 전용).
  - position actuator의 PD 게인/토크 한계(so101_new_calib.xml 주석: "servo proportional
    gain 16 가정으로 역산")는 실측이 아니라 추정값이다 — 절대적인 포화 숫자가 아니라
    "같은 설정에서 baseline 대비 proposed가 포화/역행을 얼마나 줄이는지"의 상대 비교로만
    쓴다.
  - 작업 범위(seg_len)는 SO-101의 실제 가동범위에 맞춰 point-mass보다 훨씬 작게(기본 6cm)
    잡았다 — point-mass의 0.3m 세그먼트를 그대로 쓰면 IK가 발산하거나 관절 한계에 걸린다
    (실측 확인).

사용 예:
    PYTHONPATH=. python ai_layer/tools/joint_dynamics_bench.py --out /tmp/joint_proposed.csv
    PYTHONPATH=. python ai_layer/tools/joint_dynamics_bench.py --anchor none --no-corrector \
        --out /tmp/joint_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from ai_layer.control_bridge.ai_node import AiNode, BlankImageSource
from ai_layer.control_bridge.chunk_builder import EefMode
from ai_layer.control_bridge.protocol import AnchorMode
from ai_layer.rl.reward import _point_to_polyline
from ai_layer.tools.interface_benchmark import (
    AblationConfig,
    MockGroundTruthPredictor,
    PointMassChunkSink,
    PointMassSnapshotSource,
    PointMassWorld,
    bent_polyline,
)

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "so101"
MJCF_PATH = ASSETS_DIR / "scene.xml"
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_SITE_NAME = "gripperframe"
# 실측(이 세션): EE가 바닥(z=0)에서 12cm쯤 뜨고 베이스에서 23cm쯤 뻗은, 몸체와 안 부딪히는
# "준비 자세". bent_polyline이 이 자세의 EE 위치를 원점으로 그 주변에 그려진다.
Q_SEED_DEG = np.array([0.0, -55.0, 40.0, 60.0, 0.0])


class SO101Plant:
    """실제 관절(PD position actuator, 토크 한계) 동역학을 갖는 SO-101 팔. interface_benchmark의
    PointMass와 같은 역할(buffer가 요구하는 Cartesian 목표 -> "진짜 위치")을 하되, 중간에
    IK + mj_step 물리가 끼어서 즉시 도달하지 못할 수 있다."""

    def __init__(self, q_seed_deg: np.ndarray | None = None) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)  # IK 전용 스크래치 — mj_step으로 절대 전진 안 시킴
        self.site = self.model.site(EE_SITE_NAME).id
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES])
        self.dadr = np.array([self.model.joint(n).dofadr[0] for n in ARM_JOINT_NAMES])
        self.qrange = self.model.jnt_range[[self.model.joint(n).id for n in ARM_JOINT_NAMES]]
        self.act_adr = np.array([self.model.actuator(n).id for n in ARM_JOINT_NAMES])
        self.forcerange = self.model.actuator_forcerange[self.act_adr].copy()

        q0 = np.deg2rad(q_seed_deg if q_seed_deg is not None else Q_SEED_DEG)
        self.data.qpos[self.qadr] = q0
        self.data.ctrl[self.act_adr] = q0
        mujoco.mj_forward(self.model, self.data)
        self.home_pos = self._ee_pos(self.data)

    def _ee_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site].copy()

    def solve_ik_from_current(self, desired_pos: np.ndarray, iters: int = 8, tol: float = 1e-4) -> np.ndarray:
        """물리 관절각(data.qpos, 실제 도달한 값)을 출발점으로 목표 EE 위치까지 감쇠최소제곱
        IK(piper_sim.py와 동일 방식) — 스크래치 MjData에서만 풀고 물리 data는 절대 안 건드린다.
        반환값은 "이번 틱 목표 관절각"(제어기 setpoint)일 뿐, 실제로 거기 도달하는지는
        apply_and_step() 이후 true_ee_pos()로 확인해야 한다."""
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)  # mj_jacSite가 올바른 값을 내려면 필수(실측 확인 — 빠뜨리면 IK가 수mm~수cm 오차로 멈춤)
            err = desired_pos - self._ee_pos(d)
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site)
            J = jacp[:, self.dadr]
            lam2 = 1e-4
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), err)
            q = d.qpos[self.qadr] + np.clip(dq, -0.2, 0.2)
            d.qpos[self.qadr] = np.clip(q, self.qrange[:, 0], self.qrange[:, 1])
        return d.qpos[self.qadr].copy()

    def apply_and_step(self, q_target: np.ndarray, n_substeps: int = 1) -> None:
        self.data.ctrl[self.act_adr] = q_target
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

    def true_ee_pos(self) -> np.ndarray:
        return self._ee_pos(self.data)

    def torque_saturated(self, frac: float = 0.98) -> bool:
        f = np.abs(self.data.actuator_force[self.act_adr])
        return bool(np.any(f >= frac * self.forcerange[:, 1]))

    def actuator_force(self) -> np.ndarray:
        return self.data.actuator_force[self.act_adr].copy()


def joint_control_tick_loop(
    world: PointMassWorld, plant: SO101Plant, tick_dt_s: float, duration_s: float, stop: threading.Event,
    noise_std: float = 0.0005, seed: int = 0, ik_iters: int = 8,
) -> None:
    """interface_benchmark.control_tick_loop과 같은 역할이지만, buffer가 내놓는 목표(cmd_pos)를
    바로 "진짜 위치"로 쓰지 않고 IK+물리를 거쳐서 실제 도달한 위치(true_pos)를 쓴다. cmd_pos와
    true_pos의 차이(ik_tracking_err_m)가 "물리적으로 얼마나 못 따라가는지"의 직접적인 지표다."""
    w = world
    rng = np.random.default_rng(seed)
    n_substeps = max(1, round(tick_dt_s / plant.model.opt.timestep))
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
            cmd_pos, eef_bit, mode = w.buffer.next_step(now_ns)
            q_target = plant.solve_ik_from_current(cmd_pos, iters=ik_iters)
            plant.apply_and_step(q_target, n_substeps)
            true_pos = plant.true_ee_pos()
            ik_err = float(np.linalg.norm(true_pos - cmd_pos))
            saturated = plant.torque_saturated()

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
            w.current_pos = pos  # 다음 틱 Source/Sink가 보는 "현재 위치" — 물리 결과 그대로
            w.eef_active = active
            w.log.append({
                "t_ns": now_ns, "tick_dt_s": actual_dt,
                "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
                "true_x": true_pos[0], "true_y": true_pos[1], "true_z": true_pos[2],
                "cmd_x": cmd_pos[0], "cmd_y": cmd_pos[1], "cmd_z": cmd_pos[2],
                "ik_tracking_err_m": ik_err, "torque_saturated": saturated,
                "dist_to_line": dist_to_line, "progress": progress, "mode": mode.name,
                "eef_active": active, "trigger_toggles_so_far": toggles,
            })
        time.sleep(max(0.0, tick_dt_s - (time.monotonic() - now)))


def run_joint_benchmark(
    cfg: AblationConfig, *, duration: float = 6.0, tick_dt_ms: float = 2.0,
    corner_deg: float = 90.0, seg_len: float = 0.06, speed: float = 0.01, seed: int = 0,
    noise_std: float = 0.0005, ik_iters: int = 8, verbose: bool = True,
) -> dict:
    """joint_dynamics_bench 버전 run_benchmark — interface_benchmark.run_benchmark와 같은 반환
    형태(지표 dict + log)를 쓰므로 make_plots.py류 재사용이 쉽다."""
    plant = SO101Plant()
    polyline = bent_polyline(corner_deg, seg_len) + plant.home_pos  # 로봇이 닿는 범위에 경로 배치
    world = PointMassWorld(polyline=polyline, cfg=cfg)
    world.current_pos = plant.home_pos.copy()
    world.corrector.reset(world.current_pos.copy())
    world.buffer.reset(world.current_pos.copy())  # IDLE 기본 위치를 원점이 아니라 실제 팔 시작 위치로

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
        try:
            while not stop.is_set():
                out = node.step()
                if out is None:
                    time.sleep(0.01)
        finally:
            node.close()

    ai_thread = threading.Thread(target=_ai_loop, daemon=True)
    ctrl_thread = threading.Thread(
        target=joint_control_tick_loop,
        args=(world, plant, tick_dt_ms / 1000.0, duration, stop, noise_std, seed, ik_iters),
    )
    if verbose:
        print(f"[joint-bench] anchor={cfg.anchor_level} buffer_baseline={cfg.buffer_baseline} "
              f"trigger_baseline={cfg.trigger_baseline} corrector={cfg.use_corrector} "
              f"corner_deg={corner_deg} seg_len={seg_len}")
    ai_thread.start()
    ctrl_thread.start()
    ctrl_thread.join()
    stop.set()
    ai_thread.join(timeout=2.0)

    log = world.log
    if not log:
        return {"ticks": 0}
    tick_dts = np.array([r["tick_dt_s"] for r in log][1:])
    dists = np.array([r["dist_to_line"] for r in log])
    progresses = np.array([r["progress"] for r in log])
    pos_err = np.array([
        [r["pos_x"] - r["true_x"], r["pos_y"] - r["true_y"], r["pos_z"] - r["true_z"]] for r in log
    ])
    ik_err = np.array([r["ik_tracking_err_m"] for r in log])
    saturated_ticks = sum(1 for r in log if r["torque_saturated"])
    draining = sum(1 for r in log if r["mode"] == "DRAINING")
    holding = sum(1 for r in log if r["mode"] == "HOLDING")
    backward = np.diff(progresses) < -0.001  # interface_benchmark.py와 동일 임계값(노이즈 대비 1mm)
    n_reversals = int(np.sum(np.diff(backward.astype(int)) == 1))
    max_backstep = float(-np.diff(progresses)[backward].min()) if backward.any() else 0.0
    metrics = {
        "ticks": len(log),
        "tick_jitter_std_ms": float(tick_dts.std() * 1e3),
        "tick_jitter_max_ms": float(tick_dts.max() * 1e3),
        "dist_rms_m": float(np.sqrt((dists**2).mean())),
        "dist_max_m": float(dists.max()),
        "pos_error_rms_m": float(np.sqrt((pos_err**2).sum(axis=1)).mean()),
        "ik_tracking_err_rms_m": float(np.sqrt((ik_err**2).mean())),  # 물리(관절 동역학)가 못 따라간 정도
        "ik_tracking_err_max_m": float(ik_err.max()),
        "torque_saturated_ticks": saturated_ticks,  # 토크 한계(±3.35Nm)에 닿은 틱 수
        "trigger_toggles": int(log[-1]["trigger_toggles_so_far"]),
        "reversal_events": n_reversals,
        "max_backstep_m": max_backstep,
        "draining_ticks": draining,
        "holding_ticks": holding,
    }
    if verbose:
        print(f"[joint-bench] ticks={metrics['ticks']} tick_jitter_std={metrics['tick_jitter_std_ms']:.3f}ms "
              f"tick_jitter_max={metrics['tick_jitter_max_ms']:.3f}ms")
        print(f"[joint-bench] ik_tracking_err rms={metrics['ik_tracking_err_rms_m']*1e3:.3f}mm "
              f"max={metrics['ik_tracking_err_max_m']*1e3:.3f}mm torque_saturated_ticks={saturated_ticks}")
        print(f"[joint-bench] 진동(8번): reversal_events={metrics['reversal_events']} "
              f"max_backstep={metrics['max_backstep_m']*1e3:.3f}mm")
        print(f"[joint-bench] trigger_toggles={metrics['trigger_toggles']} DRAINING={draining} HOLDING={holding}")
    return {**metrics, "log": log, "chunk_log": world.chunk_log}


def main() -> None:
    ap = argparse.ArgumentParser(description="SO-101 관절 동역학 위에서 AI<->제어 인터페이스 벤치마크")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--tick-dt-ms", type=float, default=2.0)
    ap.add_argument("--corner-deg", type=float, default=90.0)
    ap.add_argument("--seg-len", type=float, default=0.06)
    ap.add_argument("--speed", type=float, default=0.01)
    ap.add_argument("--anchor", choices=["none", "commit_only", "commit_refine"], default="commit_refine")
    ap.add_argument("--buffer-baseline", action="store_true")
    ap.add_argument("--trigger-baseline", action="store_true")
    ap.add_argument("--no-corrector", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-std", type=float, default=0.0005)
    ap.add_argument("--ik-iters", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = AblationConfig(
        anchor_level=args.anchor, buffer_baseline=args.buffer_baseline,
        trigger_baseline=args.trigger_baseline, use_corrector=not args.no_corrector,
    )
    result = run_joint_benchmark(
        cfg, duration=args.duration, tick_dt_ms=args.tick_dt_ms, corner_deg=args.corner_deg,
        seg_len=args.seg_len, speed=args.speed, seed=args.seed, noise_std=args.noise_std,
        ik_iters=args.ik_iters,
    )
    if args.out and result.get("log"):
        log = result["log"]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
        print(f"[joint-bench] saved {args.out}")


if __name__ == "__main__":
    main()
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
