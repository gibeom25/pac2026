#!/usr/bin/env python
"""joint_dynamics_bench.py의 baseline/proposed를 MuJoCo 오프스크린 렌더로 영상(mp4) + 스냅샷(png)
으로 남긴다 — 그래프 숫자 말고 실제로 팔이 어떻게 움직이는지(baseline은 떨고 proposed는
매끄러운지) 눈으로 비교하기 위함.

2026-10-10(2차, 기범 피드백 — "deploy 시간이 너무 짧다 / 선 길이도 늘려야 / 바닥에 그림을
그려야") 반영:
  - 기본 길이를 5s -> 26s로(경로를 끝까지 다 그릴 만큼), seg_len을 0.06 -> 0.12m로 늘렸다
    (2세그먼트 총 0.24m — 이 팔의 가동범위에서 IK가 안정적으로 수렴하는 한계 근처까지
    실측 확인: seg_len=0.15+corner=120°에서부터 IK가 깨짐).
  - 경로를 바닥 근처(z≈0.05m, 기존 조인트 동역학 벤치마크의 z≈0.13m보다 훨씬 낮은 별도
    seed 자세)에 배치하고, 트리거가 켜진 동안 실제 도달 위치(true_pos, 명령값이 아니라
    물리 결과)에서 비드가 자유낙하해 바닥에 쌓이는 걸 그대로 그린다 — "바닥에 그림을
    그린다"는 요청을 record_mujoco.py에 이미 있는 비드 낙하 시각화(BeadDrop/
    _draw_bead_trail)를 그대로 재사용해서 구현(새로 만들지 않음).
  - SO101Plant에 q_seed_deg 파라미터를 추가해서(joint_dynamics_bench.py 쪽 수정) 이
    "바닥용" seed는 여기서만 쓰고, 이미 측정/보고한 조인트 동역학 수치(Experiment_Plan.md
    표)는 기존 seed 그대로 영향 없음.

joint_dynamics_bench.py/run_joint_benchmark()를 그대로 쓰지 않고 따로 둔 이유: 렌더 호출 자체가
수 ms~수십 ms 걸려서 2ms 틱 예산을 자주 넘기는데, 그 틱에서 물리를 "고정 n_substeps"만큼만
전진시키면 물리 시간이 실제 경과 시간보다 뒤처지는 왜곡이 생긴다(렌더링 때문에 로봇이 더 뒤처진
것처럼 보이는 가짜 효과). 그래서 여기서는 매 틱 "실제로 지난 시간"만큼 n_substeps를 다시
계산해서 물리가 항상 실제 wall-clock을 따라잡게 한다 — AI 추론 지연(time.sleep 기반, 실측
분포)과 틱 타이밍은 joint_dynamics_bench.py와 동일(실시간 스레드 2개: AI 추론 스레드 + 제어
틱 스레드).

실행:
    PYTHONPATH=. python ai2ctrl_layer/render_joint_sim.py --duration 26 --seed 1
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from ai_layer.control_bridge.ai_node import AiNode, BlankImageSource
from ai_layer.control_bridge.chunk_builder import EefMode
from ai_layer.control_bridge.protocol import AnchorMode
from ai_layer.rl.reward import _point_to_polyline
from ai_layer.tools.record_mujoco import BEAD_RGBA, BeadDrop, _draw_bead_trail
from ai2ctrl_layer.interface_benchmark import (
    AblationConfig,
    MockGroundTruthPredictor,
    PointMassChunkSink,
    PointMassSnapshotSource,
    PointMassWorld,
    bent_polyline,
)
from ai2ctrl_layer.joint_dynamics_bench import SO101Plant

OUT_DIR = Path("docs/ablation_results")
CAMERA_HW = (480, 640)
POLYLINE_RGBA = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)  # REF 경로(바닥 체커 위에서 잘 보이는 빨강)
TARGET_RGBA = np.array([1.0, 0.55, 0.0, 0.95], dtype=np.float32)  # 이번 틱 chunk 목표(주황)
BEAD_MIN_SPACING_M = 0.0015  # 이 거리 이상 움직였을 때만 새 비드 — 너무 촘촘히 찍어 geom이 넘치는 것 방지

# "바닥에 그림을 그린다" 요청 전용 seed — EE가 바닥(z=0) 근처(약 4.8cm)에서 시작하게.
# joint_dynamics_bench.py의 기본 Q_SEED_DEG(EE z≈13cm)와 분리되어 있어 기존 조인트 동역학
# 벤치마크 수치(Experiment_Plan.md)에는 영향 없음.
FLOOR_SEED_DEG = np.array([0.0, -22.0, 60.0, 0.0, 0.0])

BASELINE_CFG = AblationConfig(anchor_level="none", buffer_baseline=True, trigger_baseline=True, use_corrector=False)
PROPOSED_CFG = AblationConfig(anchor_level="commit_refine", buffer_baseline=False, trigger_baseline=False, use_corrector=True)


def _make_camera(home_pos: np.ndarray) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = home_pos + np.array([0.08, 0.0, 0.0])  # 경로가 +x로 뻗어나가므로 중심을 살짝 앞으로
    cam.distance = 0.62  # 길어진 경로(0.24m) 전체가 들어오게 point-mass용(0.5)보다 멀리
    cam.azimuth = 120.0
    cam.elevation = -35.0
    return cam


def _draw_markers(scene: mujoco.MjvScene, polyline: np.ndarray, cmd_pos: np.ndarray) -> None:
    """REF 경로(빨강, 듬성듬성) + 이번 틱 chunk가 요구한 목표(주황) — point-mass 3D 궤적
    그래프(04_trajectory_3d.png)와 같은 색 관례(REF=검정 계열, chunk 제안=주황)."""
    mat = np.eye(3).flatten()
    for p in polyline[::4]:
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.004, 0, 0]),
                             pos=p, mat=mat, rgba=POLYLINE_RGBA)
        scene.ngeom += 1
    if scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.008, 0, 0]),
                             pos=cmd_pos, mat=mat, rgba=TARGET_RGBA)
        scene.ngeom += 1


def record_joint_sim(
    cfg: AblationConfig, name: str, *, duration: float = 26.0, tick_dt_ms: float = 2.0,
    corner_deg: float = 90.0, seg_len: float = 0.12, speed: float = 0.01, seed: int = 1,
    noise_std: float = 0.0005, ik_iters: int = 8, fps: int = 24, out_dir: Path = OUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plant = SO101Plant(q_seed_deg=FLOOR_SEED_DEG)
    renderer = mujoco.Renderer(plant.model, height=CAMERA_HW[0], width=CAMERA_HW[1], max_geom=4000)
    cam = _make_camera(plant.home_pos)

    polyline = bent_polyline(corner_deg, seg_len) + plant.home_pos
    world = PointMassWorld(polyline=polyline, cfg=cfg)
    world.current_pos = plant.home_pos.copy()
    world.corrector.reset(world.current_pos.copy())
    world.buffer.reset(world.current_pos.copy())

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

    tick_dt_s = tick_dt_ms / 1000.0
    rng = np.random.default_rng(seed)
    trigger = world.trigger
    poly_t = torch.from_numpy(polyline).float().unsqueeze(0)
    frames: list[np.ndarray] = []
    bead_points: list[BeadDrop] = []
    last_bead_pos: np.ndarray | None = None
    prev_active: bool | None = None
    snap_fracs = [0.0, 1 / 3, 2 / 3, 0.97]
    snap_idx = 0
    frame_interval_s = 1.0 / fps
    next_frame_t = 0.0
    cmd_pos = plant.home_pos.copy()

    ai_thread.start()
    t_start = time.monotonic()
    last_phys_t = t_start
    while True:
        now = time.monotonic()
        sim_t = now - t_start
        if sim_t >= duration:
            break
        now_ns = time.monotonic_ns()

        with world.lock:
            cmd_pos, eef_bit, mode = world.buffer.next_step(now_ns)
            q_target = plant.solve_ik_from_current(cmd_pos, iters=ik_iters)
            # 렌더 때문에 이번 루프가 평소보다 오래 걸렸어도, 물리는 "실제로 지난 시간"만큼
            # 전진시킨다(고정 n_substeps를 쓰면 렌더 오버헤드가 그대로 "로봇이 더 못 따라간
            # 것"처럼 왜곡된다 — 모듈 docstring 참고).
            phys_now = time.monotonic()
            elapsed = phys_now - last_phys_t
            last_phys_t = phys_now
            n_substeps = max(1, round(elapsed / plant.model.opt.timestep))
            plant.apply_and_step(q_target, n_substeps)
            true_pos = plant.true_ee_pos()

            measured_pos = true_pos + rng.normal(0.0, noise_std, 3)
            if cfg.use_corrector:
                world.corrector.predict(int(tick_dt_s * 1e9))
                pos = world.corrector.update(measured_pos)
            else:
                pos = measured_pos
            dist_t, _, _ = _point_to_polyline(torch.from_numpy(pos).float().unsqueeze(0), poly_t)
            active = trigger.decide(eef_bit, float(dist_t.item()))
            world.current_pos = pos
            world.eef_active = active

        # 비드(바닥에 그리는 선): 트리거가 켜진 동안, 실제 도달 위치(true_pos — 명령이 아니라
        # 물리 결과)에서 일정 간격 이상 움직였을 때만 새로 찍는다(너무 촘촘히 찍으면 geom 초과).
        if active and (last_bead_pos is None or np.linalg.norm(true_pos - last_bead_pos) >= BEAD_MIN_SPACING_M):
            bead_points.append(BeadDrop(true_pos))
            last_bead_pos = true_pos.copy()
        prev_active = active
        for b in bead_points:
            b.step(tick_dt_s)

        if sim_t >= next_frame_t:
            renderer.update_scene(plant.data, camera=cam)
            _draw_markers(renderer.scene, polyline, cmd_pos)
            _draw_bead_trail(renderer.scene, bead_points, BEAD_RGBA)
            frames.append(renderer.render())
            next_frame_t += frame_interval_s
        if snap_idx < len(snap_fracs) and sim_t >= snap_fracs[snap_idx] * duration:
            renderer.update_scene(plant.data, camera=cam)
            _draw_markers(renderer.scene, polyline, cmd_pos)
            _draw_bead_trail(renderer.scene, bead_points, BEAD_RGBA)
            img = renderer.render()
            path = out_dir / f"sim_{name}_t{snap_idx}.png"
            imageio.imwrite(path, img)
            print(f"[render] saved {path}")
            snap_idx += 1

        remain = tick_dt_s - (time.monotonic() - now)
        if remain > 0:
            time.sleep(remain)

    stop.set()
    ai_thread.join(timeout=2.0)
    renderer.close()

    video_path = out_dir / f"sim_{name}.mp4"
    imageio.mimsave(video_path, frames, fps=fps, quality=8)
    print(f"[render] {name}: {len(frames)} frames, {len(bead_points)} beads -> {video_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=26.0)
    ap.add_argument("--seg-len", type=float, default=0.12)
    ap.add_argument("--speed", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out-dir", default="docs/ablation_results")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    print("recording baseline...")
    record_joint_sim(BASELINE_CFG, "baseline", duration=args.duration, seg_len=args.seg_len,
                      speed=args.speed, seed=args.seed, fps=args.fps, out_dir=out_dir)
    print("recording proposed...")
    record_joint_sim(PROPOSED_CFG, "proposed", duration=args.duration, seg_len=args.seg_len,
                      speed=args.speed, seed=args.seed, fps=args.fps, out_dir=out_dir)

    import os
    import sys

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
