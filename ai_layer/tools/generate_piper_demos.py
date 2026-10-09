#!/usr/bin/env python
"""Piper 부스 시뮬레이션(envs/piper_sim.py)에서 스크립트 전문가 시연 데이터셋을 자동 생성한다.

generate_demos.py(로봇 팔 없는 기존 시뮬)와 같은 경로 계획(scripted_expert.plan_episode), 외란, 품질 검사,
병렬 생성을 쓰고, 기록 형식은 실물 Piper 데이터셋(ddyyuu/piper-user-20261008-flange-2cam-v1)과 같다:
  observation.state (9)          플랜지 pose [xyz(베이스), rot6d]
  observation.images.wrist       손목 카메라 240x320
  observation.images.overview    고정 카메라 240x320 (로봇 팔까지 렌더 — 기존 시뮬과 달리 검은 화면이 아님)
  observation.joint_position (6) 관절각 [rad]
  action (7)                     직전 프레임 대비 플랜지 실측 증분 + 트리거
그래서 실물 데이터와 섞어 학습하거나(train_bc.py --repo-id 실물 시뮬) 이 데이터로만 사전학습한 뒤 실물로
파인튜닝할 수 있다.

에피소드마다 무작위: 장면/variant, 부스(고정 카메라 위치·방향, 용지 위치·방향, 밝기, 바닥 반사 — sample_booth),
시작 위치, 펜 기울기, 경로 속도, 작업 높이, 외란, 필라멘트 색. meta/scripted_params.jsonl에 기록.

실행:
  PYTHONPATH=. .venv/bin/python ai_layer/tools/generate_piper_demos.py --repo-id piper-sim-test --num-episodes 2
  PYTHONPATH=. .venv/bin/python ai_layer/tools/generate_piper_demos.py --repo-id piper-sim-1000 --num-episodes 1000 --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.envs.piper_sim import (  # noqa: E402
    CAMERA_HW,
    HOME_FLANGE_POS,
    PEN_TIP_FLANGE,
    PiperSim,
    sample_booth,
    seam_to_booth_xy,
    tilted_flange_R,
)
from ai_layer.envs.seam_ground_truth import SeamGroundTruth, gap_inner_mask  # noqa: E402
from ai_layer.kinematics import pose_delta, pose_to_state  # noqa: E402
from ai_layer.tools.generate_demos import (  # noqa: E402
    COVERAGE_RADIUS,
    MAX_OFF_TARGET_FRAC,
    OFF_TARGET_DIST,
    _sample_bead_rgba,
    generate_parallel,
)
from ai_layer.tools.record_mujoco import (  # noqa: E402
    DATASETS_DIR,
    SCENE_VARIANTS,
    BalancedSceneSampler,
    _build_combos,
    _existing_episode_count,
)
from ai_layer.tools.scripted_expert import (  # noqa: E402
    GT_DENSE_POINTS,
    PATH_DS,
    TRIGGER_OFF_XY_TOL,
    TRIGGER_ON_XY_TOL,
    TRIGGER_Z_TOL,
    nearest_on_path,
    plan_episode,
    sample_params,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ROBOT_TYPE = "piper_ee_sim_scripted"  # "_ee_"가 들어가야 data.detect_dataset_kind가 EE 포맷으로 인식
STATE_NAMES = ["x", "y", "z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
JOINT_NAMES = [f"joint_{i}_rad" for i in range(1, 7)]
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (9,), "names": STATE_NAMES},
    "observation.images.wrist": {"dtype": "image", "shape": (*CAMERA_HW, 3), "names": ["height", "width", "channels"]},
    "observation.images.overview": {"dtype": "image", "shape": (*CAMERA_HW, 3), "names": ["height", "width", "channels"]},
    "observation.joint_position": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Piper 부스 시뮬레이션 시연 데이터셋 자동 생성.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="로컬 저장 경로 (없으면 datasets/<repo-id>)")
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scene", choices=["balanced", *SCENE_VARIANTS], default="balanced")
    p.add_argument("--variant", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tilt-deg", type=float, nargs=2, default=(12.0, 17.0), metavar=("MIN", "MAX"),
                   help="플랜지 기울기 범위 [deg] — 실물 시연은 14.5° 고정")
    p.add_argument("--hover-mm", type=float, nargs=2, default=(5.0, 12.0), metavar=("MIN", "MAX"),
                   help="경로 추종 중 펜 끝 높이 범위 [mm]")
    p.add_argument("--max-disturbances", type=int, default=2)
    p.add_argument("--disturb-mm", type=float, nargs=2, default=(5.0, 12.0), metavar=("MIN", "MAX"))
    p.add_argument("--max-linear-speed", type=float, default=0.06, help="명령 목표 이동 속도 상한 [m/s]")
    p.add_argument("--min-coverage", type=float, default=0.95)
    p.add_argument("--bead-color", choices=["random", "fixed"], default="random")
    p.add_argument("--dashed", choices=["cut", "bridge"], default="cut")
    p.add_argument("--task", default="follow seam with Piper tool (mujoco piper booth, scripted)")
    p.add_argument("--workers", type=int, default=1, help="병렬 생성 프로세스 수")
    p.add_argument("--overwrite", action="store_true", help="기존 데이터셋을 지우고 새로 생성")
    return p.parse_args()


def _dataset_root(args) -> Path:
    return Path(args.root) if args.root else DATASETS_DIR / args.repo_id


def build_dataset(args) -> LeRobotDataset:
    root = _dataset_root(args)
    if root.exists():
        if _existing_episode_count(root) and not args.overwrite:
            sys.exit(f"[piper-gen] 기존 데이터셋이 있습니다({root}) — --overwrite를 주세요.")
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=FEATURES, root=root, robot_type=ROBOT_TYPE, use_videos=False
    )


def _schedule_disturbances(follow_range, n: int, mm_range, rng: np.random.Generator) -> dict[int, np.ndarray]:
    f0, f1 = follow_range
    lo, hi = f0 + int(0.15 * (f1 - f0)), f0 + int(0.85 * (f1 - f0))
    if n == 0 or hi - lo < 60 * n:
        return {}
    frames = rng.choice(np.arange(lo, hi, 60), size=min(n, len(range(lo, hi, 60))), replace=False)
    out = {}
    for f in frames:
        a = rng.uniform(0, 2 * np.pi)
        mag = rng.uniform(*mm_range) * 1e-3
        out[int(f)] = np.array([np.cos(a) * mag, np.sin(a) * mag, rng.uniform(0.0, 0.004)])
    return out


def _bead_quality(beads: np.ndarray, path_xy: np.ndarray, path_on: np.ndarray) -> tuple[float, float]:
    """(그려야 할 구간 커버리지, 잘못 떨어진 비드 비율) — generate_demos._bead_quality와 같은 기준."""
    if len(beads) == 0:
        return 0.0, 0.0
    probe = path_xy[path_on][::4]
    d_path_to_bead = np.min(np.linalg.norm(probe[:, None] - beads[None], axis=2), axis=1)
    d_all = np.linalg.norm(beads[:, None] - path_xy[None], axis=2)
    nearest = np.argmin(d_all, axis=1)
    bad = (d_all[np.arange(len(beads)), nearest] > OFF_TARGET_DIST) | gap_inner_mask(path_on, PATH_DS)[nearest]
    return float(np.mean(d_path_to_bead < COVERAGE_RADIUS)), float(np.mean(bad))


def run_episode(args, scene: str, variant: int, gt: SeamGroundTruth, rng: np.random.Generator, dataset) -> tuple[bool, dict]:
    booth = sample_booth(rng)
    sim = PiperSim(scene, variant, booth)
    dt = 1.0 / args.fps
    try:
        params = sample_params(rng, 0.0, args.max_disturbances, tuple(h * 1e-3 for h in args.hover_mm))
        tilt = float(rng.uniform(*args.tilt_deg))
        R = tilted_flange_R(tilt, yaw_deg=float(rng.uniform(-3, 3)))
        bead_name, sim.bead_rgba = _sample_bead_rgba(rng) if args.bead_color == "random" else ("natural", sim.bead_rgba)
        home = HOME_FLANGE_POS + rng.normal(0, [0.02, 0.01, 0.005])
        sim.move_flange(home, R)
        home_tip = sim.tip_pos()

        plan = plan_episode(gt, scene, variant, home_tip, params, dt, dashed=args.dashed,
                            xy_transform=lambda xy: seam_to_booth_xy(xy, booth))
        disturb = _schedule_disturbances(plan.follow_range, params.n_disturb, args.disturb_mm, rng)
        tip_off = R @ PEN_TIP_FLANGE
        cmd = sim.flange_pose()[0]
        prev_pose = None
        trigger = False
        f0, f1 = plan.follow_range
        max_step = args.max_linear_speed * dt
        tip_min_z = np.inf
        for k in range(len(plan.tip_ref)):
            if k in disturb:  # 밖에서 밀린 것 — action에 넣지 않도록 직전 pose를 밀린 뒤로 다시 잡는다
                cmd = cmd + disturb[k]
                prev_pose = sim.move_flange(cmd, R)
            step = (plan.tip_ref[k] - tip_off) - cmd
            n = np.linalg.norm(step)
            cmd = cmd + (step * (max_step / n) if n > max_step else step)
            pose = sim.move_flange(cmd, R)
            tip = sim.tip_pos()
            tip_min_z = min(tip_min_z, tip[2])

            if f0 <= k < f1:
                i, d = nearest_on_path(plan.path_xy, tip[:2])
                z_ok = abs(tip[2] - plan.tip_ref[k, 2]) < TRIGGER_Z_TOL
                trigger = bool(plan.path_on[i] and z_ok and d < (TRIGGER_OFF_XY_TOL if trigger else TRIGGER_ON_XY_TOL))
            else:
                trigger = False
            sim.step_beads(trigger, dt)

            delta = np.zeros(6) if prev_pose is None else pose_delta(prev_pose, pose)
            prev_pose = pose
            dataset.add_frame({
                "observation.state": pose_to_state(pose).astype(np.float32),
                "observation.images.wrist": sim.render("wrist"),
                "observation.images.overview": sim.render("overview"),
                "observation.joint_position": sim.joint_positions().astype(np.float32),
                "action": np.array([*delta, float(trigger)], dtype=np.float32),
                "task": args.task,
            })

        beads = np.array([b.pos[:2] for b in sim.bead_points]) if sim.bead_points else np.zeros((0, 2))
        coverage, off_target = _bead_quality(beads, plan.path_xy, plan.path_on)
        ok = coverage >= args.min_coverage and off_target <= MAX_OFF_TARGET_FRAC
        if ok:
            dataset.save_episode()
        else:
            dataset.clear_episode_buffer()
        info = {
            "scene": scene, "variant": variant, "frames": len(plan.tip_ref), "coverage": round(coverage, 4),
            "off_target": round(off_target, 4), "tip_min_z": round(float(tip_min_z), 4), "tilt_deg": round(tilt, 2),
            "bead_color": bead_name, "booth": booth.to_json(),
            "disturbances": {str(k): v.round(4).tolist() for k, v in disturb.items()},
            **{k: v for k, v in vars(params).items() if k not in ("disturbances", "tilt_deg")},
        }
        return ok, info
    finally:
        sim.close()


def generate(args, label: str = "") -> tuple[int, int]:
    rng = np.random.default_rng(args.seed)
    dataset = build_dataset(args)
    root = Path(dataset.root)
    sampler = BalancedSceneSampler(
        _build_combos(args.scene, args.variant), root / "meta" / "scene_balance.json", rng=random.Random(args.seed)
    )
    gt = SeamGroundTruth(num_points=GT_DENSE_POINTS)
    params_log = root / "meta" / "scripted_params.jsonl"
    saved = discarded = 0
    t_start = time.perf_counter()
    try:
        while saved < args.num_episodes:
            scene, variant = sampler.pick()
            ok, info = run_episode(args, scene, variant, gt, rng, dataset)
            tag = f"{scene}:{variant} {info['frames']}f cov={info['coverage']:.3f} off={info['off_target']:.3f}"
            if ok:
                sampler.commit(scene, variant)
                saved += 1
                info["episode_index"] = dataset.meta.total_episodes - 1
                with params_log.open("a") as f:
                    f.write(json.dumps(info) + "\n")
                print(f"[piper-gen{label}] {saved}/{args.num_episodes} 저장 — {tag}", flush=True)
            else:
                discarded += 1
                print(f"[piper-gen{label}] 폐기(품질 미달) — {tag}", flush=True)
    except KeyboardInterrupt:
        print("\n[piper-gen] 중단됨 — 이미 저장된 에피소드는 유지됩니다.")
    finally:
        dataset.finalize()
    print(f"[piper-gen{label}] 완료: 저장 {saved}개, 폐기 {discarded}개, {time.perf_counter() - t_start:.0f}s — {root}")
    return saved, discarded


def _worker(args, label: str) -> None:
    try:
        generate(args, label)
    except KeyboardInterrupt:
        pass


def main() -> None:
    args = parse_args()
    if args.workers > 1:
        generate_parallel(args, worker=_worker)
    else:
        generate(args)


if __name__ == "__main__":
    main()
