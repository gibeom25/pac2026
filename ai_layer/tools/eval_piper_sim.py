#!/usr/bin/env python
"""BC 체크포인트를 Piper 부스 시뮬레이션(envs/piper_sim.py)에서 직접 돌려(closed-loop) 선 따라 그리기를 채점한다.

tools/eval_bc_offline.py는 정해진 시연 위에서 한 번씩 예측만 해 보는 평가라 정책 자신의 실수가 쌓이는 효과를
못 본다. 여기서는 정책이 낸 액션으로 실제로 플랜지를 움직이고, 그 결과 이미지를 다시 정책에 넣는다.
모든 체크포인트가 같은 장면/부스/시작 자세(--seed)로 돈다.

지표 (에피소드 평균):
  coverage     그려야 할 구간 중 비드가 2mm(cov2) / 5mm(cov5) 안에 떨어진 비율
  progress     펜 끝이 경로를 따라 어디까지 갔는지(경로 5mm 안에서 도달한 가장 먼 호 길이 비율)
  track_mm     펜 끝이 경로 위(작업 높이 근처)에 있을 때 경로까지의 평균 거리 [mm]
  off_bead     경로에서 3mm 넘게 벗어난 곳에 떨어진 비드 비율
  trig         트리거 켜진 프레임 비율
  tip_min_mm   펜 끝 최저 높이 [mm] (0 근처면 바닥에 닿음)

주의: 시뮬 점수는 실물 점수가 아니다(바닥 반사, 손으로 그린 선, 조명이 다름). 같은 시뮬에서 모델끼리 순위를
매기는 용도다. 또 실물 시연의 트리거 라벨은 켜짐 비율이 4%뿐이라, 실물로 학습한 정책은 트리거를 거의 안 켤 수
있다 — 그래서 경로 추종(progress, track_mm)도 따로 본다. 기존 팔 없는 시뮬(SO101 리그)로만 학습한 체크포인트는
state 좌표계가 달라서 여기서 의미 있게 움직이지 않는다.

사용 예:
  PYTHONPATH=. .venv/bin/python ai_layer/tools/eval_piper_sim.py \\
      --checkpoint outputs/bc_piper_ho/last outputs/bc_piper_wrist/last --episodes-per-scene 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scipy.spatial.transform import Rotation  # noqa: E402

from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE  # noqa: E402

from ai_layer.bc_inference import load_bc_checkpoint, predict_chunk  # noqa: E402
from ai_layer.configs.so101_act_bc import IMAGE_KEY, OVERVIEW_IMAGE_KEY  # noqa: E402
from ai_layer.envs.piper_sim import (  # noqa: E402
    HOME_FLANGE_POS,
    PiperSim,
    sample_booth,
    seam_path_booth,
    tilted_flange_R,
)
from ai_layer.envs.seam_ground_truth import SCENE_NAMES, SeamGroundTruth, dash_on_mask, gap_inner_mask  # noqa: E402
from ai_layer.kinematics import pose_to_state  # noqa: E402
from ai_layer.perception.seam_cv import SeamGrooveDetector  # noqa: E402
from ai_layer.perception.seam_features import seam_features_from_rgb  # noqa: E402
from ai_layer.tools.scripted_expert import GT_DENSE_POINTS, PATH_DS, _resample  # noqa: E402

HOVER_BAND = 0.02  # 펜 끝이 이 높이 아래일 때만 track_mm 계산 [m]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--scenes", nargs="+", default=SCENE_NAMES)
    p.add_argument("--episodes-per-scene", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=40.0, help="에피소드 최대 길이")
    p.add_argument("--exec-steps", type=int, default=8, help="chunk에서 몇 스텝 실행하고 다시 추론할지")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-overview-input", action="store_true", help="고정캠을 쓰는 정책에도 검은 화면을 넣는다")
    p.add_argument("--video-dir", default=None, help="에피소드별 mp4 저장 (손목|고정 나란히, 매 스텝 실시간 속도)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="결과 JSON (기본: 첫 체크포인트 폴더/eval_piper_sim.json)")
    return p.parse_args()


def episode_specs(args) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    specs = []
    for scene in args.scenes:
        for _ in range(args.episodes_per_scene):
            specs.append({
                "scene": scene, "variant": int(rng.integers(0, 5)), "booth": sample_booth(rng),
                "home": HOME_FLANGE_POS + rng.normal(0, [0.02, 0.01, 0.005]),
                "tilt": float(rng.uniform(13.0, 16.0)),
            })
    return specs


def score(tips: np.ndarray, beads: np.ndarray, trig: np.ndarray, path_xy: np.ndarray, path_on: np.ndarray) -> dict:
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))])
    d_tip = np.linalg.norm(tips[:, None, :2] - path_xy[None], axis=2)
    near_i, near_d = np.argmin(d_tip, axis=1), np.min(d_tip, axis=1)
    low = tips[:, 2] < HOVER_BAND
    reach = near_d < 0.005
    out = {
        "progress": float(s[near_i[reach]].max() / s[-1]) if reach.any() else 0.0,
        "track_mm": float(near_d[low].mean() * 1000) if low.any() else float("nan"),
        "trig": float(trig.mean()),
        "tip_min_mm": float(tips[:, 2].min() * 1000),
    }
    probe = path_xy[path_on][::4]
    if len(beads):
        d_pb = np.min(np.linalg.norm(probe[:, None] - beads[None], axis=2), axis=1)
        out["cov2"], out["cov5"] = float(np.mean(d_pb < 0.002)), float(np.mean(d_pb < 0.005))
        d_all = np.linalg.norm(beads[:, None] - path_xy[None], axis=2)
        nearest = np.argmin(d_all, axis=1)
        bad = (d_all[np.arange(len(beads)), nearest] > 0.003) | gap_inner_mask(path_on, PATH_DS)[nearest]
        out["off_bead"] = float(bad.mean())
    else:
        out["cov2"] = out["cov5"] = 0.0
        out["off_bead"] = float("nan")
    return out


def run_episode(policy, pre, post, spec: dict, gt: SeamGroundTruth, args, detector, writer=None) -> dict:
    booth = spec["booth"]
    sim = PiperSim(spec["scene"], spec["variant"], booth)
    dt = 1.0 / args.fps
    uses_overview = OVERVIEW_IMAGE_KEY in policy.config.image_features
    try:
        R = tilted_flange_R(spec["tilt"])
        pose = sim.move_flange(spec["home"], R)
        pos, R = pose[:3].copy(), Rotation.from_rotvec(pose[3:]).as_matrix()
        path_xy = _resample(seam_path_booth(gt, spec["scene"], spec["variant"], booth), PATH_DS)
        path_on = dash_on_mask(path_xy, gt.dash_pattern(spec["scene"], spec["variant"]))
        tips, trigs = [], []
        chunk, k = None, 0
        for _ in range(int(args.max_seconds * args.fps)):
            if chunk is None or k >= args.exec_steps:
                wrist = sim.render("wrist")
                obs = {
                    IMAGE_KEY: torch.from_numpy(wrist).permute(2, 0, 1).float().div(255)[None],
                    OBS_STATE: torch.from_numpy(pose_to_state(sim.flange_pose_xyzrotvec())).float()[None],
                    OBS_ENV_STATE: torch.from_numpy(seam_features_from_rgb(detector, wrist)).float()[None],
                }
                if uses_overview and not args.no_overview_input:
                    obs[OVERVIEW_IMAGE_KEY] = torch.from_numpy(sim.render("overview")).permute(2, 0, 1).float().div(255)[None]
                obs = {kk: v.to(args.device) for kk, v in obs.items()}
                chunk = predict_chunk(policy, pre, post, obs)[0].numpy()
                k = 0
            a = chunk[k]
            k += 1
            pos = pos + a[:3]
            R = Rotation.from_rotvec(a[3:6]).as_matrix() @ R
            achieved = sim.move_flange(pos, R)
            pos, R = achieved[:3].copy(), Rotation.from_rotvec(achieved[3:]).as_matrix()  # 바닥/관절 한계로 잘린 만큼 반영
            trig = bool(a[6] > 0.5)
            sim.step_beads(trig, dt)
            tips.append(sim.tip_pos())
            trigs.append(trig)
            if writer is not None:
                writer.append_data(np.concatenate([sim.render("wrist"), sim.render("overview")], axis=1))
        beads = np.array([b.pos[:2] for b in sim.bead_points]) if sim.bead_points else np.zeros((0, 2))
        return score(np.array(tips), beads, np.array(trigs), path_xy, path_on)
    finally:
        sim.close()


def main() -> None:
    args = parse_args()
    specs = episode_specs(args)
    gt = SeamGroundTruth(num_points=GT_DENSE_POINTS)
    detector = SeamGrooveDetector()
    keys = ["cov2", "cov5", "progress", "track_mm", "off_bead", "trig", "tip_min_mm"]
    results = {}
    for ck in args.checkpoint:
        policy, pre, post = load_bc_checkpoint(ck, device=args.device)
        name = f"{Path(ck).parent.name}/{Path(ck).name}"
        rows = []
        for i, spec in enumerate(specs):
            writer = None
            if args.video_dir:
                import imageio

                vdir = Path(args.video_dir) / Path(ck).parent.name
                vdir.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(vdir / f"{i:02d}_{spec['scene']}_{spec['variant']}.mp4", fps=args.fps)
            try:
                r = run_episode(policy, pre, post, spec, gt, args, detector, writer)
            finally:
                if writer is not None:
                    writer.close()
            r.update(scene=spec["scene"], variant=spec["variant"])
            rows.append(r)
            print(f"[{name}] {i + 1}/{len(specs)} {spec['scene']}:{spec['variant']} "
                  + " ".join(f"{kk}={r[kk]:.3f}" for kk in keys), flush=True)
        results[name] = rows
        del policy
        torch.cuda.empty_cache()

    width = max(len(n) for n in results) + 2
    print("\n== 평균 ==")
    print("".ljust(width) + "".join(kk.rjust(11) for kk in keys))
    summary = {}
    for name, rows in results.items():
        summary[name] = {kk: float(np.nanmean([r[kk] for r in rows])) for kk in keys}
        print(name.ljust(width) + "".join(f"{summary[name][kk]:11.3f}" for kk in keys))
    print("\n== 장면별 progress ==")
    print("".ljust(width) + "".join(sc[:11].rjust(12) for sc in args.scenes))
    for name, rows in results.items():
        print(name.ljust(width) + "".join(
            f"{np.mean([r['progress'] for r in rows if r['scene'] == sc]):12.3f}" for sc in args.scenes))

    out = Path(args.out) if args.out else Path(args.checkpoint[0]) / "eval_piper_sim.json"
    out.write_text(json.dumps({"summary": summary, "episodes": results, "args": {
        k: v for k, v in vars(args).items() if k != "checkpoint"} | {"checkpoint": args.checkpoint}},
        indent=2, ensure_ascii=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
