#!/usr/bin/env python
"""스크립트 전문가(scripted_expert.py)로 시뮬레이션 시연 데이터셋을 자동 생성한다.

record_mujoco.py와 같은 EE-native LeRobotDataset 포맷/같은 EpisodeTicker 물리를 쓰고, 사람
입력 대신 ScriptedEEController가 조작한다(자세한 배경은 scripted_expert.py docstring). 기본은
창 없이 최대한 빠르게 돌고, --view를 주면 MuJoCo 뷰어로 실시간 속도로 보여준다.

에피소드마다 무작위로 바뀌는 것(meta/scripted_params.jsonl에 기록):
  씬/variant(BalancedSceneSampler, record_mujoco.py와 동일), 경로 속도, 작업 높이, 접근 높이,
  펜 기울기(위쪽이 로봇 쪽으로 0~--max-tilt-deg), 하강 지점 오프셋, 곡선 감속 정도, 외란 횟수/크기.
외란: 경로 추종 중 목표를 옆으로 순간 밀어낸다(--disturb-mm). 밀려난 그 프레임은 "에이전트의
행동"이 아니라 바깥에서 가해진 것이므로 action 라벨에 넣지 않고(prev_pose를 밀린 뒤 자세로
다시 잡음), 이후 전문가가 경로로 복귀하는 동작만 기록된다 — BC가 복귀 행동을 배우게 하려는 것.

점선(dashed): 기본(--dashed cut)은 선이 끊긴 구간에서 분사를 멈춘다(펜은 경로를 그대로 따라감).
--dashed bridge면 예전처럼 끊긴 구간도 이어서 그린다(gen_seam_textures.py의 원래 GT 규약).

품질 검사: 비드가 그려야 할 구간을 --min-coverage 이상 덮지 못했거나, 경로에서 3mm 넘게
벗어나거나 점선의 끊긴 구간 안쪽에 떨어진 비드가 2% 넘거나, 막대가 바닥에 닿으면 그 에피소드는
폐기하고 다른 파라미터로 다시 뽑는다.

병렬 생성(--workers N): 워커 프로세스 N개가 각자 <root>_shards/wNN에 데이터셋 조각을 만들고, 다 끝나면
lerobot aggregate_datasets로 <root> 하나로 합친다(scripted_params.jsonl/scene_balance.json도 합침).
워커 k의 시드는 seed*1000+k라서 --workers 값이 다르면 같은 --seed여도 다른 데이터가 나온다.

실행:
  PYTHONPATH=. uv run python ai_layer/tools/generate_demos.py --repo-id me/so101-weld-scripted --num-episodes 300
  PYTHONPATH=. uv run python ai_layer/tools/generate_demos.py --repo-id me/so101-weld-scripted --num-episodes 1000 --workers 16
  PYTHONPATH=. uv run python ai_layer/tools/generate_demos.py --repo-id me/so101-weld-scripted --num-episodes 2 --view
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

if not os.environ.get("DISPLAY") and "--view" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")  # 디스플레이 없는 서버에서도 오프스크린 렌더

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai_layer.envs.seam_ground_truth import SeamGroundTruth, gap_inner_mask  # noqa: E402
from ai_layer.tools.episode_ticker import EpisodeTicker, MujocoDualCamera  # noqa: E402
from ai_layer.tools.record_mujoco import (  # noqa: E402
    ACTION_KEYS,
    CAMERA_HW,
    CAMERA_NAME,
    DATASETS_DIR,
    MAX_ANGULAR_SPEED_DEFAULT,
    SCENE_VARIANTS,
    STATE_KEYS,
    BalancedSceneSampler,
    _build_combos,
    _draw_bead_trail,
    _ee_pose_xyzrotvec,
    _existing_episode_count,
    _mjcf_path,
    _rod_tip_world,
)
from ai_layer.tools.scripted_expert import (  # noqa: E402
    GT_DENSE_POINTS,
    PATH_DS,
    ScriptedEEController,
    nearest_on_path,
    plan_episode,
    sample_params,
)

from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import hw_to_dataset_features  # noqa: E402

# 2026-10-08: 실물은 3D 펜으로 그리는데 필라멘트 색이 정해지지 않았다 — 비드를 늘 같은 미색으로만
# 그리면 정책이 "비드 = 미색"에 묶이므로 에피소드마다 흔한 PLA 필라멘트 색 중 하나를 고르고
# 밝기/채도를 조금씩 흔든다(--bead-color fixed로 끄면 기존 미색). 검정은 선(검정)과 구분이 안 돼
# 어렵지만 실물에서 쓸 수도 있으므로 일부러 넣었다.
FILAMENT_PALETTE = {
    "white": (0.93, 0.93, 0.92), "natural": (0.85, 0.83, 0.78), "black": (0.08, 0.08, 0.09),
    "gray": (0.5, 0.5, 0.52), "red": (0.8, 0.1, 0.1), "orange": (0.95, 0.45, 0.08),
    "yellow": (0.95, 0.85, 0.1), "green": (0.15, 0.65, 0.2), "blue": (0.1, 0.3, 0.85),
    "sky": (0.4, 0.7, 0.95), "purple": (0.5, 0.2, 0.7), "pink": (0.95, 0.5, 0.7),
}

ROBOT_TYPE = "so101_ee_mujoco_scripted"  # "ee_mujoco"가 들어가야 data.detect_dataset_kind가 EE 포맷으로 인식
COVERAGE_RADIUS = 0.002
OFF_TARGET_DIST = 0.003
MAX_OFF_TARGET_FRAC = 0.02


class WristOnlyCamera(MujocoDualCamera):
    """EpisodeTicker는 매 틱 오버뷰도 렌더하지만 데이터셋엔 손목만 들어간다 — 생성 속도를 위해 생략."""

    def get_overview_frame(self, data, bead_points):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="스크립트 전문가로 시뮬레이션 시연 데이터셋 자동 생성.")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="로컬 저장 경로 (없으면 datasets/<repo-id>)")
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--scene", choices=["balanced", *SCENE_VARIANTS], default="balanced")
    p.add_argument("--variant", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tilt-deg", type=float, default=5.0,
                   help="펜 위쪽이 로봇 베이스 쪽으로 기우는 각도 최대 [deg] — 에피소드마다 0~이 값에서 뽑음")
    p.add_argument("--max-disturbances", type=int, default=2, help="에피소드당 외란 최대 횟수 (0이면 외란 없음)")
    p.add_argument("--disturb-mm", type=float, nargs=2, default=(5.0, 12.0), metavar=("MIN", "MAX"),
                   help="외란 크기 범위 [mm]")
    p.add_argument("--max-linear-speed", type=float, default=0.06,
                   help="추종 속도 상한 [m/s] — 외란 뒤 복귀 속도가 사실상 이 값")
    p.add_argument("--max-angular-speed", type=float, default=MAX_ANGULAR_SPEED_DEFAULT)
    p.add_argument("--min-coverage", type=float, default=0.95, help="이보다 경로를 덜 덮으면 폐기")
    p.add_argument("--bead-color", choices=["random", "fixed"], default="random",
                   help="random: 에피소드마다 필라멘트 색(FILAMENT_PALETTE) 무작위, fixed: 기존 미색")
    p.add_argument("--dashed", choices=["cut", "bridge"], default="cut",
                   help="점선: cut=끊긴 구간에서 분사 멈춤, bridge=끊긴 구간도 이어서 그림")
    p.add_argument("--hover-mm", type=float, nargs=2, default=(9.0, 11.0), metavar=("MIN", "MAX"),
                   help="경로 추종 중 도구 끝 높이 범위 [mm] — 실물 3D 펜은 약 1cm 띄워 그림 (MIN_TIP_Z=6mm보다 커야 함)")
    p.add_argument("--task", default="weld seam following demo (mujoco ee-only, scripted)")
    p.add_argument("--view", action="store_true", help="MuJoCo 뷰어로 보면서 실시간 속도로 생성")
    p.add_argument("--workers", type=int, default=1,
                   help="병렬 생성 프로세스 수 (1보다 크면 조각으로 나눠 만든 뒤 합친다, --view/--resume 불가)")
    exist = p.add_mutually_exclusive_group()
    exist.add_argument("--overwrite", action="store_true", help="기존 데이터셋을 지우고 새로 생성")
    exist.add_argument("--resume", action="store_true", help="기존 데이터셋 뒤에 이어서 생성")
    args = p.parse_args()
    if args.workers > 1 and (args.view or args.resume):
        p.error("--workers는 --view, --resume과 같이 쓸 수 없다.")
    return args


def _dataset_root(args: argparse.Namespace) -> Path:
    return Path(args.root) if args.root else DATASETS_DIR / args.repo_id


def build_dataset(args: argparse.Namespace) -> LeRobotDataset:
    root = _dataset_root(args)
    if root.exists():
        n = _existing_episode_count(root)
        if n == 0 or args.overwrite:
            shutil.rmtree(root)
        elif args.resume:
            print(f"[gen] 이어서 생성합니다 (기존 {n}개 에피소드 뒤에 추가): {root}")
            return LeRobotDataset(repo_id=args.repo_id, root=root)
        else:
            sys.exit(f"[gen] 기존 데이터셋이 있습니다({root}, 에피소드 {n}개) — --resume 또는 --overwrite를 주세요.")

    hw_obs = {name: float for name in STATE_KEYS}
    hw_obs["wrist"] = CAMERA_HW
    hw_action = {name: float for name in ACTION_KEYS}
    features = {
        **hw_to_dataset_features(hw_obs, "observation", use_video=False),
        **hw_to_dataset_features(hw_action, "action", use_video=False),
    }
    return LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=features, root=root, robot_type=ROBOT_TYPE, use_videos=False
    )


def _sample_bead_rgba(rng: np.random.Generator) -> tuple[str, np.ndarray]:
    name = str(rng.choice(list(FILAMENT_PALETTE)))
    rgb = np.clip(np.asarray(FILAMENT_PALETTE[name]) * rng.uniform(0.85, 1.1) + rng.normal(0, 0.03, 3), 0, 1)
    return name, np.array([*rgb, 0.95], dtype=np.float32)


def _ticker_args(args: argparse.Namespace) -> argparse.Namespace:
    """EpisodeTicker가 읽는 필드만 채운 Namespace (입력 반전 없음, 길이 제한은 계획이 정함)."""
    return argparse.Namespace(
        fps=args.fps, episode_seconds=None, task=args.task,
        max_linear_speed=args.max_linear_speed, max_angular_speed=args.max_angular_speed,
        invert_x=False, invert_y=False, invert_z=False, invert_roll=False, invert_pitch=False,
    )


def _schedule_disturbances(plan, n: int, mm_range, rng: np.random.Generator) -> list[dict]:
    f0, f1 = plan.follow_range
    lo, hi = f0 + int(0.15 * (f1 - f0)), f0 + int(0.85 * (f1 - f0))
    if n == 0 or hi - lo < 60 * n:
        return []
    frames = sorted(rng.choice(np.arange(lo, hi, 60), size=min(n, len(range(lo, hi, 60))), replace=False).tolist())
    out = []
    for f in frames:
        i, _ = nearest_on_path(plan.path_xy, plan.tip_ref[f, :2])
        tx, ty = plan.path_tangent[i]
        normal = np.array([-ty, tx]) * rng.choice([-1.0, 1.0])
        mag = rng.uniform(*mm_range) * 1e-3
        offset = [float(normal[0] * mag), float(normal[1] * mag), float(rng.uniform(0.0, 0.004))]
        out.append({"frame": int(f), "offset": offset})
    return out


def _apply_disturbance(ticker: EpisodeTicker, offset) -> None:
    ticker.target_pos = ticker.target_pos + np.asarray(offset)
    ticker.data.mocap_pos[ticker.mocap_idx] = ticker.target_pos
    for _ in range(ticker.substeps):  # 물리 바디가 밀린 자리로 따라가게 한 프레임 분량 진행 (기록 안 함)
        mujoco.mj_step(ticker.model, ticker.data)
    ticker.prev_pose = _ee_pose_xyzrotvec(ticker.data, ticker.ee_bid, ticker.rod_gid)


def _bead_quality(ticker: EpisodeTicker, path_xy: np.ndarray, path_on: np.ndarray) -> tuple[float, float]:
    """(그려야 할 구간 커버리지, 잘못 떨어진 비드 비율). 비드는 낙하 후 xy가 그대로라 생성 위치로 본다.

    잘못 떨어진 비드 = 경로에서 OFF_TARGET_DIST 넘게 벗어났거나, 점선의 끊긴 구간 "안쪽"(경계에서
    seam_ground_truth.GAP_MARGIN 넘게 들어간 곳)에 떨어진 것. 경계 근처는 비드 반지름만큼 번지는 게 정상이라 봐준다.
    """
    if not ticker.bead_points:
        return 0.0, 0.0
    beads = np.array([b.pos[:2] for b in ticker.bead_points])
    probe = path_xy[path_on][::4]
    d_path_to_bead = np.min(np.linalg.norm(probe[:, None] - beads[None], axis=2), axis=1)
    d_all = np.linalg.norm(beads[:, None] - path_xy[None], axis=2)
    nearest = np.argmin(d_all, axis=1)
    gap_inner = gap_inner_mask(path_on, PATH_DS)
    bad = (d_all[np.arange(len(beads)), nearest] > OFF_TARGET_DIST) | gap_inner[nearest]
    return float(np.mean(d_path_to_bead < COVERAGE_RADIUS)), float(np.mean(bad))


def run_episode(args, model, data, cam, scene, variant, gt, rng, dataset, viewer) -> tuple[str, dict]:
    ticker = EpisodeTicker(_ticker_args(args), model, data, cam, scene, variant)
    params = sample_params(rng, args.max_tilt_deg, args.max_disturbances, tuple(h * 1e-3 for h in args.hover_mm))
    if args.bead_color == "random":
        bead_name, cam.bead_rgba = _sample_bead_rgba(rng)
    else:
        bead_name = "natural"
    home_tip = _rod_tip_world(data, ticker.rod_gid)
    plan = plan_episode(gt, scene, variant, home_tip, params, ticker.dt, dashed=args.dashed)
    params.disturbances = _schedule_disturbances(plan, params.n_disturb, args.disturb_mm, rng)
    disturb_at = {d["frame"]: d["offset"] for d in params.disturbances}
    ctl = ScriptedEEController(ticker, plan)

    n_frames = len(plan.tip_ref)
    contact_frames = 0
    coverage = off_target = 0.0
    result = None
    for k in range(n_frames):
        t_loop = time.perf_counter()
        if k in disturb_at:
            _apply_disturbance(ticker, disturb_at[k])
        last = k == n_frames - 1
        bad = False
        if last:
            coverage, off_target = _bead_quality(ticker, plan.path_xy, plan.path_on)
            bad = coverage < args.min_coverage or off_target > MAX_OFF_TARGET_FRAC or contact_frames > 0
        result = ticker.tick(ctl, dataset, force_end=last and not bad, force_discard=last and bad)
        contact_frames += int(ticker.last_contact)
        if viewer is not None:
            viewer.user_scn.ngeom = 0
            _draw_bead_trail(viewer.user_scn, ticker.bead_points, cam.bead_rgba)
            viewer.sync()
            if not viewer.is_running():
                raise KeyboardInterrupt
            time.sleep(max(0.0, ticker.dt - (time.perf_counter() - t_loop)))
        if result is not None:
            break

    info = {
        "scene": scene, "variant": variant, "frames": n_frames, "coverage": round(coverage, 4),
        "off_target": round(off_target, 4), "contact_frames": contact_frames,
        "bead_color": bead_name, "bead_rgba": [round(float(c), 3) for c in cam.bead_rgba],
        **{k: v for k, v in vars(params).items()},
    }
    return result, info


def generate(args: argparse.Namespace, label: str = "") -> tuple[int, int]:
    """args.num_episodes개를 저장할 때까지 생성. (저장 수, 폐기 수) 반환."""
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
            model = mujoco.MjModel.from_xml_path(str(_mjcf_path(scene, variant)))
            data = mujoco.MjData(model)
            cam = WristOnlyCamera(model)
            try:
                if args.view:
                    with mujoco.viewer.launch_passive(model, data) as viewer:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
                        result, info = run_episode(args, model, data, cam, scene, variant, gt, rng, dataset, viewer)
                else:
                    result, info = run_episode(args, model, data, cam, scene, variant, gt, rng, dataset, None)
            finally:
                cam.close()

            tag = (f"{scene}:{variant} {info['frames']}f cov={info['coverage']:.3f} "
                   f"off={info['off_target']:.3f} disturb={len(info['disturbances'])}")
            if result == "saved":
                sampler.commit(scene, variant)
                saved += 1
                info["episode_index"] = dataset.meta.total_episodes - 1
                with params_log.open("a") as f:
                    f.write(json.dumps(info) + "\n")
                print(f"[gen{label}] {saved}/{args.num_episodes} 저장 — {tag}", flush=True)
            else:
                discarded += 1
                print(f"[gen{label}] 폐기(품질 미달) — {tag}", flush=True)
    except KeyboardInterrupt:
        print("\n[gen] 중단됨 — 이미 저장된 에피소드는 유지됩니다.")
    finally:
        dataset.finalize()  # 안 부르면 parquet footer가 안 써져서 데이터셋이 깨진다

    print(f"[gen{label}] 완료: 저장 {saved}개, 폐기 {discarded}개, {time.perf_counter() - t_start:.0f}s — {root}")
    return saved, discarded


def _worker(args: argparse.Namespace, label: str) -> None:
    try:
        generate(args, label)
    except KeyboardInterrupt:
        pass  # generate()가 이미 finalize함 — 부모가 저장된 만큼 합친다


def generate_parallel(args: argparse.Namespace, worker=None) -> None:
    """worker(args, label): 조각 하나를 만드는 최상위 함수 (기본: 이 모듈의 _worker). generate_piper_demos.py가 자기 것을 넘긴다."""
    root = _dataset_root(args)
    if root.exists():
        n = _existing_episode_count(root)
        if n > 0 and not args.overwrite:
            sys.exit(f"[gen] 기존 데이터셋이 있습니다({root}, 에피소드 {n}개) — --overwrite를 주세요.")
        shutil.rmtree(root)
    shard_dir = root.with_name(root.name + "_shards")
    if shard_dir.exists():
        shutil.rmtree(shard_dir)

    n_workers = min(args.workers, args.num_episodes)
    counts = [args.num_episodes // n_workers + (k < args.num_episodes % n_workers) for k in range(n_workers)]
    shards = []
    for k, n in enumerate(counts):
        a = copy.copy(args)
        a.repo_id, a.root = f"{args.repo_id}_w{k:02d}", str(shard_dir / f"w{k:02d}")
        a.num_episodes, a.seed, a.workers, a.overwrite = n, args.seed * 1000 + k, 1, True
        shards.append(a)

    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")  # MuJoCo 렌더 컨텍스트를 fork로 복제하지 않도록
    procs = [ctx.Process(target=worker or _worker, args=(a, f" w{k:02d}")) for k, a in enumerate(shards)]
    for proc in procs:
        proc.start()
    print(f"[gen] 워커 {n_workers}개 시작 — 에피소드 {counts}", flush=True)
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        print("\n[gen] 중단됨 — 워커 정리를 기다린 뒤 저장된 에피소드만 합칩니다.", flush=True)
        for proc in procs:
            proc.join()

    done = [a for a in shards if _existing_episode_count(Path(a.root)) > 0]
    if not done:
        sys.exit("[gen] 저장된 에피소드가 없습니다.")
    print(f"[gen] 조각 {len(done)}개 합치는 중 → {root}", flush=True)
    aggregate_datasets(
        repo_ids=[a.repo_id for a in done], aggr_repo_id=args.repo_id,
        roots=[Path(a.root) for a in done], aggr_root=root,
    )

    # aggregate_datasets는 조각 순서대로 에피소드 번호를 이어 붙인다 — 생성 로그도 같은 순서로 번호를 다시 매김.
    offset, balance = 0, {}
    with (root / "meta" / "scripted_params.jsonl").open("w") as out:
        for a in done:
            meta = Path(a.root) / "meta"
            for line in (meta / "scripted_params.jsonl").read_text().splitlines():
                info = json.loads(line)
                info["episode_index"] += offset
                info["shard_seed"] = a.seed
                out.write(json.dumps(info) + "\n")
            for key, c in json.loads((meta / "scene_balance.json").read_text()).items():
                balance[key] = balance.get(key, 0) + c
            offset += _existing_episode_count(Path(a.root))
    (root / "meta" / "scene_balance.json").write_text(json.dumps(balance, indent=2, sort_keys=True))
    shutil.rmtree(shard_dir)
    print(f"[gen] 완료: 에피소드 {_existing_episode_count(root)}개, {time.perf_counter() - t_start:.0f}s — {root}")


def main() -> None:
    args = parse_args()
    if args.workers > 1:
        generate_parallel(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
