"""녹화 직후 데이터셋 점검. 학습 전에 반드시 통과시킬 것.

두 가지 포맷을 자동 판별(ai_layer.data.detect_dataset_kind)해서 검사 항목을 달리한다:
  - "joint" (실물 lerobot-record, 관절공간): 관절 이름/순서, degree 범위, gripper 0~100 검사.
  - "ee" (record_mujoco.py, EE-native, 2026-09-23~): 관절이 없으므로 그 대신 state/action
    차원(9/7)과 gripper 채널(action 마지막)이 0/1 바이너리인지 검사.
둘 다 공통으로 확인하는 것:
  1. fps == 30 (DT_AI_SEC 1/30)
  2. 이미지 키 observation.images.wrist 존재, (H, W, 3)
  3. 에피소드 수/길이, timestamp 가 1/fps 간격인지
  4. 청크 증분 크기: 제어 validator 실효 한계(|dp| 0.005 m/step, |w| 0.05 rad/step) 대비 몇 %가 넘는지
     -> 많이 넘으면 티칭 속도를 줄여서 다시 녹화해야 한다 (넘는 스텝은 다리에서 잘려 궤적이 느려짐)
  5. seam 인식률: 특징이 전부 0인 프레임 비율 (선을 못 본 프레임)

실행: cd pac2026-team && PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/tools/check_dataset.py \
        --repo-id <user>/<name> --root <로컬경로> [--max-samples 300]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from ai_layer.configs.so101_act_bc import ACTION_DIM, CHUNK_SIZE, DT_AI_SEC, IMAGE_KEY, STATE_DIM
from ai_layer.control_bridge.chunk_builder import ChunkLimits
from ai_layer.data import detect_dataset_kind
from ai_layer.kinematics import JOINT_NAMES, URDF_PATH

OK, WARN, FAIL = "✅", "⚠️ ", "❌"


def urdf_joint_limits_deg() -> dict[str, tuple[float, float]]:
    root = ET.parse(URDF_PATH).getroot()
    lim = {}
    for j in root.iter("joint"):
        name = j.get("name")
        if name in JOINT_NAMES[:-1]:
            l = j.find("limit")
            lim[name] = (np.degrees(float(l.get("lower"))), np.degrees(float(l.get("upper"))))
    return lim


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-samples", type=int, default=300, help="6·7번 검사에 쓸 최대 프레임 수")
    args = ap.parse_args()

    problems = 0

    def report(ok: bool, msg: str, warn: bool = False) -> None:
        nonlocal problems
        tag = OK if ok else (WARN if warn else FAIL)
        if not ok and not warn:
            problems += 1
        print(f"{tag} {msg}")

    ds = LeRobotDataset(args.repo_id, root=args.root)
    meta = ds.meta
    kind = detect_dataset_kind(args.repo_id, args.root)
    print(f"dataset: {args.repo_id} root={ds.root} kind={kind}")
    print(f"  episodes={meta.total_episodes} frames={meta.total_frames} fps={meta.fps}")

    # 1. fps
    want_fps = int(round(1 / DT_AI_SEC))
    report(int(meta.fps) == want_fps, f"fps={meta.fps} (필요: {want_fps})")

    hf = ds.hf_dataset
    robot_type = getattr(meta, "robot_type", None) or meta.info.get("robot_type", "")

    if kind == "joint":
        # 2(joint). 관절 순서
        for key in (OBS_STATE, ACTION):
            names = meta.features.get(key, {}).get("names") or []
            stripped = [n.split(".")[0] for n in names]
            report(stripped == JOINT_NAMES, f"{key} 관절 순서 {stripped}")

        # 4(joint). 단위 — 관절 5개는 degree(URDF 한계 안), gripper는 실물 RANGE_0_100
        state = np.stack([np.asarray(x) for x in hf[OBS_STATE]])  # (N, 6)
        lim = urdf_joint_limits_deg()
        for i, jn in enumerate(JOINT_NAMES[:-1]):
            lo, hi = lim[jn]
            mn, mx = float(state[:, i].min()), float(state[:, i].max())
            inside = lo - 5 <= mn and mx <= hi + 5
            looks_deg = mx - mn > 2.0 or abs(mx) > 3.2  # 라디안이면 전부 ±3.14 안
            report(inside and looks_deg, f"{jn}: [{mn:.1f}, {mx:.1f}]  URDF 한계 [{lo:.0f}, {hi:.0f}] deg" +
                   ("" if looks_deg else "  ← 라디안처럼 보임 (use_degrees=true 확인)"))
        g = state[:, -1]
        report(0 <= g.min() and g.max() <= 100, f"gripper: [{g.min():.1f}, {g.max():.1f}] (실물 녹화 기대 0~100)")
    else:
        # 2(ee)/4(ee): record_mujoco.py EE-native — 관절 없음. state/action 차원 + gripper 바이너리만 확인.
        state_shape = tuple(meta.features[OBS_STATE]["shape"])
        action_shape = tuple(meta.features[ACTION]["shape"])
        report(state_shape == (STATE_DIM,), f"observation.state shape={state_shape} (필요: ({STATE_DIM},))")
        report(action_shape == (ACTION_DIM,), f"action shape={action_shape} (필요: ({ACTION_DIM},))")
        action = np.stack([np.asarray(x) for x in hf[ACTION]])  # (N, ACTION_DIM)
        g = action[:, -1]
        g_binary = set(np.unique(np.round(g, 3)).tolist()) <= {0.0, 1.0}
        report(g_binary, f"gripper(action 마지막 채널): unique={sorted(set(g.tolist()))[:6]}... (바이너리 0/1 기대)")
    print(f"   robot_type={robot_type!r} ({kind}) — joint/ee 데이터는 포맷이 달라 섞어 학습하면 안 됨")

    # 5. 에피소드/timestamp
    ep_idx = np.asarray(hf["episode_index"]).reshape(-1)
    ts = np.asarray(hf["timestamp"]).reshape(-1)
    lengths = np.bincount(ep_idx)
    report(lengths.min() >= CHUNK_SIZE * 2, f"에피소드 길이 min={lengths.min()} max={lengths.max()} 프레임 (권장 ≥ {CHUNK_SIZE*2})", warn=True)
    dts = np.diff(ts)[np.diff(ep_idx) == 0]
    bad = np.abs(dts - 1 / meta.fps) > 1e-3
    report(not bad.any(), f"timestamp 간격 이상 {int(bad.sum())}개 (기대 {1/meta.fps*1e3:.1f} ms)")

    # 6·7. 변환 후 검사 (실제 학습 입력을 만들어 본다 — kind에 맞는 Dataset 클래스로)
    from ai_layer.data import load_bc_dataset

    bc = load_bc_dataset(args.repo_id, root=args.root, precompute_seam=False)
    n = len(bc)
    idx = np.linspace(0, n - 1, min(n, args.max_samples)).astype(int)
    pos_lim, rot_lim = ChunkLimits().effective(int(round(DT_AI_SEC * 1e9)))
    over_p = over_r = total = 0
    seam_zero = 0
    for i in idx:
        item = bc[int(i)]
        a = item[ACTION].numpy()
        valid = ~item["action_is_pad"].numpy()
        a = a[valid]
        total += len(a)
        over_p += int((np.linalg.norm(a[:, :3], axis=1) > pos_lim).sum())
        over_r += int((np.linalg.norm(a[:, 3:6], axis=1) > rot_lim).sum())
        if float(np.abs(item["observation.environment_state"].numpy()).sum()) == 0.0:
            seam_zero += 1
    fp, fr = over_p / max(total, 1), over_r / max(total, 1)
    report(fp < 0.05, f"위치 증분 > {pos_lim*1e3:.1f} mm/step 인 스텝 {fp*100:.1f}% (5% 넘으면 티칭을 더 천천히)", warn=fp < 0.2)
    report(fr < 0.05, f"회전 증분 > {rot_lim:.3f} rad/step 인 스텝 {fr*100:.1f}%", warn=fr < 0.2)
    sz = seam_zero / len(idx)
    report(sz < 0.2, f"선 인식 실패 프레임 {sz*100:.1f}% (seam_cv 설정/조명 확인. tools/seam_preview.py 로 튜닝)", warn=sz < 0.5)

    print("\n결과:", "학습 가능" if problems == 0 else f"문제 {problems}개 — 고치기 전엔 학습하지 말 것")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
