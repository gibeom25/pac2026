#!/usr/bin/env python
"""RL(SAC)로 학습한 정책을 시뮬에서 굴려서(rollout) BC 데이터셋 포맷으로 기록한다.

"RL이 다듬은 동작을 BC(ACT) 가중치에 증류"하는 2단계 흐름의 1단계(2026-10-08, 기범 — "teaching
방식보다 RL fine-tuning이 맞을 것 같다"는 요청에 대한 답: SAC와 ACT가 서로 다른 네트워크라
SAC 가중치를 ACT로 직접 못 옮기므로, 대신 SAC가 시뮬에서 굴린 더 나은 궤적을 이 스크립트로
기록 → train_bc.py --init-checkpoint로 기존 BC 가중치에 지도학습으로 증류). 2단계는:
    PYTHONPATH=. python ai_layer/train_bc.py --repo-id <이 스크립트가 만든 데이터셋> \
        --init-checkpoint <기존 BC 체크포인트> --epochs <적게>

**중요한 주의사항**: so101_seam_env.py(RL 학습용 환경)는 "물리는 모든 scene_a4*.xml이 동일하니
텍스처 아무거나 하나만 대표로 로드"하고(고정 scene_a4.xml, "curve" 텍스처), 매 에피소드
ground-truth 폴리라인은 그와 무관하게 6형태×5variant 중 무작위로 고른다 — RL 정책이 애초에
이미지를 관측으로 안 쓰므로(observation.images.wrist는 0 placeholder) 텍스처가 안 맞아도 학습
자체엔 문제가 없었다. 근데 **여기서는 카메라 이미지를 실제로 기록**해야 하므로, 그대로 가져다
쓰면 "화면엔 항상 curve가 보이는데 행동은 전혀 다른 모양(branch, corner...)을 따라간" 엉터리
(이미지, 행동) 쌍이 기록된다(실제로 발견한 문제). 그래서 에피소드마다 실제로 뽑힌 (형태, variant)
에 맞는 텍스처 MJCF(record_mujoco.py와 동일한 scene_a4_<형태>_<variant>.xml)를 그때그때 새로
로드해서 렌더링한다 — 물리/바디 이름은 모든 scene_a4*.xml이 동일하므로(so101_seam_env.py 주석
확인됨) 안전하다.

물리 적용 방식은 so101_seam_env.py의 step()과 동일하게 맞췄다(SAC가 낸 action이 이미 "이번
스텝 EEF-delta"이므로 — teleop처럼 속도를 적분하는 게 아니라 그대로 target_pos에 더함). 다만
바닥 접촉 시 RL처럼 즉시 종료하지 않고 record_mujoco.py의 높이 클램프(MIN_TIP_Z)를 쓴다 — 이
데이터셋은 "녹화된 시연"으로 쓰일 거라 teleop 데이터와 같은 규약(바닥 뚫지 않고 그 높이에서
버팀)을 따르는 게 일관적이다.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import mujoco
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors
from lerobot.utils.rotation import Rotation

from ai_layer.configs.so101_sac import CONTINUOUS_ACTION_DIM, IMAGE_KEY, build_sac_dataset_stats, build_so101_sac_config
from ai_layer.envs.seam_ground_truth import N_VARIANTS, SCENE_NAMES, SeamGroundTruth
from ai_layer.envs.so101_seam_env import (
    EE_BODY_NAME,
    FLOOR_GEOM_NAME,
    IDENTITY_QUAT,
    MOCAP_BODY_NAME,
    MOCAP_HOME,
    PATH_NUM_POINTS,
    ROD_GEOM_NAME,
    SO101SeamEnvCfg,
    _ee_pose_xyzrotvec,
    _rod_tip_world,
    _rotmat_to_mujoco_quat,
)
from ai_layer.kinematics import pose_delta, pose_to_state
from ai_layer.rl.reward import _point_to_polyline
from ai_layer.tools.episode_ticker import MujocoDualCamera
from ai_layer.tools.record_mujoco import (
    ACTION_KEYS,
    BEAD_STRIDE,
    CAMERA_HW,
    MIN_TIP_Z,
    STATE_KEYS,
    WORKSPACE_X,
    WORKSPACE_Y,
    WORKSPACE_Z,
    BeadDrop,
    _contact_pos,
    _mjcf_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC로 학습한 정책을 시뮬에서 굴려 BC 데이터셋(EE-native)으로 기록한다.")
    p.add_argument("--sac-checkpoint", required=True, help="train_rl.py가 저장한 SAC 체크포인트 폴더")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="비우면 datasets/<repo-id>")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode-seconds", type=float, default=12.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--task", default="weld seam following demo (rl rollout distilled, qt-gui compatible)")
    p.add_argument("--dry-run", action="store_true", help="저장 없이 롤아웃만 돌려보기")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_dataset(args: argparse.Namespace) -> LeRobotDataset:
    hw_obs = {name: float for name in STATE_KEYS}
    hw_obs["wrist"] = CAMERA_HW
    hw_action = {name: float for name in ACTION_KEYS}
    features = {
        **hw_to_dataset_features(hw_obs, "observation", use_video=False),
        **hw_to_dataset_features(hw_action, "action", use_video=False),
    }
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type="so101_ee_mujoco_rl_rollout",
        use_videos=False,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    device = torch.device(args.device)

    sac_cfg = build_so101_sac_config()
    sac_cfg.device = str(device)
    # train_bc.py의 --init-checkpoint와 같은 이유로 config=sac_cfg를 넘긴다 — 안 그러면
    # from_pretrained가 로컬 경로를 HF Hub repo_id로 착각해서 HFValidationError가 난다(실측 확인).
    policy = SACPolicy.from_pretrained(args.sac_checkpoint, config=sac_cfg)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_sac_pre_post_processors(sac_cfg, dataset_stats=build_sac_dataset_stats())

    def norm_obs(obs: dict) -> dict:
        out = preprocessor(dict(obs))
        return {k: v for k, v in out.items() if k.startswith("observation.") and torch.is_tensor(v)}

    seam_gt = SeamGroundTruth(num_points=PATH_NUM_POINTS)
    env_cfg = SO101SeamEnvCfg()
    action_scale = np.array([env_cfg.action_scale_pos] * 3 + [env_cfg.action_scale_rot] * 3)
    dt = 1.0 / args.fps
    physics_dt = env_cfg.physics_dt
    decimation = env_cfg.decimation
    max_steps = int(args.episode_seconds * args.fps)

    dataset = None if args.dry_run else build_dataset(args)
    print("[rl_rollout] --dry-run: 저장 없이 롤아웃만 확인" if args.dry_run else f"[rl_rollout] 데이터셋: {dataset.root}")

    saved = 0
    attempts = 0
    while saved < args.num_episodes:
        attempts += 1
        scene = SCENE_NAMES[random.randrange(len(SCENE_NAMES))]
        variant = random.randrange(N_VARIANTS)
        target_polyline, curvature, thickness = seam_gt.load(scene, variant)

        # RL 학습 때 쓰던 고정 scene_a4.xml이 아니라, 실제 뽑힌 (형태, variant)에 맞는 텍스처
        # MJCF를 로드한다 — 모듈 docstring 참고(이미지-행동 불일치 방지).
        model = mujoco.MjModel.from_xml_path(str(_mjcf_path(scene, variant)))
        data = mujoco.MjData(model)
        model.opt.timestep = physics_dt
        mocap_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MOCAP_BODY_NAME)
        mocap_idx = model.body_mocapid[mocap_bid]
        ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)
        rod_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, ROD_GEOM_NAME)
        floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)

        target_pos = MOCAP_HOME.copy()
        R_cmd = np.eye(3)
        data.mocap_pos[mocap_idx] = target_pos
        data.mocap_quat[mocap_idx] = IDENTITY_QUAT
        mujoco.mj_forward(model, data)

        dual_cam = MujocoDualCamera(model)
        bead_points: list[BeadDrop] = []
        prev_pose: np.ndarray | None = None

        for step in range(max_steps):
            tip = _rod_tip_world(data, rod_gid)
            pt_t = torch.from_numpy(tip).float().unsqueeze(0)
            poly_t = torch.from_numpy(target_polyline).float().unsqueeze(0)
            _, _, seg_idx = _point_to_polyline(pt_t, poly_t)
            lookahead_idx = min(int(seg_idx.item()) + 2, PATH_NUM_POINTS - 1)
            rel = target_polyline[lookahead_idx] - tip
            env_state = np.concatenate([rel, [curvature, thickness]]).astype(np.float32)

            pose = _ee_pose_xyzrotvec(data, ee_bid, rod_gid)
            state9 = pose_to_state(pose).astype(np.float32)
            image_placeholder = np.zeros((3, 240, 320), dtype=np.float32)  # SAC는 관측에 이미지 안 씀
            obs = {
                "observation.state": torch.from_numpy(state9).unsqueeze(0),
                "observation.environment_state": torch.from_numpy(env_state).unsqueeze(0),
                IMAGE_KEY: torch.from_numpy(image_placeholder).unsqueeze(0),
            }

            with torch.no_grad():
                action = policy.select_action(norm_obs(obs))
            action_np = action.squeeze(0).cpu().numpy()
            continuous = np.clip(action_np[:CONTINUOUS_ACTION_DIM], -1.0, 1.0)
            bit = float(action_np[CONTINUOUS_ACTION_DIM] > 0.5)
            delta6 = continuous * action_scale  # so101_seam_env.py step()과 동일 관례

            target_pos = target_pos + delta6[:3]
            target_pos[0] = float(np.clip(target_pos[0], *WORKSPACE_X))
            target_pos[1] = float(np.clip(target_pos[1], *WORKSPACE_Y))
            target_pos[2] = float(np.clip(target_pos[2], *WORKSPACE_Z))
            data.mocap_pos[mocap_idx] = target_pos

            if np.any(delta6[3:6]):
                R_cmd = Rotation.from_rotvec(delta6[3:6]).as_matrix() @ R_cmd
            data.mocap_quat[mocap_idx] = _rotmat_to_mujoco_quat(R_cmd)

            for _ in range(decimation):
                mujoco.mj_step(model, data)

            tip = _rod_tip_world(data, rod_gid)
            if tip[2] < MIN_TIP_Z:
                # record_mujoco.py와 동일 규약 — 바닥을 뚫지 않고 그 높이에서 버틴다(RL의
                # 즉시종료 대신, teleop 데이터와 같은 모양의 "시연"으로 기록하기 위함).
                target_pos[2] += MIN_TIP_Z - tip[2]
            # so101_seam_env.py step()과 동일한 안전 컷오프 — 선에서 off_seam_safety_dist보다
            # 멀면 정책이 트리거를 켜도 실제로는 비활성(2026-10-08: 처음엔 이 게이트 없이
            # action_np[6]>0.5를 그대로 썼는데, 리뷰에서 "RL이 학습 중 이 안전장치 밖에서 트리거를
            # 켤 수 있는 상태를 롤아웃이 그대로 기록하면 distill된 BC가 그 선 밖 도포까지 배운다"는
            # 걸 지적받아 추가함 — RL이 실제로 보상/학습받은 것과 롤아웃 기록이 어긋나면 안 됨).
            dist_to_seam = float(np.linalg.norm(tip[None, :] - target_polyline, axis=1).min())
            bit = bit if dist_to_seam <= env_cfg.off_seam_safety_dist else 0.0
            if bit and step % BEAD_STRIDE == 0:
                bead_points.append(BeadDrop(tip.copy()))
            for b in bead_points:
                b.step(dt)

            wrist_frame = dual_cam.get_wrist_frame(data, bead_points)

            pose = _ee_pose_xyzrotvec(data, ee_bid, rod_gid)
            recorded_delta6 = np.zeros(6) if prev_pose is None else pose_delta(prev_pose, pose)
            prev_pose = pose

            if dataset is not None:
                state9_rec = pose_to_state(pose)
                obs_values = {**dict(zip(STATE_KEYS, state9_rec.tolist())), "wrist": wrist_frame}
                action_values = {**dict(zip(ACTION_KEYS[:6], recorded_delta6.tolist())), "gripper": bit}
                obs_frame = build_dataset_frame(dataset.features, obs_values, prefix="observation")
                action_frame = build_dataset_frame(dataset.features, action_values, prefix="action")
                dataset.add_frame({**obs_frame, **action_frame, "task": args.task})

        dual_cam.close()
        if dataset is not None:
            dataset.save_episode()
        saved += 1
        print(f"[rl_rollout] 에피소드 {saved}/{args.num_episodes} 저장 (scene={scene} variant={variant}, 시도 {attempts}회째)")

    if dataset is not None:
        dataset.finalize()
        print(f"[rl_rollout] 완료 — 데이터셋: {dataset.root}")
    else:
        print("[rl_rollout] dry-run 완료 (저장된 데이터 없음)")


if __name__ == "__main__":
    main()
