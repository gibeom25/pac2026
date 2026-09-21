"""BC 경로 끝-끝 실행 검증 (가짜 데이터셋). 실로봇 녹화 없이 코드가 실제로 도는지 본다.

  1. LeRobotDataset 형식(so101_follower 녹화와 같은 키/단위)으로 작은 가짜 데이터셋 생성
     - 관절 5개 degree + gripper 0~100, 이미지 240x320 에 검은 선 하나 (seam_cv가 잡을 것)
  2. SO101BCDataset 로드 → 변환 결과 shape/값 확인
  3. train_bc.py 와 같은 절차로 몇 스텝 학습 → 체크포인트 저장
  4. bc_inference.load_bc_checkpoint 로 복원 → 청크 예측 → 단위/shape 확인

실행: cd pac2026-team && PYTHONPATH=. /home/dy/pac2026/env_lerobot/bin/python ai_layer/tools/bc_synthetic_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ai_layer.bc_inference import load_bc_checkpoint, predict_chunk
from ai_layer.configs.so101_act_bc import ACTION_DIM, CHUNK_SIZE, IMAGE_KEY, STATE_DIM, build_so101_act_config
from ai_layer.data.so101_bc_dataset import SO101BCDataset
from ai_layer.kinematics import JOINT_NAMES
from ai_layer.train_bc import save_checkpoint

FPS = 30
H, W = 240, 320


def make_fake_dataset(root: Path, episodes: int = 2, frames: int = 70) -> str:
    repo_id = "local/so101_fake"
    names = [f"{j}.pos" for j in JOINT_NAMES]
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (6,), "names": names},
        ACTION: {"dtype": "float32", "shape": (6,), "names": names},
        IMAGE_KEY: {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channels"]},
    }
    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features, root=root, use_videos=False)
    rng = np.random.default_rng(0)
    for ep in range(episodes):
        phase = rng.uniform(0, np.pi, size=5)
        for k in range(frames):
            t = k / FPS
            arm_cmd = 15.0 * np.sin(0.8 * t + phase)  # deg, 리더 명령
            arm_obs = arm_cmd - 1.0  # 팔로워는 살짝 뒤처짐 (추종 오차)
            grip = 50.0 + 40.0 * np.sin(0.5 * t)  # 0~100
            state = np.concatenate([arm_obs, [grip - 1.0]]).astype(np.float32)
            action = np.concatenate([arm_cmd, [grip]]).astype(np.float32)
            img = np.full((H, W, 3), 230, dtype=np.uint8)
            x0 = int(40 + 20 * np.sin(0.3 * t))
            cv2.line(img, (x0, 20), (x0 + 200, 210), (20, 20, 20), thickness=4)
            ds.add_frame({OBS_STATE: state, ACTION: action, IMAGE_KEY: img, "task": "follow line"})
        ds.save_episode()
    ds.finalize()
    return repo_id


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="so101_bc_synth_"))
    try:
        root = tmp / "dataset"
        repo_id = make_fake_dataset(root)
        print(f"[1] fake dataset at {root}")

        dataset = SO101BCDataset(repo_id=repo_id, root=root, precompute_seam=True)
        item = dataset[5]
        print(f"[2] item keys={list(item)} state={tuple(item[OBS_STATE].shape)} "
              f"action={tuple(item[ACTION].shape)} env={tuple(item[OBS_ENV_STATE].shape)} img={tuple(item[IMAGE_KEY].shape)}")
        assert item[OBS_STATE].shape == (STATE_DIM,)
        assert item[ACTION].shape == (CHUNK_SIZE, ACTION_DIM)
        assert item[ACTION][:, 5].abs().max() == 0, "yaw 고정 실패"
        assert (item[ACTION][:, 6] >= 0).all() and (item[ACTION][:, 6] <= 100).all(), "그리퍼 0~100 아님"
        assert item[OBS_ENV_STATE].abs().sum() > 0, "seam 특징이 전부 0 (선 인식 실패)"
        # rot6d 두 열이 단위 벡터인지
        r = item[OBS_STATE][3:9]
        assert abs(r[:3].norm() - 1) < 1e-4 and abs(r[3:].norm() - 1) < 1e-4
        print(f"    first delta (t->t+dt) = {item[ACTION][0, :3].numpy()}  seam={item[OBS_ENV_STATE].numpy()}")
        # 에피소드 마지막 프레임: 패딩 마스크가 대부분 True 여야 함
        last = dataset[69]
        assert last["action_is_pad"].sum() >= CHUNK_SIZE - 1, "패딩 마스크 이상"

        stats = dataset.compute_stats(max_samples=64)
        print(f"[3] stats keys={list(stats)} action mean={stats[ACTION]['mean'].numpy().round(4)}")
        assert stats[IMAGE_KEY]["mean"].shape == (3, 1, 1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = build_so101_act_config()
        cfg.device = device
        pre, post = make_act_pre_post_processors(cfg, dataset_stats=stats)
        policy = ACTPolicy(cfg).to(device)
        policy.train()
        optimizer = cfg.get_optimizer_preset().build(policy.parameters())
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
        losses = []
        for step, batch in enumerate(loader):
            batch = pre(batch)
            assert batch[ACTION].abs().mean() < 5, "정규화가 안 된 것 같음 (액션 크기)"
            loss, loss_dict = policy.forward(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step >= 5:
                break
        print(f"[4] train steps={len(losses)} loss first={losses[0]:.4f} last={losses[-1]:.4f}")

        ckpt = save_checkpoint(tmp / "out", "last", policy, pre, post, {"note": "synthetic"})
        files = sorted(p.name for p in ckpt.iterdir())
        print(f"[5] checkpoint files: {files}")
        assert "model.safetensors" in files and "policy_preprocessor.json" in files

        policy2, pre2, post2 = load_bc_checkpoint(ckpt, device=device)
        obs = {k: item[k] for k in (IMAGE_KEY, OBS_STATE, OBS_ENV_STATE)}
        chunk = predict_chunk(policy2, pre2, post2, obs)
        print(f"[6] predicted chunk shape={tuple(chunk.shape)} step0={chunk[0, 0].numpy().round(4)}")
        assert chunk.shape == (1, CHUNK_SIZE, ACTION_DIM)
        # 비정규화 확인: 그리퍼 채널은 0~100 근처, 위치는 미터 단위(수 cm 이하)
        assert 0 <= chunk[0, :, 6].mean() <= 100
        assert chunk[0, :, :3].abs().max() < 0.5
        print("BC SYNTHETIC TEST OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
