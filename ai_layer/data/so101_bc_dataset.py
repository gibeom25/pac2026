"""lerobot 원본(관절공간) 녹화 데이터셋 -> BC 학습용 EEF-delta + seam CV 특징 데이터셋.

`lerobot-record --robot.type=so101_follower --teleop.type=so101_leader ...`로 녹화한
LeRobotDataset(관절공간, degree 단위)을 감싸서, __getitem__ 시점에:
  1. observation.state / action의 관절각(JOINT_NAMES 순서) -> EEF-delta로 변환 (kinematics.py)
  2. 이미지 -> seam_cv로 로컬 경로 특징 추출 -> observation.environment_state
로 바꿔 ai_layer/configs/so101_act_bc.py의 ACTConfig 입출력 스펙에 맞는 배치를 만든다.

가정 (실제 녹화 데이터로 최초 실행 시 반드시 확인할 것):
  - `observation.state`, `action`이 JOINT_NAMES(kinematics.py) 순서로 정렬된 (T, 6) 텐서.
    lerobot 데이터셋 메타(`dataset.meta.features["observation.state"]["names"]`)로 실제
    순서를 확인 후 다르면 REORDER_INDEX로 보정한다.
  - gripper 채널(6번째)은 로봇 액션(pump 신호 아님) 그대로 사용 — pump 신호로 치환하려면
    3.1/3.4 설계에 맞춰 별도 매핑 함수를 이 파일에 추가할 것.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from ai_layer.configs.so101_act_bc import CHUNK_SIZE, DT_AI_SEC, IMAGE_KEY
from ai_layer.kinematics import (
    build_kinematics,
    eef_pose_traj_to_delta_traj,
    joint_traj_to_eef_pose_traj,
)
from ai_layer.perception.seam_cv import SeamGrooveDetector


class SO101BCDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        camera_key: str = "observation.images.wrist",
        chunk_size: int = CHUNK_SIZE,
        dt_ai_sec: float = DT_AI_SEC,
    ):
        delta_timestamps = {
            "action": [t * dt_ai_sec for t in range(chunk_size)],
        }
        self.raw = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
        self.camera_key = camera_key
        self.kin = build_kinematics()
        self.seam = SeamGrooveDetector()
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return len(self.raw)

    def _to_eef_delta(self, joint_deg_seq: np.ndarray) -> np.ndarray:
        """(T, 6) 관절각(deg) -> (T, 7) EEF-delta + gripper. T=1이면 그 프레임의 delta=0."""
        poses = joint_traj_to_eef_pose_traj(self.kin, joint_deg_seq)
        deltas = eef_pose_traj_to_delta_traj(poses)
        gripper = joint_deg_seq[:, -1:]  # JOINT_NAMES 마지막 = gripper
        return np.concatenate([deltas, gripper], axis=1)

    def _seam_features(self, image_chw: torch.Tensor) -> np.ndarray:
        """CHW float[0,1] 이미지 -> (5,) seam 특징: lookahead 상대위치(px, 2) 정규화 + Δrow,Δcol
        방향 성분 대신 곡률/굵기까지 포함한 축약 벡터. depth 없는 2D 전용 모드(3.1 CV 모듈 참고).
        """
        img = (image_chw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h, w = img.shape[:2]
        points = self.seam.detect(img)
        if not points:
            return np.zeros(5, dtype=np.float32)
        target = self.seam.lookahead_target(points, current_idx=0, lookahead=10)
        cur = points[0]
        dr = (target.pixel[0] - cur.pixel[0]) / h
        dc = (target.pixel[1] - cur.pixel[1]) / w
        return np.array(
            [dr, dc, cur.curvature, cur.thickness_px / max(h, w), len(points) / (h * w)],
            dtype=np.float32,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.raw[idx]

        # observation.state: 단일 프레임 (6,) 관절각(deg) -> EEF pose 6dim (delta 없음, 절대 proprioception)
        state_joint_deg = item["observation.state"].numpy().reshape(1, -1)
        state_eef_pose = joint_traj_to_eef_pose_traj(self.kin, state_joint_deg)[0]

        # action: delta_timestamps로 이미 (chunk_size, 6) 관절각 chunk
        action_joint_deg = item["action"].numpy().reshape(self.chunk_size, -1)
        action_eef_delta = self._to_eef_delta(action_joint_deg)

        seam_feat = self._seam_features(item[self.camera_key])

        return {
            IMAGE_KEY: item[self.camera_key],
            "observation.state": torch.from_numpy(state_eef_pose).float(),
            "observation.environment_state": torch.from_numpy(seam_feat).float(),
            "action": torch.from_numpy(action_eef_delta).float(),
            # LeRobotDataset이 delta_timestamps 기반 청크에 자동으로 채워주는 패딩 마스크
            # (에피소드 끝에서 chunk_size만큼 미래 프레임이 없을 때 True) — ACTPolicy.forward()가 요구.
            "action_is_pad": item["action_is_pad"],
        }
