"""lerobot 원본(관절공간) 녹화 데이터셋 -> BC 학습용 EEF-delta + seam CV 특징 데이터셋.

`lerobot-record --robot.type=so101_follower --teleop.type=so101_leader --dataset.fps=30 ...`로
녹화한 LeRobotDataset(관절공간, degree 단위)을 감싸서, __getitem__ 시점에:
  1. observation.state 관절각 -> tcp_link 절대 pose -> observation.state (9,) [xyz + rot6d]
  2. action 관절각 청크 -> tcp_link 증분 EEF-delta (chunk, 7) [dxyz, drotvec, gripper]
  3. 이미지 -> seam_cv 로컬 경로 특징 (5,) -> observation.environment_state
로 바꿔 ai_layer/configs/so101_act_bc.py의 ACTConfig 입출력 스펙에 맞는 배치를 만든다.

시간 정렬 (ActionChunk 규약과 일치):
  - observation.state = 시점 t의 팔로워 실제 pose.
  - action[k]        = 시점 t + (k+1)·dt 의 리더 명령 pose. (delta_timestamps를 dt부터 시작)
  - delta[0]         = state(t) -> action(t+dt). 즉 "지금 위치에서 33ms 뒤 목표까지".
    (action(t) − state(t) 는 추종 오차라서 쓰지 않는다.)

가정 (실제 녹화 데이터로 최초 실행 시 반드시 확인할 것):
  - `observation.state`, `action`이 JOINT_NAMES(kinematics.py) 순서로 정렬된 (6,) 텐서.
    lerobot 0.4.x so101_follower는 기본 use_degrees=True → 관절 5개는 degree.
    메타(`dataset.meta.features["observation.state"]["names"]`)로 실제 순서를 확인한다.
  - gripper 채널(7번째)은 녹화값을 그대로 둔다 (변환 없음). 실로봇 `lerobot-record`로 딴
    데이터는 RANGE_0_100(0~100)이지만, 2026-09-23부터 그리퍼 조는 구동하지 않고 설계문서
    2절 gripper_signal[0/1]을 그대로 쓰기로 해서 `record_mujoco.py`가 만드는 MuJoCo 미러
    데이터셋은 이미 이진(0.0/1.0)이다 — 소스가 섞이면 이 채널의 값 범위가 다르니 학습 전에
    `check_dataset.py` 등으로 확인할 것.
  - 녹화 fps == round(1/DT_AI_SEC) 이어야 한다 (기본 30). 다르면 생성자에서 오류.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ai_layer.configs.so101_act_bc import (
    ACTION_DIM,
    CHUNK_SIZE,
    DT_AI_SEC,
    IMAGE_KEY,
    SEAM_FEATURE_DIM,
    STATE_DIM,
    ZERO_YAW,
)
from ai_layer.kinematics import (
    JOINT_NAMES,
    build_kinematics,
    eef_pose_traj_to_delta_traj,
    joint_traj_to_eef_pose_traj,
    pose_to_state,
    zero_yaw,
)
from ai_layer.perception.seam_cv import SeamGrooveDetector
from ai_layer.perception.seam_features import seam_features_from_chw

# ImageNet 통계 (torchvision resnet18 사전학습 가중치 기준). 이미지 정규화에 사용.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class SO101BCDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        camera_key: str = IMAGE_KEY,
        chunk_size: int = CHUNK_SIZE,
        dt_ai_sec: float = DT_AI_SEC,
        precompute_seam: bool = True,
    ):
        # action[k] = t + (k+1)·dt. (k=0이 dt 뒤 명령이 되도록 1부터 시작)
        delta_timestamps = {
            ACTION: [(k + 1) * dt_ai_sec for k in range(chunk_size)],
        }
        self.raw = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)

        expected_fps = int(round(1.0 / dt_ai_sec))
        if int(self.raw.fps) != expected_fps:
            raise ValueError(
                f"데이터셋 fps={self.raw.fps} 인데 DT_AI_SEC={dt_ai_sec:.4f}s 는 fps={expected_fps}를 요구한다. "
                f"녹화 시 --dataset.fps={expected_fps} 로 다시 녹화하거나 DT_AI_SEC를 맞출 것."
            )
        if camera_key not in self.raw.meta.features:
            raise KeyError(
                f"이미지 키 {camera_key!r} 가 데이터셋에 없다. 있는 키: "
                f"{[k for k in self.raw.meta.features if k.startswith('observation.images')]}"
            )
        self._check_joint_order()

        self.camera_key = camera_key
        self.kin = build_kinematics()
        self.seam = SeamGrooveDetector()
        self.chunk_size = chunk_size

        # seam 특징은 이미지 디코딩 + 파이썬 CV라 가장 비싸다. 프레임별로 한 번만 계산해 캐시.
        self._seam_cache: np.ndarray | None = None
        if precompute_seam:
            self.precompute_seam_features()

    # ------------------------------------------------------------------ 검증
    def _check_joint_order(self) -> None:
        """녹화 메타의 관절 이름 순서가 JOINT_NAMES와 같은지 확인. 다르면 명확히 실패."""
        for key in (OBS_STATE, ACTION):
            names = self.raw.meta.features[key].get("names")
            if not names:
                continue
            stripped = [n.split(".")[0] for n in names]  # "shoulder_pan.pos" -> "shoulder_pan"
            if stripped != JOINT_NAMES:
                raise ValueError(
                    f"{key} 관절 순서 {stripped} 가 JOINT_NAMES {JOINT_NAMES} 와 다르다. "
                    "REORDER_INDEX 보정을 추가하기 전엔 학습하지 말 것."
                )

    def __len__(self) -> int:
        return len(self.raw)

    # ------------------------------------------------------------------ 변환
    def _to_eef_delta(
        self, joint_deg_seq: np.ndarray, anchor_pose: np.ndarray | None = None
    ) -> np.ndarray:
        """(T, 6) 관절각(deg) -> (T, 7) EEF-delta + gripper.

        anchor_pose(관측 시점 EEF)가 있으면 delta[0] = anchor -> 첫 미래 명령 pose.
        그리퍼는 마지막 채널 그대로 둔다(0~100).
        """
        poses = joint_traj_to_eef_pose_traj(self.kin, joint_deg_seq)
        deltas = eef_pose_traj_to_delta_traj(poses, anchor_pose=anchor_pose)
        if ZERO_YAW:
            deltas = zero_yaw(deltas)  # 5자유도 실로봇: yaw 증분은 0
        gripper = joint_deg_seq[:, -1:]  # JOINT_NAMES 마지막 = gripper
        return np.concatenate([deltas, gripper], axis=1)

    def _seam_features(self, image_chw: torch.Tensor) -> np.ndarray:
        """CHW float[0,1] 이미지 -> (5,) seam 특징. 추론(control_bridge)과 같은 함수를 쓴다."""
        return seam_features_from_chw(self.seam, image_chw)

    def precompute_seam_features(self, log_every: int = 500) -> np.ndarray:
        """전 프레임 seam 특징을 한 번 계산해 (N, 5)로 캐시. 학습 중 __getitem__은 조회만 한다."""
        n = len(self.raw)
        feats = np.zeros((n, SEAM_FEATURE_DIM), dtype=np.float32)
        # 이미지 한 장만 빠르게 꺼내기 위해 delta_timestamps 없는 별도 뷰를 쓴다.
        plain = LeRobotDataset(self.raw.repo_id, root=self.raw.root)
        for i in range(n):
            feats[i] = self._seam_features(plain[i][self.camera_key])
            if log_every and i % log_every == 0:
                print(f"[seam precompute] {i}/{n}")
        self._seam_cache = feats
        return feats

    # ------------------------------------------------------------------ 통계
    def compute_stats(self, max_samples: int | None = None, seed: int = 0) -> dict[str, dict[str, torch.Tensor]]:
        """정규화 통계 (lerobot processor 형식). 변환 후 값(EEF pose/delta/seam)으로 계산한다.

        원본 데이터셋 meta.stats는 관절각 기준이라 쓸 수 없다. 이미지는 ImageNet 통계를 쓴다
        (사전학습 resnet18과 일치, 값은 (3,1,1)로 브로드캐스트).
        """
        n = len(self)
        idx = np.arange(n)
        if max_samples is not None and n > max_samples:
            idx = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
        states = np.zeros((len(idx), STATE_DIM), dtype=np.float64)
        envs = np.zeros((len(idx), SEAM_FEATURE_DIM), dtype=np.float64)
        actions = []
        for j, i in enumerate(idx):
            item = self[int(i)]
            states[j] = item[OBS_STATE].numpy()
            envs[j] = item[OBS_ENV_STATE].numpy()
            valid = ~item["action_is_pad"].numpy()
            actions.append(item[ACTION].numpy()[valid])
        actions = np.concatenate(actions, axis=0) if actions else np.zeros((1, ACTION_DIM))

        def ms(x: np.ndarray) -> dict[str, torch.Tensor]:
            return {
                "mean": torch.from_numpy(x.mean(axis=0).astype(np.float32)),
                "std": torch.from_numpy(x.std(axis=0).astype(np.float32)),
            }

        stats = {
            OBS_STATE: ms(states),
            OBS_ENV_STATE: ms(envs),
            ACTION: ms(actions),
            self.camera_key: {
                "mean": torch.from_numpy(IMAGENET_MEAN.copy()),
                "std": torch.from_numpy(IMAGENET_STD.copy()),
            },
        }
        return stats

    # ------------------------------------------------------------------ 배치
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.raw[idx]

        # observation.state: 시점 t 팔로워 관절각 (6,) -> tcp_link 절대 pose -> (9,) [xyz + rot6d]
        state_joint_deg = item[OBS_STATE].numpy().reshape(1, -1)
        state_pose = joint_traj_to_eef_pose_traj(self.kin, state_joint_deg)[0]  # (6,) xyz+rotvec
        state_vec = pose_to_state(state_pose).astype(np.float32)

        # action: (chunk_size, 6) 관절각, k번째 = t+(k+1)dt 명령 -> (chunk_size, 7) 증분
        action_joint_deg = item[ACTION].numpy().reshape(self.chunk_size, -1)
        action_eef_delta = self._to_eef_delta(action_joint_deg, anchor_pose=state_pose)

        if self._seam_cache is not None:
            seam_feat = self._seam_cache[idx]
        else:
            seam_feat = self._seam_features(item[self.camera_key])

        return {
            self.camera_key: item[self.camera_key],
            OBS_STATE: torch.from_numpy(state_vec),
            OBS_ENV_STATE: torch.from_numpy(np.asarray(seam_feat, dtype=np.float32)),
            ACTION: torch.from_numpy(action_eef_delta.astype(np.float32)),
            # LeRobotDataset이 delta_timestamps 청크에 채워주는 패딩 마스크
            # (에피소드 끝에서 미래 프레임이 없을 때 True) — ACTPolicy.forward()가 요구.
            "action_is_pad": item["action_is_pad"],
        }
