"""record_mujoco.py가 만든 EE-native LeRobotDataset -> BC(ACT) 학습용 배치.

so101_bc_dataset.SO101BCDataset은 "실물 lerobot-record로 관절공간(degree)에 녹화된 데이터셋"을
전제로 관절각 -> FK -> EEF pose 변환을 거친다. record_mujoco.py(2026-09-23 EE-only 리그 전환
이후)로 모은 데이터셋은 애초에 관절이 없고 observation.state/action을 EE-native로 직접 기록한다
(record_mujoco.py 모듈 docstring 참고):
  observation.state (9,) = [x, y, z, rot6d(6)]        -- kinematics.pose_to_state와 이미 동일 표현
  action (7,)            = [dx, dy, dz, drx, dry, drz, gripper]  -- 직전 프레임 대비 증분, 이미 계산됨
그래서 이 데이터셋은 SO101BCDataset처럼 FK/델타 재계산을 할 필요가 없다 — LeRobotDataset이
delta_timestamps로 채워주는 chunk를 그대로 꺼내 쓰면 된다. record_mujoco.py의 action[i]는
"pose(i-1) -> pose(i)의 증분"으로 저장되므로, 현재 프레임 t 기준 상대시각 (k+1)*dt에서 가져오는
action은 정확히 프레임 t+k+1의 저장값 = "pose(t+k) -> pose(t+k+1) 증분" = ACTConfig가 기대하는
chunk[k](=t+(k+1)dt 시점 명령)와 그대로 일치한다 (SO101BCDataset의 anchor_pose 재계산이 필요한
이유는 실로봇에서 "리더 명령"과 "팔로워 실측"이 서로 다른 소스라 재기준이 필요했기 때문 — 이
데이터셋은 state/action이 같은 실측 궤적에서 나와 애초에 어긋날 일이 없다).

2026-10-06(기범): BC 학습 타깃에서 yaw(action의 drz)만 0으로 마스킹한다(ZERO_YAW, 기존
so101_act_bc.py/so101_bc_dataset.py의 실로봇 경로와 동일 결정 — "로봇마다 다른 구조를
반영해야 하는 건 사실 yaw뿐" 참고: 실로봇 IK가 5D(XYZ+roll/pitch)만 풀어서 yaw는 애초에 안
받아들여지지만, roll/pitch는 실제로 쓰이는 자유도라 — 특히 코너/분기 구간에서 도포 각도가
중요할 수 있어서 — 남겨둔다). observation.state(9,)의 rot6d는 그대로 6DOF를 담는다(관측에는
yaw도 들어감 — 녹화 당시 실제 궤적을 복원 가능하게). 녹화 원본(raw, LeRobotDataset)에는 yaw가
안 지워진 채 그대로 남아 다른 로봇/후속 분석에 쓸 수 있다.

2026-10-06(2차): `xyz_only=True`면 roll/pitch까지 마저 0으로 마스킹해서(drx/dry/drz 전부) 4DOF
(xyz+그리퍼)짜리 가벼운 MVP를 먼저 학습할 수 있게 했다 — "녹화는 지금처럼 6DOF 다 하고 학습
때만 끄자"는 결정(기범): 녹화(조이스틱 티칭)는 되돌릴 수 없는 비용이라 지금 깎으면 나중에 rpy가
필요할 때 데이터를 통째로 다시 모아야 하지만, 학습 타깃 마스킹은 플래그 하나라 공짜로 되돌릴 수
있다. 그래서 녹화 쪽(record_mujoco.py)은 전혀 안 건드리고 여기서만 선택적으로 더 깎는다.
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
from ai_layer.kinematics import YAW_INDEX
from ai_layer.perception.seam_cv import SeamGrooveDetector
from ai_layer.perception.seam_features import seam_features_from_chw

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class SO101EEDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        camera_key: str = IMAGE_KEY,
        chunk_size: int = CHUNK_SIZE,
        dt_ai_sec: float = DT_AI_SEC,
        precompute_seam: bool = True,
        xyz_only: bool = False,
    ):
        self.xyz_only = xyz_only
        delta_timestamps = {
            ACTION: [(k + 1) * dt_ai_sec for k in range(chunk_size)],
        }
        self.raw = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)

        expected_fps = int(round(1.0 / dt_ai_sec))
        if int(self.raw.fps) != expected_fps:
            raise ValueError(
                f"데이터셋 fps={self.raw.fps} 인데 DT_AI_SEC={dt_ai_sec:.4f}s 는 fps={expected_fps}를 요구한다. "
                f"record_mujoco.py --fps={expected_fps} 로 다시 녹화하거나 DT_AI_SEC를 맞출 것."
            )
        if camera_key not in self.raw.meta.features:
            raise KeyError(
                f"이미지 키 {camera_key!r} 가 데이터셋에 없다. 있는 키: "
                f"{[k for k in self.raw.meta.features if k.startswith('observation.images')]}"
            )
        self._check_shapes()

        self.camera_key = camera_key
        self.seam = SeamGrooveDetector()
        self.chunk_size = chunk_size

        self._seam_cache: np.ndarray | None = None
        if precompute_seam:
            self.precompute_seam_features()

    def _check_shapes(self) -> None:
        """record_mujoco.py의 EE-native 포맷(9/7차원 벡터, 관절 이름 없음)인지 확인."""
        state_shape = tuple(self.raw.meta.features[OBS_STATE]["shape"])
        action_shape = tuple(self.raw.meta.features[ACTION]["shape"])
        if state_shape != (STATE_DIM,) or action_shape != (ACTION_DIM,):
            raise ValueError(
                f"observation.state shape={state_shape} action shape={action_shape} 가 EE-native 스펙"
                f"(state=({STATE_DIM},), action=({ACTION_DIM},))과 다르다 — 관절공간 데이터셋이면 "
                "SO101BCDataset을 쓸 것 (ai_layer/data/so101_bc_dataset.py)."
            )

    def __len__(self) -> int:
        return len(self.raw)

    def _seam_features(self, image_chw: torch.Tensor) -> np.ndarray:
        return seam_features_from_chw(self.seam, image_chw)

    def precompute_seam_features(self, log_every: int = 500) -> np.ndarray:
        n = len(self.raw)
        feats = np.zeros((n, SEAM_FEATURE_DIM), dtype=np.float32)
        plain = LeRobotDataset(self.raw.repo_id, root=self.raw.root)
        for i in range(n):
            feats[i] = self._seam_features(plain[i][self.camera_key])
            if log_every and i % log_every == 0:
                print(f"[seam precompute] {i}/{n}")
        self._seam_cache = feats
        return feats

    def compute_stats(self, max_samples: int | None = None, seed: int = 0) -> dict[str, dict[str, torch.Tensor]]:
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

        return {
            OBS_STATE: ms(states),
            OBS_ENV_STATE: ms(envs),
            ACTION: ms(actions),
            self.camera_key: {
                "mean": torch.from_numpy(IMAGENET_MEAN.copy()),
                "std": torch.from_numpy(IMAGENET_STD.copy()),
            },
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.raw[idx]

        if self._seam_cache is not None:
            seam_feat = self._seam_cache[idx]
        else:
            seam_feat = self._seam_features(item[self.camera_key])

        action = item[ACTION].reshape(self.chunk_size, -1).float()
        if self.xyz_only:
            # roll/pitch/yaw 전부 마스킹 — 4DOF(xyz+그리퍼) MVP용. ZERO_YAW보다 우선한다(yaw도
            # 어차피 이 안에 포함).
            action = action.clone()
            action[:, 3:6] = 0.0
        elif ZERO_YAW:
            # 2026-10-06(기범): yaw(drz)는 로봇마다 의미가 달라지는 월드 회전이고(실로봇 IK도
            # 5D라 애초에 안 풀림, so101_act_bc.ZERO_YAW 참고), roll/pitch는 실로봇 IK가 실제로
            # 받아들이는 자유도라 학습 타깃에 남긴다. 녹화된 원본(raw)에는 6DOF 그대로 있으니
            # 다른 로봇/후속 분석에 그 데이터를 그대로 쓸 수 있다 — 여기서 BC 학습 타깃을 만들
            # 때만 마스킹한다.
            action = action.clone()
            action[:, YAW_INDEX] = 0.0

        return {
            self.camera_key: item[self.camera_key],
            OBS_STATE: item[OBS_STATE].float(),
            OBS_ENV_STATE: torch.from_numpy(np.asarray(seam_feat, dtype=np.float32)),
            ACTION: action,
            "action_is_pad": item["action_is_pad"],
        }
