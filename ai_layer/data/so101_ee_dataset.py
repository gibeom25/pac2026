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

2026-10-08: 고정(overview) 카메라 드롭아웃 — configs/so101_act_bc.OVERVIEW_IMAGE_KEY 참고.
`use_overview=True`면 항상 overview 이미지 칸을 채워서 돌려준다: 데이터셋에 그 키가 없으면(시뮬)
검은 화면(0), 있으면(실물) `overview_dropout` 확률로 0, 아니면 실제 이미지.
실물 고정 카메라는 바닥의 작은 거치대에 놓여 있어 부스마다/세션마다 조금씩 틀어질 수 있으므로,
실제 이미지를 쓸 때는 작은 무작위 이동/회전/확대(OVERVIEW_JITTER)를 준다.

2026-10-08(속도): action chunk를 LeRobotDataset delta_timestamps로 받으면 lerobot이 chunk 32프레임의
행 전체를 읽어서 쓰지도 않는 이미지 32장을 매 샘플 디코딩한다(샘플당 ~120ms, 1000 에피소드
학습이 데이터 로딩에 묶임). 그래서 action/episode_index 열만 처음에 메모리로 올려 두고 chunk를
여기서 직접 자른다 — 의미(t+1..t+chunk, 에피소드 끝을 넘으면 마지막 값 반복 + action_is_pad)는
delta_timestamps와 같다. seam 특징 사전계산도 DataLoader 워커로 병렬화하고, 결과를 데이터셋 폴더
cache/에 저장해 다음 학습 때 다시 계산하지 않는다(seam CV 코드가 바뀌면 파일명이 바뀌어 새로 계산).

2026-10-09(평가): `exclude_episodes`/`only_episodes`로 에피소드 일부만 보이게 할 수 있다(실물 시연 일부를
학습에서 빼 두고 tools/eval_bc_offline.py로 평가하기 위함). 행 번호는 원본 그대로 두고 보이는 행만
인덱스로 고르므로 seam 캐시와 chunk 자르기는 그대로 쓴다. `augment=False`면 고정 카메라 jitter를 끈다.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from ai_layer.configs.so101_act_bc import (
    ACTION_DIM,
    CHUNK_SIZE,
    DT_AI_SEC,
    IMAGE_KEY,
    OVERVIEW_IMAGE_KEY,
    SEAM_FEATURE_DIM,
    STATE_DIM,
    ZERO_YAW,
)
from ai_layer.kinematics import YAW_INDEX
from ai_layer.perception import seam_cv, seam_features
from ai_layer.perception.seam_cv import SeamGrooveDetector
from ai_layer.perception.seam_features import seam_features_from_chw

# 고정 카메라 흔들림 근사: 최대 이동(이미지 크기 비율), 회전(도), 확대 범위
OVERVIEW_JITTER = {"translate": 0.04, "degrees": 3.0, "scale": (0.95, 1.05)}

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
        use_overview: bool = False,
        overview_dropout: float = 0.3,
        exclude_episodes: list[int] | None = None,
        only_episodes: list[int] | None = None,
        augment: bool = True,
    ):
        self.xyz_only = xyz_only
        self.use_overview = use_overview
        self.overview_dropout = overview_dropout
        self.augment = augment
        # delta_timestamps 없이 연다 — chunk는 _action_chunk가 메모리의 action 배열에서 자른다(모듈 docstring)
        self.raw = LeRobotDataset(repo_id, root=root)

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
        self.has_overview = OVERVIEW_IMAGE_KEY in self.raw.meta.features
        self.seam = SeamGrooveDetector()
        self.chunk_size = chunk_size

        cols = self.raw.hf_dataset.select_columns([ACTION, "episode_index"]).with_format("numpy")[:]
        self._actions = np.asarray(np.stack(cols[ACTION]), dtype=np.float32)  # (N, ACTION_DIM)
        ep = np.asarray(cols["episode_index"])
        if np.any(np.diff(ep) < 0):
            raise ValueError("episode_index가 행 순서대로 정렬돼 있지 않다 — chunk를 자를 수 없음.")
        # 각 행이 속한 에피소드의 마지막 행 + 1
        change = np.flatnonzero(np.diff(ep)) + 1
        ends = np.append(change, len(ep))
        self._ep_end = np.repeat(ends, np.diff(np.concatenate([[0], ends])))
        self.episode_index = ep
        keep = np.ones(len(ep), dtype=bool)
        if only_episodes is not None:
            keep &= np.isin(ep, only_episodes)
        if exclude_episodes:
            keep &= ~np.isin(ep, exclude_episodes)
        self._index = np.flatnonzero(keep)  # 이 Dataset의 i번째 샘플 = 원본 행 _index[i]

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
        return len(self._index)

    def _seam_features(self, image_chw: torch.Tensor) -> np.ndarray:
        return seam_features_from_chw(self.seam, image_chw)

    def _seam_cache_path(self) -> Path:
        h = hashlib.sha1()
        for mod in (seam_cv, seam_features):
            h.update(Path(mod.__file__).read_bytes())
        key = self.camera_key.split(".")[-1]
        return Path(self.raw.root) / "cache" / f"seam_{key}_{len(self.raw)}_{h.hexdigest()[:10]}.npy"

    def precompute_seam_features(self, num_workers: int | None = None, use_cache: bool = True) -> np.ndarray:
        n = len(self.raw)
        path = self._seam_cache_path()
        if use_cache and path.exists():
            self._seam_cache = np.load(path)
            print(f"[seam precompute] 캐시 사용: {path}")
            return self._seam_cache

        workers = num_workers if num_workers is not None else min(24, os.cpu_count() or 1)
        loader = DataLoader(
            _SeamFeatureSource(self.raw, self.camera_key), batch_size=256, num_workers=workers,
            collate_fn=lambda b: np.stack(b),
        )
        feats = np.zeros((n, SEAM_FEATURE_DIM), dtype=np.float32)
        done = 0
        for batch in loader:
            feats[done:done + len(batch)] = batch
            done += len(batch)
            if done // 256 % 100 == 0 or done == n:
                print(f"[seam precompute] {done}/{n} (워커 {workers})", flush=True)
        self._seam_cache = feats
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, feats)
        return feats

    def _action_chunk(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """action[t+1 .. t+chunk] — 에피소드 끝을 넘는 칸은 마지막 값으로 채우고 pad로 표시."""
        q = idx + 1 + np.arange(self.chunk_size)
        end = self._ep_end[idx]
        is_pad = q >= end
        action = torch.from_numpy(self._actions[np.minimum(q, end - 1)])
        return action, torch.from_numpy(is_pad)

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

        stats = {
            OBS_STATE: ms(states),
            OBS_ENV_STATE: ms(envs),
            ACTION: ms(actions),
        }
        for key in [self.camera_key] + ([OVERVIEW_IMAGE_KEY] if self.use_overview else []):
            stats[key] = {
                "mean": torch.from_numpy(IMAGENET_MEAN.copy()),
                "std": torch.from_numpy(IMAGENET_STD.copy()),
            }
        return stats

    def _overview_image(self, item: dict, like: torch.Tensor) -> torch.Tensor:
        # torch.rand — DataLoader 워커마다 torch 시드는 따로 잡히지만 numpy는 같은 시드로 복제된다
        if self.has_overview and torch.rand(()).item() >= self.overview_dropout:
            img = item[OVERVIEW_IMAGE_KEY]
            return self._jitter(img) if self.augment else img
        return torch.zeros_like(like)

    @staticmethod
    def _jitter(img: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.v2 import functional as F

        _, h, w = img.shape
        j = OVERVIEW_JITTER
        r = lambda lo, hi: lo + (hi - lo) * torch.rand(()).item()  # noqa: E731
        return F.affine(
            img,
            angle=r(-j["degrees"], j["degrees"]),
            translate=[round(r(-j["translate"], j["translate"]) * w), round(r(-j["translate"], j["translate"]) * h)],
            scale=r(*j["scale"]),
            shear=[0.0, 0.0],
        )

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self._index[i])
        item = self.raw[idx]

        if self._seam_cache is not None:
            seam_feat = self._seam_cache[idx]
        else:
            seam_feat = self._seam_features(item[self.camera_key])

        action, is_pad = self._action_chunk(idx)
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

        out = {
            self.camera_key: item[self.camera_key],
            OBS_STATE: item[OBS_STATE].float(),
            OBS_ENV_STATE: torch.from_numpy(np.asarray(seam_feat, dtype=np.float32)),
            ACTION: action,
            "action_is_pad": is_pad,
        }
        if self.use_overview:
            out[OVERVIEW_IMAGE_KEY] = self._overview_image(item, item[self.camera_key])
        return out


class _SeamFeatureSource(Dataset):
    """seam 특징 사전계산용 — 이미지 한 장만 디코딩해서 (5,) 특징으로 바꾼다 (DataLoader 워커에서 병렬)."""

    def __init__(self, raw: LeRobotDataset, camera_key: str):
        self.raw = raw
        self.camera_key = camera_key
        self.detector: SeamGrooveDetector | None = None

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, i: int) -> np.ndarray:
        if self.detector is None:
            self.detector = SeamGrooveDetector()
        img = self.raw.hf_dataset[i][self.camera_key]
        return seam_features_from_chw(self.detector, img)
