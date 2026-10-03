"""BC 학습용 데이터셋 kind 판별 + 선택.

두 가지 소스가 섞여 있다:
  - record_mujoco.py (2026-09-23~): EE-only 리그, EE-native 포맷. robot_type을
    "so101_ee_mujoco_joystick"로 고정 기록한다 (observation.state/action에 관절 이름 없음).
  - 실물 lerobot-record: 관절공간(degree) 포맷. robot_type에 "ee_mujoco"가 없다.
둘은 observation.state/action의 차원과 의미가 다르므로 서로 다른 Dataset 클래스가 필요하다
(so101_ee_dataset.SO101EEDataset / so101_bc_dataset.SO101BCDataset) — train_bc.py와
tools/check_dataset.py가 이 판별을 공유한다.
"""

from __future__ import annotations

import json
from pathlib import Path


def _resolve_root(repo_id: str, root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / repo_id


def detect_dataset_kind(repo_id: str, root: str | Path | None) -> str:
    """meta/info.json의 robot_type으로 "ee" / "joint"를 구분해 반환."""
    info_path = _resolve_root(repo_id, root) / "meta" / "info.json"
    robot_type = json.loads(info_path.read_text()).get("robot_type", "") if info_path.exists() else ""
    return "ee" if "ee_mujoco" in str(robot_type) else "joint"


def load_bc_dataset(repo_id: str, root: str | Path | None = None, **kwargs):
    """robot_type에 맞는 Dataset(SO101EEDataset 또는 SO101BCDataset)을 만들어 반환."""
    if detect_dataset_kind(repo_id, root) == "ee":
        from ai_layer.data.so101_ee_dataset import SO101EEDataset

        return SO101EEDataset(repo_id, root=root, **kwargs)
    from ai_layer.data.so101_bc_dataset import SO101BCDataset

    return SO101BCDataset(repo_id, root=root, **kwargs)
