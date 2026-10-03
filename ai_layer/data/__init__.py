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

# tools/record_mujoco.py의 DATASETS_DIR과 반드시 같은 값이어야 한다 — 거기서 저장한 데이터셋을
# --root 없이 여기서도 찾아야 함(2026-10-03: --root를 생략했더니 여기는 여전히 옛 HF 캐시
# 기본값을 보고 있어서 로컬 데이터를 못 찾고 HF Hub에 같은 이름으로 물어보려다 401이 난 버그).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"


def resolve_dataset_root(repo_id: str, root: str | Path | None) -> Path:
    """--root를 생략했을 때의 기본 위치. record_mujoco.py가 저장하는 곳과 반드시 일치해야 한다."""
    if root is not None:
        return Path(root)
    return DATASETS_DIR / repo_id


def detect_dataset_kind(repo_id: str, root: str | Path | None) -> str:
    """meta/info.json의 robot_type으로 "ee" / "joint"를 구분해 반환."""
    info_path = resolve_dataset_root(repo_id, root) / "meta" / "info.json"
    robot_type = json.loads(info_path.read_text()).get("robot_type", "") if info_path.exists() else ""
    return "ee" if "ee_mujoco" in str(robot_type) else "joint"


def load_bc_dataset(repo_id: str, root: str | Path | None = None, **kwargs):
    """robot_type에 맞는 Dataset(SO101EEDataset 또는 SO101BCDataset)을 만들어 반환.

    root는 여기서 resolve_dataset_root()로 구체 경로로 바꿔서 넘긴다 — None을 그대로 넘기면
    LeRobotDataset이 자기 기본값(HF 캐시)을 쓰게 되어 위 detect_dataset_kind()가 본 경로와
    어긋난다.
    """
    resolved_root = resolve_dataset_root(repo_id, root)
    if detect_dataset_kind(repo_id, resolved_root) == "ee":
        from ai_layer.data.so101_ee_dataset import SO101EEDataset

        return SO101EEDataset(repo_id, root=resolved_root, **kwargs)
    from ai_layer.data.so101_bc_dataset import SO101BCDataset

    return SO101BCDataset(repo_id, root=resolved_root, **kwargs)
