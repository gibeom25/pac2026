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


def require_local_dataset(repo_id: str, root: str | Path | None) -> Path:
    """root를 resolve하고 meta/info.json이 실제로 있는지 확인, 없으면 바로 명확한 에러.

    확인 없이 바로 LeRobotDataset(...)을 만들면, 로컬에 아무것도 없을 때 lerobot이 "그럼 HF Hub
    repo겠지"라고 넘겨짚고 거기 물어보다가 401/RepositoryNotFoundError로 죽는다(2026-10-03,
    실제로 겪음 — record_mujoco.py가 에피소드를 하나도 못 저장하고 끝난 빈 디렉토리에 대고
    check_dataset.py를 돌렸을 때). 그 네트워크 에러 대신 "여기 데이터가 없다"는 걸 바로 알려준다.
    """
    resolved = resolve_dataset_root(repo_id, root)
    info_path = resolved / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} 가 없다 — {resolved} 에 녹화된 데이터셋이 없다. "
            f"record_mujoco.py가 적어도 한 에피소드를 저장하고 끝까지 돌았는지(마지막에 "
            f"'[record] 데이터셋: ...' 줄이 찍혔는지) 확인할 것. --root를 실수로 다른 경로로 "
            f"줬을 수도 있다."
        )
    return resolved


def detect_dataset_kind(repo_id: str, root: str | Path | None) -> str:
    """meta/info.json의 robot_type으로 "ee" / "joint"를 구분해 반환."""
    info_path = resolve_dataset_root(repo_id, root) / "meta" / "info.json"
    robot_type = json.loads(info_path.read_text()).get("robot_type", "") if info_path.exists() else ""
    return "ee" if "ee_mujoco" in str(robot_type) else "joint"


def load_bc_dataset(repo_id: str, root: str | Path | None = None, **kwargs):
    """robot_type에 맞는 Dataset(SO101EEDataset 또는 SO101BCDataset)을 만들어 반환.

    root는 여기서 require_local_dataset()으로 구체 경로로 바꿔서 넘긴다 — None을 그대로 넘기면
    LeRobotDataset이 자기 기본값(HF 캐시)을 쓰게 되어 위 detect_dataset_kind()가 본 경로와
    어긋나고, 로컬에 데이터가 없을 때도 바로 명확한 에러를 낸다(require_local_dataset 참고).
    """
    resolved_root = require_local_dataset(repo_id, root)
    if detect_dataset_kind(repo_id, resolved_root) == "ee":
        from ai_layer.data.so101_ee_dataset import SO101EEDataset

        return SO101EEDataset(repo_id, root=resolved_root, **kwargs)
    from ai_layer.data.so101_bc_dataset import SO101BCDataset

    return SO101BCDataset(repo_id, root=resolved_root, **kwargs)
