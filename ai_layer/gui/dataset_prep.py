"""record_mujoco.py `build_dataset()`과 같은 로직이지만, 기존 데이터셋 발견 시 터미널
input()이 아니라 QMessageBox로 덮어쓰기/이어쓰기/취소를 묻는다 — GUI 프로세스엔 터미널
입력을 기대할 수 없으므로(사용자가 터미널을 안 보고 있을 수 있음) 따로 둔다. record_mujoco.py
의 `_dataset_root`/`_existing_episode_count`를 그대로 재사용하고, 그 뒤 로직만 Qt 다이얼로그로
바꿨다 — 두 함수는 안 건드렸으니 CLI 쪽 동작엔 영향 없다."""

from __future__ import annotations

import argparse
import shutil

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from PyQt6.QtWidgets import QMessageBox, QWidget

from ai_layer.tools.record_mujoco import (
    ACTION_KEYS,
    CAMERA_HW,
    STATE_KEYS,
    _dataset_root,
    _existing_episode_count,
)


class DatasetCancelled(Exception):
    """사용자가 덮어쓰기 확인 다이얼로그에서 취소를 눌렀을 때."""


def build_dataset_gui(
    args: argparse.Namespace, parent: QWidget | None = None, real: bool = False, with_overview: bool = False
) -> LeRobotDataset:
    """real=True면 robot_type을 실로봇으로 표시하고, with_overview=True면 고정(overview) 카메라도
    observation.images.overview로 저장한다(2026-10-08, configs/so101_act_bc.OVERVIEW_IMAGE_KEY 참고)."""
    hw_obs = {name: float for name in STATE_KEYS}
    hw_obs["wrist"] = CAMERA_HW
    if with_overview:
        hw_obs["overview"] = CAMERA_HW
    hw_action = {name: float for name in ACTION_KEYS}
    obs_features = hw_to_dataset_features(hw_obs, "observation", use_video=False)
    action_features = hw_to_dataset_features(hw_action, "action", use_video=False)
    features = {**obs_features, **action_features}

    root = _dataset_root(args)
    if root.exists():
        n = _existing_episode_count(root)
        if n == 0:
            shutil.rmtree(root)
        else:
            box = QMessageBox(parent)
            box.setWindowTitle("기존 데이터셋 발견")
            box.setText(f"{root}\n에피소드 {n}개가 이미 있습니다. 어떻게 할까요?")
            overwrite_btn = box.addButton("덮어쓰기", QMessageBox.ButtonRole.DestructiveRole)
            resume_btn = box.addButton("이어서 기록", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("취소", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is overwrite_btn:
                shutil.rmtree(root)
            elif clicked is resume_btn:
                return LeRobotDataset(repo_id=args.repo_id, root=root)
            else:
                raise DatasetCancelled()

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        # "_ee_"가 들어가야 data.detect_dataset_kind가 EE-native 포맷으로 인식한다
        robot_type="so101_ee_real_joystick" if real else "so101_ee_mujoco_joystick",
        use_videos=False,
    )
