"""gen_seam_textures.py가 구워둔 ground-truth 경로(json)를 월드 좌표 폴리라인으로 변환.

RL(so101_seam_env.py)의 reward(track_reward/coverage_reward)는 "진짜" 3D 용접선 경로가
필요하다. CV로 역추출(이미지 -> skeleton -> 픽셀 -> 3D 역투영)하면 카메라/텍스처 UV 변환
오차가 누적되므로, 텍스처를 그릴 때 실제로 쓴 좌표(assets/so101/textures/*.json,
gen_seam_textures.py가 함께 저장)를 그대로 가져와 world로 변환하는 쪽이 훨씬 정확하다.

mm -> world 변환 공식 (2026-10-03, MuJoCo 렌더로 직접 실측 보정):
  scene_a4*.xml의 a4_paper geom은 pos=(0.25,0,0.001) size=(0.1485,0.105,0.001), 회전 없음
  (로컬 xy = world xy). mocap_target을 world y=+0.08로 옮기고 top-down 렌더해서 화면 이동
  방향을 직접 확인한 결과:
    - x_mm(텍스처 왼쪽=0) 증가 -> world_x 증가 (텍스처 왼쪽 끝 = world_x 0.25-0.1485)
    - y_mm(텍스처 위=0) 증가 -> world_y *감소* (텍스처 위쪽 끝 = world_y 0.25+0.105... 가 아니라
      world_y = +0.105, 아래쪽 끝이 -0.105)
  즉:
    world_x = 0.25 - 0.1485 + x_mm/1000
    world_y = 0.0  + 0.105  - y_mm/1000
  (단위/부호를 바꾸게 되면 이 보정 실험을 다시 해야 한다 — 추측으로 고치지 말 것.)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "so101"
TEXTURES_DIR = ASSETS_DIR / "textures"

SCENE_NAMES = ["curve", "straight", "sharp_curve", "corner", "branch", "dashed"]
N_VARIANTS = 5  # textures/gen_seam_textures.py --variants 기본값과 맞춤 (0=기준, 1~4=노이즈)

PAPER_CENTER_X = 0.25
PAPER_CENTER_Y = 0.0
PAPER_HALF_W = 0.1485  # scene_a4*.xml a4_paper geom size[0]과 일치해야 함
PAPER_HALF_H = 0.105  # scene_a4*.xml a4_paper geom size[1]과 일치해야 함

# 작업 중 도구 끝이 떠 있어야 할 목표 높이. 접촉(z≈0)은 실패 조건(record_mujoco.py와 동일 규약)이라
# 0보다 확실히 위에 둔다. coverage_reward의 coverage_radius(0.008m)가 이 높이에서의 자연스러운
# 수직 오차를 흡수할 수 있을 정도로 작게 잡음.
HOVER_Z = 0.01


def _json_path(scene: str, variant: int) -> Path:
    name = scene if variant == 0 else f"{scene}_{variant}"
    return TEXTURES_DIR / f"a4_weld_seam_{name}.json"


def _mm_to_world_xy(path_mm: np.ndarray) -> np.ndarray:
    x = PAPER_CENTER_X - PAPER_HALF_W + path_mm[:, 0] * 0.001
    y = PAPER_CENTER_Y + PAPER_HALF_H - path_mm[:, 1] * 0.001
    return np.stack([x, y], axis=1)


def _resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    """호 길이 기준 등간격 n점 재샘플 (reward.py/so101_seam_env.py가 기대하는 고정 길이)."""
    seg = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(xy[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(seg_len) - 1)
    local = (targets - cum[idx]) / np.maximum(seg_len[idx], 1e-9)
    return xy[idx] + local[:, None] * seg[idx]


def _mean_curvature(xy_world: np.ndarray) -> float:
    """재샘플된 월드 폴리라인의 평균 |각도변화/길이| (1/m) — 에피소드 단위 대표 곡률 스칼라.

    reward.track_speed()가 곡률 클수록 감속 목표를 주는 데 쓰는 것과 같은 스칼라 단순화
    (so101_seam_env.py의 이전 구현도 에피소드당 상수 하나였음 — 점별로 바꾸지 않음).
    """
    if len(xy_world) < 3:
        return 0.0
    v1 = xy_world[1:-1] - xy_world[:-2]
    v2 = xy_world[2:] - xy_world[1:-1]
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-6) & (n2 > 1e-6)
    if not valid.any():
        return 0.0
    cos = np.clip((v1[valid] * v2[valid]).sum(axis=1) / (n1[valid] * n2[valid]), -1.0, 1.0)
    ang = np.arccos(cos)
    seg_len = (n1[valid] + n2[valid]) / 2
    return float(np.mean(ang / np.maximum(seg_len, 1e-6)))


class SeamGroundTruth:
    """(scene, variant) -> (target_polyline (num_points,3) world, curvature, thickness_m). 캐시."""

    def __init__(self, num_points: int = 20, hover_z: float = HOVER_Z):
        self.num_points = num_points
        self.hover_z = hover_z
        self._cache: dict[tuple[str, int], tuple[np.ndarray, float, float]] = {}

    def load(self, scene: str, variant: int) -> tuple[np.ndarray, float, float]:
        key = (scene, variant)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        data = json.loads(_json_path(scene, variant).read_text())
        path_mm = np.asarray(data["path_mm"], dtype=float)
        width_mm = float(data["width_mm"])

        xy = _mm_to_world_xy(path_mm)
        xy_rs = _resample_polyline(xy, self.num_points)
        polyline = np.concatenate(
            [xy_rs, np.full((self.num_points, 1), self.hover_z)], axis=1
        ).astype(np.float32)
        curvature = _mean_curvature(xy_rs)
        thickness_m = width_mm * 0.001

        result = (polyline, curvature, thickness_m)
        self._cache[key] = result
        return result
