"""Seam/Groove 고전 CV 인식 모듈.

설계 문서: docs/AI_추론계층_프레임워크.md 3.1절.
BC/RL과 분리된 경량 고전 CV로 용접선(펜/프린트 선)을 검출한다.
BC/RL은 이 모듈이 뽑아낸 "정제된 경로"만 보고 "어떻게 따라갈지(속도/부드러움)"만
학습하면 되도록, 인식 책임을 여기서 전담한다.

파이프라인 (docs 3.1):
  1. 색상/대비 threshold + 형태학적 노이즈 제거
  2. Skeletonization (1px 중심선)
  3. 순서 있는 polyline으로 정렬
  4. 끊김(gap) 보강
  5. Depth로 3D 좌표 변환 (RGBD가 있을 때)
  6. 로컬 곡률/선 굵기 특징 추출
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


@dataclass
class SeamPoint:
    """경로 위 한 점의 특징. world/EEF 좌표 변환은 depth가 주어졌을 때만 채워진다."""

    pixel: tuple[int, int]
    xyz: np.ndarray | None  # depth 없으면 None (2D 전용 모드)
    curvature: float
    thickness_px: float


@dataclass
class SeamCVConfig:
    # 이진화: 어두운 선(펜) 기준 기본값. 밝은/컬러 선이면 사용처에서 오버라이드.
    adaptive_block_size: int = 35
    adaptive_c: int = 7
    invert: bool = True  # 배경보다 어두운 선을 전경으로
    morph_kernel: int = 3
    min_component_area: int = 30
    max_gap_px: float = 25.0  # 스켈레톤 끝점 간 이 거리 이내면 연결
    curvature_window: int = 5  # 곡률 추정에 쓸 좌우 이웃 점 개수


class SeamGrooveDetector:
    """RGB(+Depth) 프레임에서 용접선 polyline과 로컬 특징을 추출."""

    def __init__(self, config: SeamCVConfig | None = None):
        self.cfg = config or SeamCVConfig()

    # ---- 1. 이진화 ----
    def _binarize(self, rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        block = self.cfg.adaptive_block_size | 1  # 홀수 강제
        thresh_type = cv2.THRESH_BINARY_INV if self.cfg.invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block, self.cfg.adaptive_c
        )
        k = self.cfg.morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 작은 노이즈 컴포넌트 제거
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= self.cfg.min_component_area:
                clean[labels == i] = 255
        return clean

    # ---- 2. Skeletonize ----
    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        return skeletonize(binary > 0)

    # ---- 3. 순서 있는 polyline 정렬 ----
    @staticmethod
    def _find_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
        """8-이웃 개수가 1인 스켈레톤 픽셀 = 끝점. 컨볼루션으로 한 번에 센다 (파이썬 픽셀 루프 없음)."""
        sk = (skeleton > 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.float32)
        kernel[1, 1] = 0.0
        neighbor_count = cv2.filter2D(sk.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        ys, xs = np.nonzero((sk > 0) & (np.rint(neighbor_count).astype(int) == 1))
        return list(zip(ys.tolist(), xs.tolist()))

    def _order_polyline(self, skeleton: np.ndarray) -> np.ndarray:
        """스켈레톤 픽셀을 끝점에서 시작해 인접 픽셀을 따라가며 순서를 매긴다.

        - 분기점(이웃 2개 이상)에서는 직전 진행 방향과 가장 비슷한 이웃을 고른다
          (예전: 임의의 첫 이웃 → Y자/십자에서 순서가 뒤엉김).
        - 끊긴 구간은 KD-tree로 가장 가까운 남은 점을 찾는다 (예전: 남은 점 전부와 거리 계산, O(N²)).
        """
        ys, xs = np.nonzero(skeleton)
        if len(ys) == 0:
            return np.zeros((0, 2), dtype=int)
        pts = np.stack([ys, xs], axis=1)
        remaining = set(map(tuple, pts.tolist()))
        tree = cKDTree(pts)

        endpoints = self._find_endpoints(skeleton)
        start = endpoints[0] if endpoints else next(iter(remaining))

        ordered = [start]
        remaining.discard(start)
        current = start
        prev_dir = np.zeros(2)
        while remaining:
            y, x = current
            neighbors = [
                (y + dy, x + dx)
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
                if (dy != 0 or dx != 0) and (y + dy, x + dx) in remaining
            ]
            if not neighbors:
                # 끊긴 구간: max_gap_px 이내의 가장 가까운 남은 점으로 점프 (KD-tree, 가까운 후보부터 확인)
                k = min(len(pts), 16)
                dists, idxs = tree.query([y, x], k=k)
                dists = np.atleast_1d(dists)
                idxs = np.atleast_1d(idxs)
                nearest = None
                for d, i in zip(dists, idxs):
                    cand = (int(pts[i, 0]), int(pts[i, 1]))
                    if cand in remaining:
                        nearest = cand if d <= self.cfg.max_gap_px else None
                        break
                if nearest is None:
                    # k개 안에 남은 점이 없으면 전체에서 찾는다 (드묾)
                    rem = np.array(list(remaining))
                    d_all = np.hypot(rem[:, 0] - y, rem[:, 1] - x)
                    j = int(d_all.argmin())
                    if d_all[j] > self.cfg.max_gap_px:
                        break  # 진짜 끊김 — 여기서 polyline 종료
                    nearest = (int(rem[j, 0]), int(rem[j, 1]))
                nxt = nearest
            elif len(neighbors) == 1 or not prev_dir.any():
                nxt = neighbors[0]
            else:
                # 분기점: 진행 방향(prev_dir)과 코사인 유사도가 가장 큰 이웃
                cand = np.array(neighbors, dtype=float) - np.array(current, dtype=float)
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                nxt = neighbors[int(np.argmax(cand @ prev_dir))]
            prev_dir = np.array(nxt, dtype=float) - np.array(current, dtype=float)
            prev_dir /= max(np.linalg.norm(prev_dir), 1e-6)
            ordered.append(nxt)
            remaining.discard(nxt)
            current = nxt

        return np.array(ordered)  # (N, 2) in (row, col)

    # ---- 6. 로컬 곡률 / 굵기 ----
    def _local_curvature(self, polyline_rc: np.ndarray) -> np.ndarray:
        w = self.cfg.curvature_window
        n = len(polyline_rc)
        curv = np.zeros(n)
        for i in range(n):
            a = polyline_rc[max(0, i - w)]
            b = polyline_rc[i]
            c = polyline_rc[min(n - 1, i + w)]
            v1 = b - a
            v2 = c - b
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            curv[i] = np.arccos(cos_angle) / max(n1, 1e-6)  # 각도 변화율 근사
        return curv

    def _local_thickness(self, binary: np.ndarray, polyline_rc: np.ndarray) -> np.ndarray:
        dist = cv2.distanceTransform((binary > 0).astype(np.uint8), cv2.DIST_L2, 5)
        thickness = np.array([2.0 * dist[r, c] for r, c in polyline_rc])
        return thickness

    # ---- 5. depth -> 3D ----
    @staticmethod
    def _pixels_to_xyz(
        polyline_rc: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray
    ) -> np.ndarray:
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        xyz = np.zeros((len(polyline_rc), 3))
        for i, (r, c) in enumerate(polyline_rc):
            z = float(depth[r, c])
            xyz[i] = [(c - cx) * z / fx, (r - cy) * z / fy, z]
        return xyz

    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
    ) -> list[SeamPoint]:
        """RGB(+Depth) 한 프레임에서 순서 있는 SeamPoint 리스트를 반환.

        depth/intrinsics가 없으면 xyz=None (픽셀 좌표 + 로컬 특징만 반환, 2D 전용 모드).
        """
        binary = self._binarize(rgb)
        skeleton = self._skeletonize(binary)
        polyline_rc = self._order_polyline(skeleton)
        if len(polyline_rc) == 0:
            return []

        curvature = self._local_curvature(polyline_rc)
        thickness = self._local_thickness(binary, polyline_rc)

        xyz_all = None
        if depth is not None and intrinsics is not None:
            xyz_all = self._pixels_to_xyz(polyline_rc, depth, intrinsics)

        points = []
        for i, (r, c) in enumerate(polyline_rc):
            points.append(
                SeamPoint(
                    pixel=(int(r), int(c)),
                    xyz=xyz_all[i] if xyz_all is not None else None,
                    curvature=float(curvature[i]),
                    thickness_px=float(thickness[i]),
                )
            )
        return points

    def lookahead_target(
        self, points: list[SeamPoint], current_idx: int, lookahead: int = 10
    ) -> SeamPoint | None:
        """현재 진행 인덱스 기준 lookahead만큼 앞선 목표점 반환 (제어 목표 계산용)."""
        if not points:
            return None
        idx = min(current_idx + lookahead, len(points) - 1)
        return points[idx]
