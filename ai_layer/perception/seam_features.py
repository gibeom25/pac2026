"""이미지 -> seam 특징 (5,) 한 곳에서 정의. 학습(so101_bc_dataset)과 추론(control_bridge.ai_node)이 같이 쓴다.

[lookahead 상대 row/h, 상대 col/w, 현재점 곡률, 굵기/max(h,w), 점 개수/(h·w)]
depth 없는 2D 전용 모드 (docs 3.1 CV 모듈 참고).
"""

from __future__ import annotations

import numpy as np

from ai_layer.perception.seam_cv import SeamGrooveDetector

SEAM_FEATURE_DIM = 5


def seam_features_from_rgb(detector: SeamGrooveDetector, img_rgb_uint8: np.ndarray, lookahead: int = 10) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (5,) float32."""
    h, w = img_rgb_uint8.shape[:2]
    points = detector.detect(img_rgb_uint8)
    if not points:
        return np.zeros(SEAM_FEATURE_DIM, dtype=np.float32)
    target = detector.lookahead_target(points, current_idx=0, lookahead=lookahead)
    cur = points[0]
    dr = (target.pixel[0] - cur.pixel[0]) / h
    dc = (target.pixel[1] - cur.pixel[1]) / w
    return np.array(
        [dr, dc, cur.curvature, cur.thickness_px / max(h, w), len(points) / (h * w)],
        dtype=np.float32,
    )


def seam_features_from_chw(detector: SeamGrooveDetector, image_chw_float01) -> np.ndarray:
    """(3, H, W) float[0,1] 텐서/배열 -> (5,) float32. LeRobotDataset 이미지 형식용."""
    arr = np.asarray(image_chw_float01.permute(1, 2, 0).numpy() if hasattr(image_chw_float01, "permute") else np.transpose(image_chw_float01, (1, 2, 0)))
    img = (arr * 255).astype(np.uint8)
    return seam_features_from_rgb(detector, img)
