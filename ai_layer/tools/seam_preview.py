"""사진 파일로 선 인식(seam_cv)을 미리 맞춘다. 로봇 없이 휴대폰 사진으로 가능.

입력 사진마다 인식 결과를 그린 파일(<이름>_seam.png)을 만들고 특징 5개를 출력한다.
  초록: 인식된 선(순서대로), 빨강 ◯: 시작점, 파랑 ◯: lookahead 목표점, 회색: 이진화 마스크(반투명)

실행: PYTHONPATH=. python ai_layer/tools/seam_preview.py photo1.jpg photo2.jpg --out preview/ \
        [--width 320 --height 240] [--no-invert] [--block 35] [--c 7] [--min-area 30] [--max-gap 25]
옵션 뜻:
  --no-invert   배경보다 밝은 선(흰 분필/밝은 실리콘)이면 켠다. 기본은 어두운 선(펜).
  --block, --c  적응 이진화 창 크기/오프셋. 조명 얼룩이 심하면 block 을 키운다.
  --min-area    이보다 작은 얼룩은 버린다.
  --max-gap     끊긴 선을 이어붙일 최대 픽셀 거리.
맞는 값을 찾으면 SeamCVConfig 기본값(perception/seam_cv.py)에 반영한다. 학습·추론 모두 같은 값을 쓴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ai_layer.perception.seam_cv import SeamCVConfig, SeamGrooveDetector
from ai_layer.perception.seam_features import seam_features_from_rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="seam_preview")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--no-invert", action="store_true")
    ap.add_argument("--block", type=int, default=35)
    ap.add_argument("--c", type=int, default=7)
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--max-gap", type=float, default=25.0)
    args = ap.parse_args()

    cfg = SeamCVConfig(adaptive_block_size=args.block, adaptive_c=args.c, invert=not args.no_invert,
                       min_component_area=args.min_area, max_gap_px=args.max_gap)
    det = SeamGrooveDetector(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"읽기 실패: {path}")
            continue
        bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)  # 카메라 해상도로
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        points = det.detect(rgb)
        feats = seam_features_from_rgb(det, rgb)

        vis = bgr.copy()
        mask = det._binarize(rgb)
        vis[mask > 0] = (0.5 * vis[mask > 0] + 0.5 * np.array([180, 180, 180])).astype(np.uint8)
        if points:
            pts = np.array([(p.pixel[1], p.pixel[0]) for p in points], dtype=np.int32)
            cv2.polylines(vis, [pts], isClosed=False, color=(0, 200, 0), thickness=1)
            cv2.circle(vis, tuple(pts[0]), 5, (0, 0, 255), 2)
            tgt = det.lookahead_target(points, 0, 10)
            cv2.circle(vis, (tgt.pixel[1], tgt.pixel[0]), 5, (255, 0, 0), 2)
        dst = out / f"{Path(path).stem}_seam.png"
        cv2.imwrite(str(dst), vis)
        print(f"{Path(path).name}: 점 {len(points)}개, 특징 {np.round(feats, 4).tolist()} -> {dst}")


if __name__ == "__main__":
    main()
