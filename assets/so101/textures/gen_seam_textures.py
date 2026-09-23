#!/usr/bin/env python
"""A4(297x210mm) 용접선 텍스처 여러 종 생성 (record_mujoco.py의 scene_a4_<name>.xml용).

용접선 형태 다양화 목적 (seam_cv가 실제로 다루는 케이스와 매칭):
  straight   - 직선
  curve      - 완만한 곡선 (기존 a4_weld_seam.png와 동일 파라미터)
  sharp_curve- 급격한 곡선 (곡률 큼)
  corner     - 코너(꺾임, L자)
  branch     - 분기점(Y자) - seam_cv의 분기점 순서 처리 로직 검증용
  dashed     - 점선(끊긴 선) - seam_cv의 갭브리징(cKDTree) 로직 검증용

실행: python assets/so101/textures/gen_seam_textures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PX_PER_MM = 4
W, H = 297 * PX_PER_MM, 210 * PX_PER_MM  # 1188 x 840
BG_RGB = (245, 244, 238)
LINE_RGB = (35, 30, 28)
MARGIN_MM = 20
OUT_DIR = Path(__file__).resolve().parent


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG_RGB)
    return img, ImageDraw.Draw(img)


def _draw_polyline(draw: ImageDraw.ImageDraw, pts_mm: list[tuple[float, float]], width_mm: float = 2.5,
                    dash_mm: tuple[float, float] | None = None) -> None:
    """pts_mm: (x_mm, y_mm) 리스트 (y=0이 위). dash_mm=(on, off)이면 점선."""
    pts_px = [(x * PX_PER_MM, y * PX_PER_MM) for x, y in pts_mm]
    width_px = int(width_mm * PX_PER_MM)
    r = width_px // 2

    def cap(p):
        draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=LINE_RGB)

    if dash_mm is None:
        draw.line(pts_px, fill=LINE_RGB, width=width_px, joint="curve")
        cap(pts_px[0])
        cap(pts_px[-1])
        return

    # 점선: 누적 길이 기준으로 on/off 구간을 나눠 각각 별도 선분으로 그린다.
    on_mm, off_mm = dash_mm
    seg = []
    acc = 0.0
    on = True
    seg.append(pts_px[0])
    for i in range(1, len(pts_px)):
        p0_mm, p1_mm = np.array(pts_mm[i - 1]), np.array(pts_mm[i])
        seg_len = float(np.linalg.norm(p1_mm - p0_mm))
        remaining = seg_len
        cur_mm = p0_mm
        while remaining > 0:
            budget = (on_mm if on else off_mm) - acc
            step = min(budget, remaining)
            nxt_mm = cur_mm + (p1_mm - p0_mm) * (step / max(seg_len, 1e-9))
            if on:
                seg.append((nxt_mm[0] * PX_PER_MM, nxt_mm[1] * PX_PER_MM))
            cur_mm = nxt_mm
            acc += step
            remaining -= step
            if acc >= (on_mm if on else off_mm) - 1e-6:
                if on and len(seg) > 1:
                    draw.line(seg, fill=LINE_RGB, width=width_px, joint="curve")
                    cap(seg[0])
                    cap(seg[-1])
                on = not on
                acc = 0.0
                seg = [(cur_mm[0] * PX_PER_MM, cur_mm[1] * PX_PER_MM)]
    if on and len(seg) > 1:
        draw.line(seg, fill=LINE_RGB, width=width_px, joint="curve")
        cap(seg[0])
        cap(seg[-1])


def gen_straight() -> Image.Image:
    img, draw = _new_canvas()
    y = 105.0
    pts = [(MARGIN_MM, y), (297 - MARGIN_MM, y)]
    _draw_polyline(draw, pts, width_mm=2.5)
    return img


def gen_curve() -> Image.Image:
    img, draw = _new_canvas()
    x0, x1 = MARGIN_MM, 297 - MARGIN_MM
    xs = np.linspace(x0, x1, 400)
    amp, period = 15, 180
    ys = 105 + amp * np.sin(2 * np.pi * xs / period)
    _draw_polyline(draw, list(zip(xs.tolist(), ys.tolist())), width_mm=2.5)
    return img


def gen_sharp_curve() -> Image.Image:
    img, draw = _new_canvas()
    x0, x1 = MARGIN_MM, 297 - MARGIN_MM
    xs = np.linspace(x0, x1, 500)
    amp, period = 32, 80
    ys = 105 + amp * np.sin(2 * np.pi * xs / period)
    _draw_polyline(draw, list(zip(xs.tolist(), ys.tolist())), width_mm=2.5)
    return img


def gen_corner() -> Image.Image:
    img, draw = _new_canvas()
    pts = [(MARGIN_MM, 60.0), (297 / 2, 60.0), (297 - MARGIN_MM, 160.0)]
    _draw_polyline(draw, pts, width_mm=2.5)
    return img


def gen_branch() -> Image.Image:
    img, draw = _new_canvas()
    # 메인 선
    main_pts = [(MARGIN_MM, 105.0), (297 - MARGIN_MM, 105.0)]
    _draw_polyline(draw, main_pts, width_mm=2.5)
    # 중간 지점에서 갈라지는 가지
    branch_start = (297 / 2, 105.0)
    branch_end = (297 / 2 + 60, 105.0 - 55)
    _draw_polyline(draw, [branch_start, branch_end], width_mm=2.2)
    return img


def gen_dashed() -> Image.Image:
    img, draw = _new_canvas()
    x0, x1 = MARGIN_MM, 297 - MARGIN_MM
    xs = np.linspace(x0, x1, 400)
    amp, period = 15, 180
    ys = 105 + amp * np.sin(2 * np.pi * xs / period)
    _draw_polyline(draw, list(zip(xs.tolist(), ys.tolist())), width_mm=2.5, dash_mm=(14.0, 8.0))
    return img


VARIANTS = {
    "straight": gen_straight,
    "curve": gen_curve,
    "sharp_curve": gen_sharp_curve,
    "corner": gen_corner,
    "branch": gen_branch,
    "dashed": gen_dashed,
}


def main() -> None:
    for name, fn in VARIANTS.items():
        img = fn()
        out = OUT_DIR / f"a4_weld_seam_{name}.png"
        img.save(out)
        print(f"saved {out} {img.size}")


if __name__ == "__main__":
    main()
