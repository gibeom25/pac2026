#!/usr/bin/env python
"""A4(297x210mm) 용접선 텍스처 여러 종 + 종류별 가우시안 노이즈 변형을 미리 생성.

레퍼런스 6종(형태)에 더해, 각 형태의 핵심 파라미터(직선 각도, 곡선 진폭/주기, 코너 각도,
분기 각도, 점선 간격 등)에 가우시안 노이즈를 줘서 형태당 여러 변형(variant)을 미리 구워둔다
(런타임 랜덤화 아님 — MuJoCo 텍스처를 실행 중에 바꾸려면 mjr_uploadTexture로 GPU 재업로드까지
해야 해서 매번 model reload가 필요해지고, 뷰어를 켠 채로는 매끄럽지 않다. 대신 오프라인에서
넉넉히 미리 만들어두고 record_mujoco.py --variant로 고르는 게 훨씬 간단하고 뷰어도 안 끊김).

용접선 형태 다양화 목적 (seam_cv가 실제로 다루는 케이스와 매칭):
  straight   - 직선 (각도 노이즈)
  curve      - 완만한 곡선 (진폭/주기 노이즈)
  sharp_curve- 급격한 곡선 (진폭/주기 노이즈, 곡률 큼)
  corner     - 코너(꺾임, L자) (꺾이는 위치/각도 노이즈)
  branch     - 분기점(Y자) (분기 위치/각도 노이즈) - seam_cv 분기점 순서 처리 로직 검증용
  dashed     - 점선(끊긴 선) (간격 노이즈, 밑에 깔리는 곡선도 노이즈) - seam_cv KD-tree 갭브리징 검증용

variant 0은 항상 기존 레퍼런스(노이즈 없음, 하위호환용 a4_weld_seam_<name>.png와 동일 파라미터).
variant 1..N-1은 고정 시드로 매번 같은 노이즈를 재현 가능하게 생성.

실행: python assets/so101/textures/gen_seam_textures.py [--variants N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PX_PER_MM = 4
W, H = 297 * PX_PER_MM, 210 * PX_PER_MM  # 1188 x 840
BG_RGB = (245, 244, 238)
LINE_RGB = (35, 30, 28)
MARGIN_MM = 20
OUT_DIR = Path(__file__).resolve().parent
N_VARIANTS_DEFAULT = 5


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG_RGB)
    return img, ImageDraw.Draw(img)


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def _rand_width(rng: np.random.Generator | None, nominal: float = 2.5) -> float:
    """선 굵기(mm)도 형태마다 흔들리게 — 실제 seam/groove도 굵기가 일정하지 않으므로."""
    if rng is None:
        return nominal
    return _clip(rng.normal(nominal, nominal * 0.35), 1.2, nominal * 1.8)


def _draw_polyline(draw: ImageDraw.ImageDraw, pts_mm: list[tuple[float, float]], width_mm: float = 2.5,
                    dash_mm: tuple[float, float] | None = None) -> None:
    """pts_mm: (x_mm, y_mm) 리스트 (y=0이 위). dash_mm=(on, off)이면 점선."""
    pts_px = [(x * PX_PER_MM, y * PX_PER_MM) for x, y in pts_mm]
    width_px = max(1, int(width_mm * PX_PER_MM))
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


def gen_straight(rng: np.random.Generator | None = None) -> Image.Image:
    img, draw = _new_canvas()
    y0 = 105.0
    angle_deg = 0.0 if rng is None else _clip(rng.normal(0.0, 8.0), -20.0, 20.0)  # 수평 기준 기울기
    y_jitter = 0.0 if rng is None else rng.normal(0.0, 10.0)
    x0, x1 = MARGIN_MM, 297 - MARGIN_MM
    half_run = (x1 - x0) / 2
    dy = half_run * np.tan(np.deg2rad(angle_deg))
    y_c = y0 + y_jitter
    pts = [(x0, y_c - dy), (x1, y_c + dy)]
    _draw_polyline(draw, pts, width_mm=_rand_width(rng, 2.5))
    return img


def gen_curve(rng: np.random.Generator | None = None) -> Image.Image:
    img, draw = _new_canvas()
    x0, x1 = MARGIN_MM, 297 - MARGIN_MM
    xs = np.linspace(x0, x1, 400)
    amp = 15.0 if rng is None else _clip(rng.normal(15.0, 4.0), 6.0, 35.0)
    period = 180.0 if rng is None else _clip(rng.normal(180.0, 30.0), 100.0, 260.0)
    phase = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    ys = 105 + amp * np.sin(2 * np.pi * xs / period + phase)
    _draw_polyline(draw, list(zip(xs.tolist(), ys.tolist())), width_mm=_rand_width(rng, 2.5))
    return img


def gen_sharp_curve(rng: np.random.Generator | None = None) -> Image.Image:
    img, draw = _new_canvas()
    x0, x1 = MARGIN_MM, 297 - MARGIN_MM
    xs = np.linspace(x0, x1, 500)
    amp = 32.0 if rng is None else _clip(rng.normal(32.0, 6.0), 20.0, 45.0)
    period = 80.0 if rng is None else _clip(rng.normal(80.0, 15.0), 50.0, 120.0)
    phase = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    ys = 105 + amp * np.sin(2 * np.pi * xs / period + phase)
    _draw_polyline(draw, list(zip(xs.tolist(), ys.tolist())), width_mm=_rand_width(rng, 2.5))
    return img


def gen_corner(rng: np.random.Generator | None = None) -> Image.Image:
    img, draw = _new_canvas()
    bend_x = 297 / 2 if rng is None else _clip(rng.normal(297 / 2, 20.0), 90.0, 210.0)
    y_start = 60.0 if rng is None else _clip(rng.normal(60.0, 15.0), 30.0, 90.0)
    y_end = 160.0 if rng is None else _clip(rng.normal(160.0, 15.0), 120.0, 180.0)
    pts = [(MARGIN_MM, y_start), (bend_x, y_start), (297 - MARGIN_MM, y_end)]
    _draw_polyline(draw, pts, width_mm=_rand_width(rng, 2.5))
    return img


def gen_branch(rng: np.random.Generator | None = None) -> Image.Image:
    img, draw = _new_canvas()
    main_width = _rand_width(rng, 2.5)
    main_pts = [(MARGIN_MM, 105.0), (297 - MARGIN_MM, 105.0)]
    _draw_polyline(draw, main_pts, width_mm=main_width)
    branch_x = 297 / 2 if rng is None else _clip(rng.normal(297 / 2, 25.0), 80.0, 220.0)
    branch_len = 60.0 if rng is None else _clip(rng.normal(60.0, 12.0), 35.0, 85.0)
    branch_angle_deg = 45.0 if rng is None else _clip(rng.normal(45.0, 12.0), 20.0, 70.0)
    branch_start = (branch_x, 105.0)
    branch_end = (
        branch_x + branch_len * np.cos(np.deg2rad(branch_angle_deg)),
        105.0 - branch_len * np.sin(np.deg2rad(branch_angle_deg)),
    )
    _draw_polyline(draw, [branch_start, branch_end], width_mm=_rand_width(rng, 2.2))
    return img


def gen_dashed(rng: np.random.Generator | None = None) -> Image.Image:
    img, draw = _new_canvas()
    x0, x1 = MARGIN_MM, 297 - MARGIN_MM
    xs = np.linspace(x0, x1, 400)
    amp = 15.0 if rng is None else _clip(rng.normal(15.0, 4.0), 6.0, 30.0)
    period = 180.0 if rng is None else _clip(rng.normal(180.0, 30.0), 100.0, 260.0)
    phase = 0.0 if rng is None else rng.uniform(0, 2 * np.pi)
    ys = 105 + amp * np.sin(2 * np.pi * xs / period + phase)
    on_mm = 14.0 if rng is None else _clip(rng.normal(14.0, 4.0), 6.0, 25.0)
    off_mm = 8.0 if rng is None else _clip(rng.normal(8.0, 3.0), 3.0, 16.0)
    _draw_polyline(draw, list(zip(xs.tolist(), ys.tolist())), width_mm=_rand_width(rng, 2.5), dash_mm=(on_mm, off_mm))
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
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=int, default=N_VARIANTS_DEFAULT, help="형태당 생성할 변형 개수 (0번은 항상 노이즈 없는 기준)")
    args = p.parse_args()

    for name, fn in VARIANTS.items():
        # variant 0: 기존 레퍼런스와 동일 (하위호환, a4_weld_seam_<name>.png)
        img0 = fn(rng=None)
        out0 = OUT_DIR / f"a4_weld_seam_{name}.png"
        img0.save(out0)
        print(f"saved {out0} {img0.size}")

        for i in range(1, args.variants):
            rng = np.random.default_rng(seed=hash((name, i)) & 0xFFFFFFFF)
            img = fn(rng=rng)
            out = OUT_DIR / f"a4_weld_seam_{name}_{i}.png"
            img.save(out)
            print(f"saved {out} {img.size}")


if __name__ == "__main__":
    main()
