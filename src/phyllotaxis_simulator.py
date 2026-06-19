"""Phyllotaxis mp4 simulator: fixed turn angle + sqrt radius = sunflower packing.

Modes:
  grow     - golden-angle seeds appear one by one
  still    - full golden-angle beauty shot
  angles   - compare rational, near-golden, and golden turns
  spirals  - tint Fibonacci parastichy families on the golden head

Usage:
  python phyllotaxis_simulator.py --mode still --output out/phyllotaxis_still.mp4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

SS = 2
BG = (4, 6, 13)
BOX_BG = (9, 12, 24, 218)
SOFT_WHITE = (225, 233, 242, 255)
GOLDEN_ANGLE_DEG = 137.50776
DEFAULT_N = 1700
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, max(8, size))
    except OSError:
        return ImageFont.load_default()


def _writer(output: str, fps: int):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return aa + (bb - aa) * t[:, None]


def _gradient(t: np.ndarray) -> np.ndarray:
    stops = (
        (0.00, (255, 224, 116)),
        (0.34, (255, 154, 42)),
        (0.70, (70, 222, 255)),
        (1.00, (245, 82, 221)),
    )
    out = np.zeros((len(t), 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        lo, c0 = stops[i]
        hi, c1 = stops[i + 1]
        mask = (t >= lo) & (t <= hi)
        local = np.clip((t[mask] - lo) / max(1e-6, hi - lo), 0, 1)
        out[mask] = _lerp(c0, c1, local)
    return np.clip(out, 0, 255).astype(np.uint8)


def _compute_head(
    n_seeds: int,
    turn_angle_deg: float,
    box: tuple[float, float, float, float],
    margin: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(n_seeds, dtype=np.float64)
    theta = np.deg2rad(turn_angle_deg) * n
    raw_r = np.sqrt(n)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    max_r = min(x1 - x0, y1 - y0) * (0.5 - margin)
    c = max_r / max(1.0, math.sqrt(n_seeds - 1))
    r = c * raw_r
    xy = np.column_stack((cx + r * np.cos(theta), cy + r * np.sin(theta)))
    t = raw_r / max(1.0, raw_r.max())
    sizes = 2.0 + 3.8 * t
    return xy, t, sizes


def _new_rgba(w: int, h: int) -> Image.Image:
    return Image.new("RGBA", (w * SS, h * SS), (*BG, 255))


def _draw_label(
    img: Image.Image,
    xy: tuple[float, float],
    text: str,
    panel_width: float,
    scale: float = 0.050,
) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    x, y = int(xy[0] * SS), int(xy[1] * SS)
    font = _font(int(panel_width * scale * SS))
    pad = int(max(8, panel_width * 0.012) * SS)
    bbox = draw.textbbox((x, y), text, font=font)
    rect = (x - pad, y - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rounded_rectangle(rect, radius=int(5 * SS), fill=BOX_BG)
    draw.text((x, y), text, font=font, fill=SOFT_WHITE)


def _draw_head(
    w: int,
    h: int,
    xy: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
    reveal: int | None = None,
    label: str | None = None,
    label_xy: tuple[float, float] = (28, 24),
    label_width: float | None = None,
    fast: bool = False,
) -> Image.Image:
    # fast=True (grow アニメ用): 毎フレームの全画面 GaussianBlur / bloom を省いて高速化。
    # 重い bloom は静止 hero(still) でのみ。動く構築では mid+sharp で十分見栄えする。
    count = len(xy) if reveal is None else max(0, min(int(reveal), len(xy)))
    base = _new_rgba(w, h)
    glow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mid = Image.new("RGBA", base.size, (0, 0, 0, 0))
    sharp = Image.new("RGBA", base.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    md = ImageDraw.Draw(mid, "RGBA")
    sd = ImageDraw.Draw(sharp, "RGBA")

    for (x, y), col, radius in zip(xy[:count], colors[:count], sizes[:count]):
        sx, sy = x * SS, y * SS
        r = radius * SS
        c = tuple(int(v) for v in col)
        if not fast:
            gd.ellipse((sx - r * 2.8, sy - r * 2.8, sx + r * 2.8, sy + r * 2.8), fill=(*c, 42))
        md.ellipse((sx - r * 1.35, sy - r * 1.35, sx + r * 1.35, sy + r * 1.35), fill=(*c, 150 if fast else 115))
        sd.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(*c, 230))
        sd.ellipse((sx - r * 0.28, sy - r * 0.28, sx + r * 0.28, sy + r * 0.28), fill=(255, 244, 185, 205))

    if fast:
        composed = Image.alpha_composite(Image.alpha_composite(base, mid), sharp)
        if label:
            _draw_label(composed, label_xy, label, label_width or w)
        return composed

    glow = glow.filter(ImageFilter.GaussianBlur(max(2, int(3.0 * SS))))
    mid = mid.filter(ImageFilter.GaussianBlur(max(1, int(0.55 * SS))))
    composed = Image.alpha_composite(Image.alpha_composite(Image.alpha_composite(base, glow), mid), sharp)
    bloom = composed.filter(ImageFilter.GaussianBlur(max(2, int(2.4 * SS))))
    composed = ImageChops.add(composed, bloom, scale=2.35, offset=0)
    if label:
        _draw_label(composed, label_xy, label, label_width or w)
    return composed


def _finish(img: Image.Image, w: int, h: int) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), dtype=np.uint8)


def _panel_box(index: int, w: int, h: int, cols: int = 3) -> tuple[float, float, float, float]:
    gap = w * 0.018
    top = h * 0.08
    bottom = h * 0.05
    panel_w = (w - gap * (cols + 1)) / cols
    x0 = gap + index * (panel_w + gap)
    return (x0, top, x0 + panel_w, h - bottom)


def _render_grow(output: str, duration: float, fps: int, w: int, h: int) -> None:
    # grow: ACCURATE 機構 (F15-4). 新しい種は中心(成長点)で生まれ、古い種が外へ押し出される。
    # 種 k は時刻 tnow で age=tnow-k だけ歳をとり radius ∝ √age (面積が時間に比例して増える=最古が最外周)。
    # 連続する種は黄金角ぶん離れる。毎フレーム全種の位置を再計算(全て動く)が、N~1700 の
    # vectorized sqrt/cos/sin + fast 描画で軽量(per-frame の全画面 blur は無し)。
    steps = max(1, int(duration * fps))
    n = DEFAULT_N
    k = np.arange(n, dtype=np.float64)
    angle = np.deg2rad(GOLDEN_ANGLE_DEG) * k
    cosA, sinA = np.cos(angle), np.sin(angle)
    cx, cy = w / 2.0, h / 2.0
    max_r = min(w, h) * (0.5 - 0.08)
    c = max_r / math.sqrt(max(1.0, n))
    reveal_frames = max(1, int(steps * 0.82))
    with _writer(output, fps) as wr:
        for frame in range(steps):
            p = min(1.0, frame / max(1, reveal_frames - 1))
            tnow = (p ** 0.85) * n           # 時刻(=これまでに生まれた種数の連続版)
            born = int(min(n, math.floor(tnow) + 1))
            age = np.clip(tnow - k[:born], 0.0, None)
            r = c * np.sqrt(age)             # 古い種ほど外、最新は中心(r≈0)
            tnorm = np.clip(r / max_r, 0.0, 1.0)
            xy = np.column_stack((cx + r * cosA[:born], cy + r * sinA[:born]))
            colors = _gradient(tnorm)
            sizes = 2.0 + 3.8 * tnorm
            img = _draw_head(w, h, xy, colors, sizes, reveal=None,
                             label="new seeds appear at the center", fast=True)
            wr.append_data(_finish(img, w, h))


def _render_still(output: str, duration: float, fps: int, w: int, h: int) -> None:
    # still: compute the complete golden-angle head once and hold the same beauty frame.
    steps = max(1, int(duration * fps))
    xy, t, sizes = _compute_head(DEFAULT_N, GOLDEN_ANGLE_DEG, (0, 0, w, h), margin=0.065)
    colors = _gradient(t)
    frame = _finish(_draw_head(w, h, xy, colors, sizes, label="golden angle phyllotaxis"), w, h)
    with _writer(output, fps) as wr:
        for _ in range(steps):
            wr.append_data(frame)


def _render_angles(output: str, duration: float, fps: int, w: int, h: int) -> None:
    # angles: compute each fixed-angle panel once, then hold the comparison frame.
    steps = max(1, int(duration * fps))
    canvas = _new_rgba(w, h)
    specs = (
        ("1/5 turn (72 deg)", 72.0),
        ("137.0 deg", 137.0),
        ("137.5 deg (golden)", GOLDEN_ANGLE_DEG),
    )
    for i, (label, angle) in enumerate(specs):
        box = _panel_box(i, w, h)
        xy, t, sizes = _compute_head(DEFAULT_N, angle, box, margin=0.105)
        colors = _gradient(t)
        panel = _draw_head(w, h, xy, colors, sizes * 0.78, label=label, label_xy=(box[0] + 18, box[1] + 16), label_width=box[2] - box[0])
        canvas = Image.alpha_composite(canvas, panel)
    frame = _finish(canvas, w, h)
    with _writer(output, fps) as wr:
        for _ in range(steps):
            wr.append_data(frame)


def _spiral_colors(t: np.ndarray, n: int) -> np.ndarray:
    idx = np.arange(n)
    base = _gradient(t).astype(np.float32) * 0.42
    fam21 = ((idx % 21) / 20.0)[:, None]
    fam34 = ((idx % 34) / 33.0)[:, None]
    cyan = np.array((45, 230, 255), dtype=np.float32)
    magenta = np.array((255, 72, 218), dtype=np.float32)
    gold = np.array((255, 207, 92), dtype=np.float32)
    color = base + cyan * (0.25 + 0.55 * (fam21 < 0.20)) + magenta * (0.20 + 0.50 * (fam34 > 0.80))
    color += gold * (0.18 * (1 - t[:, None]))
    return np.clip(color, 0, 255).astype(np.uint8)


def _render_spirals(output: str, duration: float, fps: int, w: int, h: int) -> None:
    # spirals: precompute golden head once; Fibonacci modulo tints expose two parastichy families.
    steps = max(1, int(duration * fps))
    xy, t, sizes = _compute_head(DEFAULT_N, GOLDEN_ANGLE_DEG, (0, 0, w, h), margin=0.075)
    colors = _spiral_colors(t, DEFAULT_N)
    img = _draw_head(w, h, xy, colors, sizes * 1.05, label="spiral arms")
    frame = _finish(img, w, h)
    with _writer(output, fps) as wr:
        for _ in range(steps):
            wr.append_data(frame)


def render(mode: str, output: str, duration: float, fps: int, width: int, height: int) -> None:
    if mode == "grow":
        _render_grow(output, duration, fps, width, height)
    elif mode == "still":
        _render_still(output, duration, fps, width, height)
    elif mode == "angles":
        _render_angles(output, duration, fps, width, height)
    elif mode == "spirals":
        _render_spirals(output, duration, fps, width, height)
    else:
        raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grow", choices=["grow", "still", "angles", "spirals"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    render(args.mode, args.output, args.duration, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
