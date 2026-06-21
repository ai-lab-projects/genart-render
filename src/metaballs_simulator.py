"""Metaballs (lava-lamp style blobs) — numpy, full-res, perfectly seamless loop.

Type-B ambient. A handful of soft "charges" move along closed Lissajous/circular paths whose
frequencies are INTEGER cycles per loop, so every center returns exactly to its start at u=1
-> no seam. The summed inverse-square-ish potential field is thresholded smoothly to give round,
merging blob boundaries (the classic metaball look). Calm deep blue/teal/violet body with a
little warm amber mixed into the hot blob cores -> liquid lava-lamp feel + bloom.

Analytic per-frame (sum over a few sources, O(W*H)) -> fast, no iterative sim.

Modes:
  python metaballs_simulator.py --mode still --output out.png
  python metaballs_simulator.py --mode loop  --output out.mp4 --seconds 24
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import imageio.v2 as imageio

OUT_W, OUT_H = 1280, 720
FPS = 30
TAU = np.float32(2.0 * np.pi)
BG = (4, 6, 13)  # #04060d dark base

# field 0..1 -> dark base -> deep blue -> teal -> cyan -> violet, hot cores warm amber
_STOPS = [(0.00, (4, 6, 13)), (0.34, (8, 30, 64)), (0.56, (16, 84, 120)),
          (0.72, (38, 146, 172)), (0.83, (110, 190, 206)),
          (0.91, (160, 140, 210)), (0.97, (224, 168, 110)), (1.00, (238, 196, 138))]


def _palette():
    xs = np.linspace(0, 1, 256)
    pos = [s[0] for s in _STOPS]
    cols = [s[1] for s in _STOPS]
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [col[c] for col in cols])
    return lut


_LUT = _palette()


def _build_balls(seed):
    rng = np.random.default_rng(seed)
    balls = []
    n = 6
    for i in range(n):
        cx = float(rng.uniform(0.30, 0.70))
        cy = float(rng.uniform(0.30, 0.70))
        ax = float(rng.uniform(0.14, 0.34))      # lissajous amplitude x
        ay = float(rng.uniform(0.14, 0.30))      # lissajous amplitude y
        fx = int(rng.integers(1, 3))              # INTEGER cycles -> seamless
        fy = int(rng.integers(1, 3))              # INTEGER cycles -> seamless
        phx = float(rng.uniform(0, 2 * math.pi))
        phy = float(rng.uniform(0, 2 * math.pi))
        strength = float(rng.uniform(0.020, 0.040))  # ~ (radius)^2 of the blob
        balls.append((cx, cy, ax, ay, fx, fy, phx, phy, strength))
    return balls


def _field(u, balls):
    """Metaball potential mapped to 0..1 at loop phase u. Integer freqs -> seamless."""
    asp = np.float32(OUT_W) / np.float32(OUT_H)
    x = (np.linspace(0, 1, OUT_W, dtype=np.float32) * asp)[None, :]
    y = np.linspace(0, 1, OUT_H, dtype=np.float32)[:, None]
    field = np.zeros((OUT_H, OUT_W), np.float32)
    for cx, cy, ax, ay, fx, fy, phx, phy, strength in balls:
        bx = (np.float32(cx) + np.float32(ax) * np.cos(TAU * np.float32(fx) * np.float32(u) + np.float32(phx))) * asp
        by = np.float32(cy) + np.float32(ay) * np.sin(TAU * np.float32(fy) * np.float32(u) + np.float32(phy))
        d2 = (x - bx) ** 2 + (y - by) ** 2 + np.float32(1e-4)
        field += np.float32(strength) / d2
    # smooth threshold: tanh-like squash gives round blob edges that merge organically.
    # offset>1 keeps the body in the deep-blue/teal range so cores reach amber but not blown white.
    n = field / (field + np.float32(2.2))
    n = np.clip(n * np.float32(1.6), 0, 1)
    # gamma keeps large dark/blue area (calm) and confines bright cores to the blob centers
    return np.power(n, np.float32(1.25)).astype(np.float32)


def _render(u, balls):
    f = _field(u, balls)
    idx = (f * 255).astype(np.int32)
    rgb = _LUT[idx]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    # cheap wide bloom: blur a half-res copy then upscale (large effective radius, low cost)
    small = img.resize((OUT_W // 2, OUT_H // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=5))
    bloom = np.asarray(small.resize((OUT_W, OUT_H), Image.BILINEAR), np.float32)
    arr = np.clip(np.asarray(img, np.float32) + 0.28 * bloom, 0, 255).astype(np.uint8)
    return arr


def render_loop(args):
    frames = max(2, int(args.seconds * args.fps))
    balls = _build_balls(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for i in range(frames):
            writer.append_data(_render(np.float32(i / frames), balls))
    finally:
        writer.close()
    print(f"[OK] metaballs {frames} frames seamless @ {OUT_W}x{OUT_H} -> {out}")


def render_still(args):
    balls = _build_balls(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(np.float32(0.18), balls)).save(args.output)
    print(f"[OK] metaballs still -> {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    (render_loop if args.mode == "loop" else render_still)(args)


if __name__ == "__main__":
    main()
