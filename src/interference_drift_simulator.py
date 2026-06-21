"""Drifting wave-interference field — numpy, full-res, perfectly seamless loop.

Type-B ambient (quasi-steady, continuously flowing -> seamless infinite loop, ideal background).
Several point wave sources whose *centers orbit on circles* (integer revolutions per loop) emit
radial sine waves; their superposition is an interference pattern (moving fringes) that drifts
quietly. Every time term is `2*pi * c * u` with c an INTEGER cycle count and u in [0,1) the loop
phase, so frame at u=0 and the conceptual frame at u=1 are identical -> no seam. Calm deep
blue / teal / violet palette + bloom. Meditative.

Computed at OUTPUT resolution (no tiny-grid upscale). Analytic (no iterative sim) -> fast.

Modes:
  python interference_drift_simulator.py --mode still --output out.png
  python interference_drift_simulator.py --mode loop  --output out.mp4 --seconds 24
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

# deep-blue -> teal -> soft cyan -> violet, with a faint warm amber lift only at the very top.
# Weighted heavily toward the dark/blue end so the field reads as a calm deep body, not pastel.
_STOPS = [(0.00, (4, 6, 16)), (0.42, (7, 24, 52)), (0.64, (12, 58, 90)),
          (0.80, (24, 112, 144)), (0.90, (60, 160, 184)),
          (0.97, (110, 124, 196)), (1.00, (150, 150, 210))]


def _palette():
    xs = np.linspace(0, 1, 256)
    pos = [s[0] for s in _STOPS]
    cols = [s[1] for s in _STOPS]
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [col[c] for col in cols])
    return lut


_LUT = _palette()


# wave sources: (orbit_radius, orbit_phase, revolutions c (INTEGER), spatial_freq, amplitude)
_SOURCES = []


def _build_sources(seed):
    rng = np.random.default_rng(seed)
    srcs = []
    n = 5
    for i in range(n):
        rad = float(rng.uniform(0.10, 0.34))
        ph = float(rng.uniform(0, 2 * math.pi))
        rev = int(rng.integers(1, 3))            # 1 or 2 revolutions per loop -> integer -> seamless
        freq = float(rng.uniform(4.0, 8.0))       # spatial ripple freq (low -> calm broad fringes, no flicker)
        amp = float(rng.uniform(0.7, 1.1))
        cx0 = float(rng.uniform(0.30, 0.70))
        cy0 = float(rng.uniform(0.30, 0.70))
        srcs.append((cx0, cy0, rad, ph, rev, freq, amp))
    return srcs


def _field(u, sources):
    """Interference scalar in [0,1] at loop phase u. Integer revolutions -> seamless."""
    asp = np.float32(OUT_W) / np.float32(OUT_H)
    x = (np.linspace(0, 1, OUT_W, dtype=np.float32) * asp)[None, :]
    y = np.linspace(0, 1, OUT_H, dtype=np.float32)[:, None]
    v = np.zeros((OUT_H, OUT_W), np.float32)
    for cx0, cy0, rad, ph, rev, freq, amp in sources:
        # source center orbits a circle; integer rev count -> returns to start at u=1
        ang = TAU * np.float32(rev) * np.float32(u) + np.float32(ph)
        cx = np.float32(cx0) * asp + np.float32(rad) * asp * np.cos(ang)
        cy = np.float32(cy0) + np.float32(rad) * np.sin(ang)
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # traveling radial wave; temporal term = 1 cycle (slow) -> seamless. Motion comes mostly
        # from the orbiting center, so fringes DRIFT gently instead of sweeping fast (no flicker).
        v += np.float32(amp) * np.sin(TAU * np.float32(freq) * r - TAU * np.float32(u) + np.float32(ph))
    n = np.clip(0.5 + v / (2.4 * len(sources)), 0, 1)
    # soft fringe shaping (not a high-contrast contour -> calm, no per-frame flicker)
    fr = 0.5 + 0.5 * np.sin(TAU * 0.5 * n)
    # gamma > 1 pushes the body toward the deep/blue end -> not washed-out pastel
    return np.power(fr, np.float32(2.3)).astype(np.float32)


def _render(u, sources):
    f = _field(u, sources)
    idx = (f * 255).astype(np.int32)
    rgb = _LUT[idx]
    # blend toward dark BG in low-value regions for a deep, calm base
    dark = np.array(BG, np.float32)
    w = (f[..., None] ** 1.3)
    rgb = dark * (1.0 - w) + rgb * w
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    # bloom via downscale-blur-upscale: a small blur on a half-res image ~= a large blur on full
    # res, but far cheaper (GaussianBlur cost grows with radius). Keeps the soft glow, fast.
    small = img.resize((OUT_W // 2, OUT_H // 2), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=4))
    bloom = np.asarray(small.resize((OUT_W, OUT_H), Image.BILINEAR), np.float32)
    arr = np.clip(np.asarray(img, np.float32) + 0.22 * bloom, 0, 255).astype(np.uint8)
    return arr


def render_loop(args):
    frames = max(2, int(args.seconds * args.fps))
    sources = _build_sources(args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for i in range(frames):
            writer.append_data(_render(np.float32(i / frames), sources))
    finally:
        writer.close()
    print(f"[OK] interference drift {frames} frames seamless @ {OUT_W}x{OUT_H} -> {out}")


def render_still(args):
    sources = _build_sources(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(np.float32(0.18), sources)).save(args.output)
    print(f"[OK] interference drift still -> {args.output}")


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
