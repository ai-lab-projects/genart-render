"""Procedural aurora borealis (northern lights) — numpy, synthesized, seamless loop.

Nature-inspired generative art (not photoreal): drifting vertical curtains of green->cyan->magenta
light over a dark starfield. Easy to make beautiful (wave math) and SEAMLESS (integer-cycle phases),
matching the Ambient Pixels brand. Pairs with the lo-fi/piano + rain soundtrack.

Curtains = a 1D fold density over x (sum of drifting sines) shaped into vertical rays with a wavy
baseline; colour shifts with height. Additive glow + bloom. Stars + sky gradient + faint ground glow.

Modes:
  python aurora_simulator.py --mode still --output out.png
  python aurora_simulator.py --mode loop  --output out.mp4 --seconds 24
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import imageio.v2 as imageio

OUT_W, OUT_H = 2560, 1440      # 1440p (detailed content needs the pixels + YouTube bitrate)
SS = 1                         # compute at native; LANCZOS not needed since we render at OUT res
FPS = 24
TAU = np.float32(2.0 * np.pi)

# aurora colour ramp by height within the curtain (0=base ... 1=top of rays)
C_BASE = np.array([10, 80, 40], np.float32)     # deep green base glow
C_MID = np.array([40, 240, 150], np.float32)    # bright green
C_HI = np.array([90, 200, 255], np.float32)     # cyan
C_TIP = np.array([170, 90, 220], np.float32)    # magenta/violet tips


def _stars(rng, w, h):
    sky = np.zeros((h, w, 3), np.float32)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky += np.array([3, 5, 12], np.float32) + np.array([0, 4, 16], np.float32) * (1 - yy)   # night gradient
    n = int(w * h / 5500)
    xs = rng.integers(0, w, n); ys = rng.integers(0, h, n)
    b = rng.uniform(0.3, 1.0, n).astype(np.float32)[:, None]
    sky[ys, xs] += np.array([200, 210, 235], np.float32) * b
    return sky


def _curtain_density(x, u):
    """1D fold pattern over x (0..1), drifting. Integer cycles -> seamless. Returns 0..~1 per column."""
    d = np.zeros_like(x)
    for k, sp, ph, amp in ((3.0, 1, 0.0, 1.0), (7.0, 2, 1.7, 0.6), (13.0, 1, 3.1, 0.4),
                           (21.0, 3, 4.4, 0.25), (2.0, 1, 5.0, 0.7)):
        d += amp * np.sin(TAU * (k * x) - TAU * sp * u + np.float32(ph))
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return d ** np.float32(2.0)                 # sharpen into distinct sheets


def _render(u, w, h, rng_sky):
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]      # 0 top .. 1 bottom
    dens = _curtain_density(xx, u)                            # (1,w)
    # wavy baseline (bottom of the curtain) drifts; rays rise upward from it
    base_y = 0.74 + 0.06 * np.sin(TAU * 2 * xx - TAU * u) + 0.03 * np.sin(TAU * 5 * xx + TAU * 2 * u)
    height = np.clip((base_y - yy) / 0.55, 0, 1)             # 0 at/below baseline-... ->1 high up
    band = np.clip(1.0 - np.abs(base_y - yy) / 0.62, 0, 1) ** 1.5   # vertical extent of glow
    # fine vertical ray texture (high-freq x flicker), drifting
    rays = 0.6 + 0.4 * np.sin(TAU * 60 * xx + 3 * np.sin(TAU * u + TAU * 4 * xx))
    intensity = (dens * band * rays).astype(np.float32)      # (h,w)
    intensity = np.clip(intensity * 1.3, 0, 1.4)
    # colour by height
    hgt = height[..., None]
    col = (C_BASE * (1 - np.clip(hgt / 0.33, 0, 1))
           + C_MID * np.clip(1 - np.abs(hgt - 0.33) / 0.33, 0, 1)
           + C_HI * np.clip(1 - np.abs(hgt - 0.66) / 0.33, 0, 1)
           + C_TIP * np.clip((hgt - 0.66) / 0.34, 0, 1))
    sky = rng_sky.copy()
    out = sky + col * intensity[..., None]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    bloom = img.filter(ImageFilter.GaussianBlur(radius=10))
    arr = np.clip(np.asarray(img, np.float32) + 0.6 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)
    return arr


def render_loop(args):
    frames = max(2, int(args.seconds * args.fps))
    rng = np.random.default_rng(args.seed)
    sky = _stars(rng, OUT_W, OUT_H)          # fixed starfield
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for i in range(frames):
            writer.append_data(_render(np.float32(i / frames), OUT_W, OUT_H, sky))
    finally:
        writer.close()
    print(f"[OK] aurora {frames} frames seamless @ {OUT_W}x{OUT_H} -> {out}")


def render_still(args):
    rng = np.random.default_rng(args.seed)
    sky = _stars(rng, OUT_W, OUT_H)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(np.float32(0.3), OUT_W, OUT_H, sky)).save(args.output)
    print(f"[OK] aurora still -> {args.output}")


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
