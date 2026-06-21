"""Procedural flowing 'plasma' — numpy, computed at full res (crisp), perfectly seamless loop.

Type-B ambient (quasi-steady, continuously flowing -> seamless infinite loop, ideal background;
see topic_backlog 'ambient loop types'). Abstract (no real referent). Sum of drifting sine fields
+ a couple of orbiting radial sources -> smooth liquid colour that endlessly flows. All time phases
are integer cycles -> the clip loops with no seam. Calm deep-blue / teal / violet palette + bloom.

Computed at OUTPUT resolution (no tiny-grid upscale -> no blur, unlike the RD experiment).

Modes:
  python plasma_simulator.py --mode still --output out.png
  python plasma_simulator.py --mode loop  --output out.mp4 --seconds 24
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import imageio.v2 as imageio

OUT_W, OUT_H = 2560, 1440
FPS = 24
TAU = np.float32(2.0 * np.pi)

# calm palette across the plasma value 0..1 (deep night-blue -> teal -> soft cyan -> violet)
_STOPS = [(0.00, (6, 16, 40)), (0.30, (14, 56, 96)), (0.55, (28, 128, 150)),
          (0.75, (120, 200, 210)), (0.90, (140, 120, 210)), (1.0, (60, 40, 96))]


def _palette():
    xs = np.linspace(0, 1, 256); pos = [s[0] for s in _STOPS]; cols = [s[1] for s in _STOPS]
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [col[c] for col in cols])
    return lut


_LUT = _palette()


def _field(u):
    """Plasma scalar field at loop phase u in [0,1). Integer time cycles -> seamless."""
    x = np.linspace(0, 1, OUT_W, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, OUT_H, dtype=np.float32)[:, None]
    asp = np.float32(OUT_W) / np.float32(OUT_H)
    xa = x * asp
    v = np.zeros((OUT_H, OUT_W), np.float32)
    # drifting plane waves (k = spatial freq, c = integer cycles over the loop)
    for k, ky, c, ph in ((2.2, 0.0, 1, 0.0), (0.0, 2.6, 1, 1.7), (1.6, 1.6, 2, 3.1), (3.1, -1.4, 2, 4.4)):
        v += np.sin(TAU * (k * xa + ky * y) - TAU * c * u + np.float32(ph))
    # two slowly ORBITING radial sources -> liquid swirling (centers move on circles = seamless)
    for c, ph, rad, amp in ((1, 0.0, 0.9, 1.3), (1, 2.5, 1.3, 1.0)):
        cx = 0.5 * asp + 0.32 * asp * np.cos(TAU * c * u + np.float32(ph))
        cy = 0.5 + 0.30 * np.sin(TAU * c * u + np.float32(ph) * 1.3)
        r = np.sqrt((xa - cx) ** 2 + (y - cy) ** 2)
        v += amp * np.sin(TAU * rad * r - TAU * c * u + np.float32(ph))
    # finer detail layer (higher freq) so it's not just soft blobs
    for k, ky, c, ph in ((6.0, 2.0, 3, 0.6), (-3.0, 7.0, 2, 2.2), (9.0, -4.0, 4, 5.0)):
        v += 0.5 * np.sin(TAU * (k * xa + ky * y) - TAU * c * u + np.float32(ph))
    n = np.clip(0.5 + v / 10.0, 0, 1)
    # ripple the value mapping -> crisp flowing contour bands (defined edges, not pure blur)
    return 0.5 + 0.5 * np.sin(TAU * 1.5 * n).astype(np.float32)


def _render(u):
    idx = (_field(u) * 255).astype(np.int32)
    img = Image.fromarray(_LUT[idx].astype(np.uint8))
    bloom = img.filter(ImageFilter.GaussianBlur(radius=6))               # less bloom = crisper
    arr = np.clip(np.asarray(img, np.float32) + 0.22 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)
    return arr


def render_loop(args):
    frames = max(2, int(args.seconds * args.fps))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for i in range(frames):
            writer.append_data(_render(np.float32(i / frames)))
    finally:
        writer.close()
    print(f"[OK] plasma {frames} frames seamless @ {OUT_W}x{OUT_H} -> {out}")


def render_still(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(np.float32(0.2))).save(args.output)
    print(f"[OK] plasma still -> {args.output}")


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
