"""Procedural Reaction-Diffusion (Gray-Scott) — numpy, abstract organic patterns that GROW.

Pure abstract generative art (no real-world referent -> not compared to 'the real thing',
unlike aurora/fire — see topic_backlog 'imitation difficulty'). Two virtual chemicals react and
diffuse; coral / labyrinth / spot patterns emerge and slowly morph. Calm, hypnotic, ambient.

Sim runs on a coarse grid (fast) and is upscaled with bloom for a soft, premium look.

Modes:
  python reaction_diffusion_simulator.py --mode still --output out.png
  python reaction_diffusion_simulator.py --mode loop  --output out.mp4 --seconds 30
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import imageio.v2 as imageio

OUT_W, OUT_H = 2560, 1440
GW, GH = 256, 144              # coarse sim grid -> bigger, calmer features when upscaled (less busy)
FPS = 24
DU, DV, DT = 0.16, 0.08, 1.0
FEED, KILL = 0.0395, 0.0598    # 'labyrinth/Turing' regime — whole-grid noise self-organizes into a maze
TAU = np.float32(2.0 * np.pi)

# calm palette: deep indigo (low V) -> teal -> soft cream (high V)
_STOPS = [(0.0, (8, 14, 34)), (0.35, (16, 60, 92)), (0.6, (34, 140, 150)),
          (0.82, (120, 205, 200)), (1.0, (235, 245, 230))]


def _palette():
    xs = np.linspace(0, 1, 256); pos = [s[0] for s in _STOPS]; cols = [s[1] for s in _STOPS]
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [col[c] for col in cols])
    return lut


_LUT = _palette()


def _laplacian(a):
    return (-a
            + 0.20 * (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1))
            + 0.05 * (np.roll(np.roll(a, 1, 0), 1, 1) + np.roll(np.roll(a, 1, 0), -1, 1)
                      + np.roll(np.roll(a, -1, 0), 1, 1) + np.roll(np.roll(a, -1, 0), -1, 1)))


def _seed(rng):
    # seed V across the WHOLE grid (Turing instability fills the frame with a maze, not sparse spots)
    U = np.ones((GH, GW), np.float32)
    mask = (rng.random((GH, GW)) > 0.62).astype(np.float32)
    mask = box_lowpass2(mask, 2)                   # soften so seeds are blobby, not single-pixel
    V = (mask > 0.4).astype(np.float32) * 0.5
    U = 1.0 - 0.5 * (mask > 0.4).astype(np.float32)
    V += rng.random((GH, GW)).astype(np.float32) * 0.02
    return U.astype(np.float32), V.astype(np.float32)


def box_lowpass2(a, win):
    k = np.ones((2 * win + 1,), np.float32) / (2 * win + 1)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
    a = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, a)
    return a


def _step(U, V, n, feed=FEED, kill=KILL):
    for _ in range(n):
        uvv = U * V * V
        U += (DU * _laplacian(U) - uvv + feed * (1 - U)) * DT
        V += (DV * _laplacian(V) + uvv - (feed + kill) * V) * DT
        np.clip(U, 0, 1, out=U); np.clip(V, 0, 1, out=V)
    return U, V


def _colorize(V):
    v = np.clip(V / 0.45, 0, 1)                       # normalize typical V range
    idx = (v * 255).astype(np.int32)
    small = _LUT[idx]
    img = Image.fromarray(small.astype(np.uint8)).resize((OUT_W, OUT_H), Image.LANCZOS)   # crisp (no extra blur)
    bloom = img.filter(ImageFilter.GaussianBlur(radius=10))
    arr = np.clip(np.asarray(img, np.float32) + 0.3 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)
    return arr


def render_still(args):
    rng = np.random.default_rng(args.seed)
    U, V = _seed(rng)
    _step(U, V, 8000)                                 # develop the maze (stable by ~6000)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_colorize(V)).save(args.output)
    print(f"[OK] reaction-diffusion still -> {args.output}")


def render_loop(args):
    frames = max(2, int(args.seconds * args.fps))
    rng = np.random.default_rng(args.seed)
    U, V = _seed(rng)
    _step(U, V, 3000)                                 # pre-develop so frame 0 already has structure
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for i in range(frames):
            writer.append_data(_colorize(V))
            # slowly oscillate feed/kill -> the maze perpetually melts & reforms = visible calm motion
            u = i / frames
            feed = 0.039 + 0.006 * np.sin(TAU * u) + 0.003 * np.sin(TAU * 2 * u)
            kill = 0.0605 + 0.004 * np.sin(TAU * u + 1.3)
            _step(U, V, 40, feed, kill)               # more sim time/frame -> faster apparent morph
    finally:
        writer.close()
    print(f"[OK] reaction-diffusion {frames} frames (evolving) @ {OUT_W}x{OUT_H} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    (render_loop if args.mode == "loop" else render_still)(args)


if __name__ == "__main__":
    main()
