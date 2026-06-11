"""Procedural shimmering water surface (sun glitter / caustic sparkle) — numpy, no footage, no GPU.

Why water (vs fire): water is abstract and wave-math-friendly, so a procedural version reads as
convincing far more easily than a campfire (user 2026-06-05: fire is high imitation-difficulty,
water is low). Also matches proven aquarium/water ambient demand.

Method: a height field = sum of moving sine/Gerstner-ish waves -> surface normals -> Blinn-Phong
specular glints toward a light = the dancing shimmer. Bloom makes the sparkles glow.

SEAMLESS LOOP: every wave's temporal frequency is an integer number of cycles per loop period,
so h(t+T) == h(t) exactly -> perfect loop (a clean advantage of wave-based water over fire).

Modes:
  python water_simulator.py --mode loop  --output outputs/water.mp4 --seconds 16
  python water_simulator.py --mode still --output outputs/water.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import imageio.v2 as imageio

OUT_W, OUT_H = 2560, 1440    # output 1440p (2K): detailed content needs the pixels + YouTube gives it way more bitrate than 720p
GW, GH = 2880, 1620          # compute 1.125x then LANCZOS-downscale -> light AA at 1440p (keeps render time reasonable)
FPS = 24


# wave components: (dir_x, dir_y, wavelength, amplitude, cycles_per_loop)
# cycles_per_loop must be INTEGER -> seamless loop. Mostly large/medium waves; few fine ones
# (too many fine waves -> uniform 'snow' sparkle instead of water glitter).
_WAVES = [
    (1.00, 0.10, 4.0, 0.60, 2),
    (-0.30, 1.00, 2.6, 0.46, 3),
    (0.70, -0.70, 1.7, 0.32, 4),
    (-0.85, -0.40, 1.15, 0.24, 6),
    (0.20, 1.00, 0.80, 0.17, 8),
    (1.00, -0.25, 0.58, 0.12, 11),
]

# palette (calm, richer blue lake — deeper body, softer highlights for an elegant look)
# selectable palettes (DEEP troughs, SHAL crests, SUN glint, SKY reflection) — same engine, new mood
PALETTES = {
    "teal":   ([5, 24, 48], [22, 104, 142], [220, 240, 255], [40, 84, 120]),     # default cool blue-teal
    "amber":  ([28, 16, 8], [120, 60, 14], [255, 226, 150], [90, 50, 20]),       # warm sunset-gold water
    "violet": ([16, 10, 36], [70, 36, 120], [220, 200, 255], [60, 40, 100]),     # dusk violet/indigo
    "emerald":([4, 28, 22], [16, 120, 90], [200, 255, 230], [30, 90, 70]),       # deep emerald/jade
}
DEEP = np.array(PALETTES["teal"][0], np.float32)
SHAL = np.array(PALETTES["teal"][1], np.float32)
SUN = np.array(PALETTES["teal"][2], np.float32)
SKY = np.array(PALETTES["teal"][3], np.float32)


def _grid():
    # world coords; wider in x for a calm lake feel, slight perspective compression toward top
    x = np.linspace(0.0, 12.0, GW, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 7.0, GH, dtype=np.float32)[:, None]
    return np.broadcast_to(x, (GH, GW)).copy(), np.broadcast_to(y, (GH, GW)).copy()


def _surface(X, Y, t, period):
    """Return height h and slope (dh/dx, dh/dy) at time t (seconds), seamless over [0,period]."""
    h = np.zeros((GH, GW), np.float32)
    hx = np.zeros((GH, GW), np.float32)
    hy = np.zeros((GH, GW), np.float32)
    for dx, dy, wl, amp, cyc in _WAVES:
        n = np.hypot(dx, dy)
        dxx, dyy = dx / n, dy / n
        k = np.float32(2.0 * np.pi / wl)
        omega = np.float32(2.0 * np.pi * cyc / period)
        phase = k * (dxx * X + dyy * Y) - omega * np.float32(t)
        s = np.sin(phase)
        c = np.cos(phase)
        h += amp * s
        hx += amp * k * dxx * c
        hy += amp * k * dyy * c
    return h, hx, hy


# caustic animation phases: integer cycles per loop -> seamless. Small = slow drift.
_CAUSTIC_CYC = (1, 1, 2, 3, 2, 3, 1)
TAU = np.float32(2.0 * np.pi)


def _shade(X, Y, t, period):
    """Underwater caustics (the bright moving light-net of a pool). Most recognizable 'water'.
    Iterative warp of a periodic field -> sharp focused ridges = caustic web. Loops via integer cycles."""
    u = np.float32((t / period) % 1.0)
    freq = np.float32(3.6)                                   # caustic cells across the frame (lower = bigger, calmer)
    aspect = np.float32(OUT_W) / np.float32(OUT_H)
    off = np.float32(250.0)                                  # large offset keeps the 1/length term tame (per the classic)
    uvx = X / np.float32(12.0)                               # 0..1 across width
    uvy = Y / np.float32(7.0)                                # 0..1 down height
    # low-frequency domain warp breaks the regular tiling into organic ripples (seamless: integer cycles)
    wu = TAU * u
    warpx = np.float32(0.9) * np.sin(uvy * TAU * np.float32(2.0) + wu) + np.float32(0.6) * np.cos(uvx * TAU + wu * np.float32(2.0))
    warpy = np.float32(0.9) * np.cos(uvx * TAU * np.float32(2.0) - wu) + np.float32(0.6) * np.sin(uvy * TAU - wu * np.float32(2.0))
    px = uvx * TAU * freq * aspect + off + warpx
    py = uvy * TAU * freq + off + warpy
    ix, iy = px.copy(), py.copy()
    c = np.ones_like(px)
    inten = np.float32(0.0045)
    for n, cyc in enumerate(_CAUSTIC_CYC):
        a = TAU * u * np.float32(cyc) + np.float32(n * 1.7)
        nix = px + (np.cos(a - ix) + np.sin(a + iy))
        niy = py + (np.sin(a - iy) + np.cos(a + ix))
        ix, iy = nix, niy
        dx = px / (np.sin(ix + a) / inten + np.float32(1e-3))
        dy = py / (np.cos(iy + a) / inten + np.float32(1e-3))
        c += 1.0 / np.sqrt(dx * dx + dy * dy + np.float32(1e-6))
    c = c / np.float32(len(_CAUSTIC_CYC))
    tone = np.float32(1.17) - np.power(np.clip(c, 0, 5), np.float32(1.4))
    # remap to high-contrast thin caustic lines (bright where tone is high)
    web = np.clip((tone - np.float32(0.55)) / np.float32(0.50), 0, 1)
    bright = np.power(web, np.float32(3.8))                  # thinner, softer lines -> calmer, more blue body
    # depth gradient: darker blue foreground (bottom) -> lighter toward the top
    horizon = (Y / np.float32(GH))[..., None]
    base = DEEP * (np.float32(1.0) - horizon) + SHAL * horizon
    col = base + SUN * (bright * np.float32(0.9))[..., None]                       # softer glints (less blown white)
    col += (SHAL * np.float32(0.7)) * np.power(web, np.float32(1.2))[..., None]    # richer blue shimmer between lines
    return np.clip(col, 0, 255).astype(np.uint8)


def _post(rgb_small: np.ndarray) -> np.ndarray:
    img = Image.fromarray(rgb_small).resize((OUT_W, OUT_H), Image.LANCZOS)   # high-quality downscale = anti-alias
    bloom = img.filter(ImageFilter.GaussianBlur(radius=12))                  # scaled for 1440p
    arr = np.clip(np.asarray(img, np.float32) + 0.5 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)
    return arr


def render_loop(args):
    frames = max(2, int(args.seconds * args.fps))
    period = args.seconds                                   # one full seamless cycle == clip length
    X, Y = _grid()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for i in range(frames):
            t = period * i / frames
            writer.append_data(_post(_shade(X, Y, t, period)))
    finally:
        writer.close()
    print(f"[OK] shimmering water: {frames} frames, seamless (integer cycles/loop) @ {GW}x{GH}->{OUT_W}x{OUT_H} -> {out}")


def render_still(args):
    X, Y = _grid()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_post(_shade(X, Y, 3.0, 16.0))).save(out)
    print(f"[OK] water still -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--palette", default="teal", choices=list(PALETTES))
    args = ap.parse_args()
    global DEEP, SHAL, SUN, SKY
    p = PALETTES[args.palette]
    DEEP = np.array(p[0], np.float32); SHAL = np.array(p[1], np.float32)
    SUN = np.array(p[2], np.float32); SKY = np.array(p[3], np.float32)
    (render_loop if args.mode == "loop" else render_still)(args)


if __name__ == "__main__":
    main()
