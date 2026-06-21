"""Procedural stylized campfire / flame generator (numpy, no footage, no GPU).

Not a physical combustion solve (that needs volumetric fluid on a GPU). This is a
fast *stylized* fire: a heat field that rises with cooling + per-pixel lateral lick
(the classic "doom fire" cellular update), mapped through a warm blackbody-ish palette,
with additive bloom, rising embers, and a dark campfire scene (log silhouettes + base glow).

Goal: a beautiful generative-art flame (monetization-safe = 100% original), NOT a
photo-real campfire. Cheap enough for the VM under run_capped.

Modes:
  python fire_simulator.py --mode loop  --output outputs/fire.mp4 --seconds 16
  python fire_simulator.py --mode still --output outputs/fire.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import imageio.v2 as imageio

OUT_W, OUT_H = 1280, 720
FW, FH = 480, 270            # heat-field resolution (upscaled to OUT)
FPS = 24


# ---- warm fire palette: heat 0..1 -> RGB (black -> deep red -> orange -> white-hot) ----
def _build_palette() -> np.ndarray:
    stops = [
        (0.00, (0, 0, 0)),
        (0.12, (28, 6, 2)),
        (0.26, (120, 22, 6)),
        (0.42, (200, 56, 8)),
        (0.58, (244, 110, 18)),
        (0.74, (255, 170, 52)),
        (0.88, (255, 224, 130)),
        (1.00, (255, 248, 222)),
    ]
    lut = np.zeros((256, 3), np.float32)
    xs = np.linspace(0, 1, 256)
    pos = [s[0] for s in stops]
    cols = [s[1] for s in stops]
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, [col[c] for col in cols])
    return lut


_PALETTE = _build_palette()


def _base_sources(width: int) -> np.ndarray:
    """Two clustered gaussians near center-bottom -> a localized campfire (not a wall of fire)."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    prof = np.zeros(width, np.float32)
    for cx, w, a in ((0.45, 0.052, 1.0), (0.55, 0.054, 0.96), (0.50, 0.085, 0.7)):
        prof += a * np.exp(-((x - cx) ** 2) / (2 * w * w))
    return np.clip(prof, 0, 1).astype(np.float32)


def _scene_background() -> np.ndarray:
    """Dark night canvas + faint warm radial glow at the fire base + crossed log silhouettes."""
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W].astype(np.float32)
    bg = np.zeros((OUT_H, OUT_W, 3), np.float32)
    bg[:] = (6, 7, 12)
    # warm ground glow centered at the fire base
    bx, by = OUT_W * 0.5, OUT_H * 0.82
    r2 = ((xx - bx) / (OUT_W * 0.34)) ** 2 + ((yy - by) / (OUT_H * 0.30)) ** 2
    glow = np.exp(-r2)
    bg += np.array([70, 34, 10], np.float32) * glow[..., None]
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # two dark logs crossing under the flame
    cx, cy = OUT_W * 0.5, OUT_H * 0.88
    for ang, col in ((18, (34, 22, 16)), (-18, (28, 18, 13))):
        dx = np.cos(np.radians(ang)) * 210
        dy = np.sin(np.radians(ang)) * 210
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=col, width=26)
        d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(54, 30, 14), width=8)
    return np.asarray(img, np.float32)


def _step_fire(heat: np.ndarray, rng: np.random.Generator, base: np.ndarray, flick: float, wind: int) -> None:
    """In-place one cellular step: rows rise, cool, and lick sideways. heat[-1]=bottom (hottest)."""
    h, w = heat.shape
    below = heat[1:, :]                                   # rows just under each target row
    # per-pixel lateral spread (-1,0,1) biased by a slow global wind -> flickering tongues
    shift = rng.integers(-1, 2, size=(h - 1, w)) + wind
    xs = np.clip(np.arange(w)[None, :] + shift, 0, w - 1)
    sampled = np.take_along_axis(below, xs, axis=1)
    # horizontal smoothing -> fuller, less streaky flame (3-tap blur)
    sampled = 0.5 * sampled + 0.25 * (np.roll(sampled, 1, axis=1) + np.roll(sampled, -1, axis=1))
    # gentle cooling -> tall licking flames (row 0 = top cools a touch more)
    cool = (rng.random((h - 1, w)).astype(np.float32) * 0.0065
            + (np.linspace(0.0090, 0.0009, h - 1))[:, None].astype(np.float32))
    heat[:-1] = np.clip(sampled - cool, 0.0, 1.0)
    # reseed the bottom row with the flickering campfire sources
    heat[-1] = np.clip(base * flick + rng.random(w).astype(np.float32) * 0.05, 0, 1)


def _render(heat: np.ndarray, bg: np.ndarray, embers, draw_embers) -> np.ndarray:
    idx = np.clip((heat * 255).astype(np.int32), 0, 255)
    fire_small = _PALETTE[idx]                            # (FH,FW,3)
    fire = np.asarray(Image.fromarray(fire_small.astype(np.uint8)).resize((OUT_W, OUT_H), Image.BILINEAR), np.float32)
    # additive composite over the dark scene, then bloom for glow
    comp = np.clip(bg + fire, 0, 255).astype(np.uint8)
    img = Image.fromarray(comp)
    bloom = img.filter(ImageFilter.GaussianBlur(radius=9))
    arr = np.clip(np.asarray(img, np.float32) + 0.5 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    draw_embers(ImageDraw.Draw(out, "RGBA"), embers)
    return np.asarray(out, np.uint8)


def _ember_funcs(rng):
    embers = []  # each: [x, y, vx, vy, life, life0]

    def update(top_y_frac):
        # spawn a couple from the flame top
        for _ in range(rng.integers(0, 3)):
            embers.append([OUT_W * (0.46 + rng.random() * 0.08),
                           OUT_H * (0.55 + rng.random() * 0.08),
                           (rng.random() - 0.5) * 0.8, -1.6 - rng.random() * 1.4,
                           1.0, 1.0])
        alive = []
        for e in embers:
            e[2] += (rng.random() - 0.5) * 0.12          # turbulent drift
            e[3] += 0.012                                 # slight gravity vs buoyancy
            e[0] += e[2]; e[1] += e[3]
            e[4] -= 0.012 + rng.random() * 0.006
            if e[4] > 0 and e[1] > OUT_H * 0.15:
                alive.append(e)
        embers[:] = alive

    def draw(d, _embers):
        for x, y, vx, vy, life, _ in _embers:
            a = int(220 * max(0.0, min(1.0, life)))
            r = 1.4 + 1.6 * life
            col = (255, 200 + int(40 * life), 120, a)
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)

    return embers, update, draw


def render_loop(args):
    frames = max(2, int(args.seconds * args.fps))
    rng = np.random.default_rng(args.seed)
    base = _base_sources(FW)
    heat = np.zeros((FH, FW), np.float32)
    bg = _scene_background()
    embers, ember_update, ember_draw = _ember_funcs(rng)
    # warm up so the flame is established at frame 0
    for _ in range(160):
        t = rng.random()
        _step_fire(heat, rng, base, 0.8 + 0.2 * t, int(round(np.sin(t * 6) * 1)))

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for i in range(frames):
            u = i / frames
            flick = 0.78 + 0.22 * (0.5 + 0.5 * np.sin(u * 2 * np.pi * 3)) + 0.06 * np.sin(u * 2 * np.pi * 11)
            wind = int(round(np.sin(u * 2 * np.pi * 2) * 1.4))
            _step_fire(heat, rng, base, float(flick), wind)
            ember_update(0.55)
            writer.append_data(_render(heat, bg, embers, ember_draw))
    finally:
        writer.close()
    print(f"[OK] stylized campfire: {frames} frames @ {FW}x{FH}->{OUT_W}x{OUT_H} + bloom + embers -> {out}")


def render_still(args):
    rng = np.random.default_rng(args.seed)
    base = _base_sources(FW)
    heat = np.zeros((FH, FW), np.float32)
    bg = _scene_background()
    embers, ember_update, ember_draw = _ember_funcs(rng)
    for _ in range(220):
        _step_fire(heat, rng, base, 0.85, int(round(rng.random() - 0.5)))
        ember_update(0.55)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(heat, bg, embers, ember_draw)).save(out)
    print(f"[OK] campfire still -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    (render_loop if args.mode == "loop" else render_still).__call__(args)


if __name__ == "__main__":
    main()
