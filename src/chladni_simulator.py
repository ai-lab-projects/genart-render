"""Chladni figure mp4 generator. Sand gathers along nodal lines of a vibrating plate.

Square-plate approximation on [0,1]^2:
  z(x,y) = cos(n*pi*x)*cos(m*pi*y) - cos(m*pi*x)*cos(n*pi*y)

Sand is brightest near z == 0, where the plate is still enough for grains to settle.

Usage:
  python chladni_simulator.py --mode sweep --output path/chladni.mp4
  python chladni_simulator.py --mode still --output path/chladni_hero.mp4 --duration 18
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = np.array([8, 8, 12], dtype=np.float32)
SAND = np.array([238, 228, 190], dtype=np.float32)
GLOW = np.array([246, 208, 105], dtype=np.float32)
CYAN = np.array([80, 205, 225], dtype=np.float32)

SWEEP_MODES = [(1, 2), (2, 3), (3, 4), (2, 5), (4, 5), (3, 7), (5, 6), (4, 7), (6, 7)]
STILL_MODE = (5, 9)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _plate_layout(width: int, height: int) -> tuple[int, int, int, int]:
    side = int(min(width, height) * 0.88)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    return x0, y0, side, side


def _sand_mask(n: int, m: int, side: int, supersample: int = 2) -> np.ndarray:
    hi = side * supersample
    axis = np.linspace(0.0, 1.0, hi, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    z = np.cos(n * np.pi * x) * np.cos(m * np.pi * y)
    z -= np.cos(m * np.pi * x) * np.cos(n * np.pi * y)

    fine = np.exp(-((z / 0.095) ** 2))
    wide = np.exp(-((z / 0.18) ** 2))
    edge_fade = _smoothstep(0.0, 0.045, x) * _smoothstep(0.0, 0.045, y)
    edge_fade *= _smoothstep(0.0, 0.045, 1.0 - x) * _smoothstep(0.0, 0.045, 1.0 - y)
    mask = np.clip(0.86 * fine + 0.22 * wide, 0.0, 1.0) * edge_fade

    img = Image.fromarray((mask * 255).astype(np.uint8), "L")
    img = img.resize((side, side), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def _colorize(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[:] = BG
    x0, y0, side, _ = _plate_layout(width, height)

    yy = np.linspace(-1.0, 1.0, side, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, side, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    vignette = np.clip(1.0 - 0.22 * (gx * gx + gy * gy), 0.70, 1.0)
    plate = BG + np.array([10, 10, 14], dtype=np.float32) * vignette[..., None]

    glow = np.clip(mask ** 0.45, 0.0, 1.0)
    core = np.clip(mask ** 1.35, 0.0, 1.0)
    rim = np.clip((mask - 0.12) / 0.60, 0.0, 1.0) ** 2.0
    plate = plate * (1.0 - 0.32 * glow[..., None]) + GLOW * (0.18 * glow[..., None])
    plate = plate * (1.0 - 0.92 * core[..., None]) + SAND * (0.92 * core[..., None])
    plate += CYAN * (0.035 * rim[..., None])

    frame[y0 : y0 + side, x0 : x0 + side] = plate
    border = np.clip(mask.max(axis=0).mean() * 0 + 1, 0, 1)
    frame[y0 : y0 + 2, x0 : x0 + side] += 24 * border
    frame[y0 + side - 2 : y0 + side, x0 : x0 + side] += 10 * border
    frame[y0 : y0 + side, x0 : x0 + 2] += 18 * border
    frame[y0 : y0 + side, x0 + side - 2 : x0 + side] += 18 * border
    return np.clip(frame, 0, 255).astype(np.uint8)


def _add_label(frame: np.ndarray, text: str, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _font(max(18, width // 54))
    pad = max(18, width // 58)
    shadow = (0, 0, 0)
    fill = (224, 232, 226)
    draw.text((pad + 1, pad + 1), text, font=font, fill=shadow)
    draw.text((pad, pad), text, font=font, fill=fill)
    return np.asarray(img, dtype=np.uint8)


def _ease(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _blend_masks(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    eased = _ease(alpha)
    blended = a * (1.0 - eased) + b * eased
    sharpen = 0.92 + 0.22 * abs(2.0 * eased - 1.0)
    return np.clip(blended * sharpen, 0.0, 1.0)


def render(mode: str, output: str, duration: float, fps: int, width: int, height: int) -> None:
    frames = max(1, int(duration * fps))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _, _, side, _ = _plate_layout(width, height)

    # Cost rule: Chladni cos fields are vectorized and computed once per figure.
    # The per-frame path only blends cached masks, colorizes, and writes uint8 frames.
    if mode == "still":
        masks = [_sand_mask(*STILL_MODE, side)]
    else:
        masks = [_sand_mask(n, m, side) for n, m in SWEEP_MODES]

    writer = imageio.get_writer(
        str(out),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=8,
    )
    try:
        for i in range(frames):
            if mode == "still":
                mask = masks[0]
                label = f"mode ({STILL_MODE[0]}, {STILL_MODE[1]})"
            else:
                pos = (i / max(1, frames - 1)) * len(masks)
                idx = min(int(pos), len(masks) - 1)
                local = pos - idx
                hold = 0.72
                if idx < len(masks) - 1 and local > hold:
                    mask = _blend_masks(masks[idx], masks[idx + 1], (local - hold) / (1.0 - hold))
                else:
                    mask = masks[idx]
                label = "a higher note -> a new figure"

            frame = _colorize(mask, width, height)
            frame = _add_label(frame, label, width, height)
            writer.append_data(frame)
    finally:
        writer.close()
    print(f"[OK] {mode} -> {out} ({frames}f @ {fps}fps, {width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sweep", choices=["sweep", "still"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    render(args.mode, args.output, args.duration, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
