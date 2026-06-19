"""Cyclic cellular automaton mp4 generator for Agentic Pixels.

Each cell holds one state in a color cycle. On every synchronous tick, a cell
advances only when enough neighbors already show its successor state. Starting
from random noise, this simple local rule self-organizes into rotating spiral
waves.

Modes:
  evolve   - random noise into mature spiral waves
  still    - one held hero frame after burn-in
  birth    - early coalescing phase only
  gallery  - three side-by-side CCA variants

Usage:
  python cyclic_ca_simulator.py --mode evolve --output outputs/cca.mp4 --seconds 18
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BG = np.array([3, 5, 11], dtype=np.float32)
FPS = 24
WIDTH = 1280
HEIGHT = 720
GRID_W = 480
GRID_H = 270
SS = 1

NEIGHBOR_SHIFTS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class CCASpec:
    label: str
    n: int
    threshold: int
    seed: int
    burn_in: int = 0
    steps_per_frame: int = 2


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


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int) -> None:
    font = _font(max(15 * SS, width // 64))
    margin = 10 * SS
    tw = int(draw.textlength(text, font=font))
    x = max(margin, min(x, width - tw - margin))
    draw.text((x + SS, y + SS), text, font=font, fill=(0, 0, 0, 190))
    draw.text((x, y), text, font=font, fill=(224, 236, 235, 232))


def _hsv_palette(n: int) -> np.ndarray:
    h = np.arange(n, dtype=np.float32) / max(1, n)
    s = np.full(n, 0.86, dtype=np.float32)
    v = 0.82 + 0.13 * np.sin(2.0 * np.pi * h + 0.55) ** 2

    i = np.floor(h * 6.0).astype(np.int16)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    rgb = np.zeros((n, 3), dtype=np.float32)
    m = i % 6
    rgb[m == 0] = np.stack([v, t, p], axis=1)[m == 0]
    rgb[m == 1] = np.stack([q, v, p], axis=1)[m == 1]
    rgb[m == 2] = np.stack([p, v, t], axis=1)[m == 2]
    rgb[m == 3] = np.stack([p, q, v], axis=1)[m == 3]
    rgb[m == 4] = np.stack([t, p, v], axis=1)[m == 4]
    rgb[m == 5] = np.stack([v, p, q], axis=1)[m == 5]
    return (rgb * 255.0).astype(np.float32)


def _init_grid(spec: CCASpec, width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    return rng.integers(0, spec.n, size=(height, width), dtype=np.uint8)


def _step(grid: np.ndarray, n: int, threshold: int) -> np.ndarray:
    successor = (grid + 1) % n
    count = np.zeros(grid.shape, dtype=np.uint8)
    for dy, dx in NEIGHBOR_SHIFTS:
        count += np.roll(np.roll(grid, dy, axis=0), dx, axis=1) == successor
    return np.where(count >= threshold, successor, grid).astype(np.uint8, copy=False)


def _advance(grid: np.ndarray, n: int, threshold: int, steps: int) -> np.ndarray:
    for _ in range(max(0, steps)):
        grid = _step(grid, n, threshold)
    return grid


def _colorize(grid: np.ndarray, palette: np.ndarray, glow: bool = True) -> np.ndarray:
    frame = palette[grid].astype(np.float32)

    h, w = grid.shape
    yy, xx = np.ogrid[:h, :w]
    dx = (xx - w * 0.5) / max(1.0, w * 0.5)
    dy = (yy - h * 0.5) / max(1.0, h * 0.5)
    vignette = np.clip(1.0 - 0.42 * (dx * dx + dy * dy), 0.52, 1.0)
    frame = BG + frame * vignette[..., None]

    if glow:
        src = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")
        bloom = np.asarray(src.filter(ImageFilter.GaussianBlur(radius=max(2, 3 * SS))), dtype=np.float32)
        frame = frame * 0.86 + bloom * 0.18

    return np.clip(frame, 0, 255).astype(np.uint8)


def _label_frame(frame: np.ndarray, label: str, x: int = 18, y: int = 16) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    _draw_label(draw, label, x, y, frame.shape[1])
    return np.asarray(img, dtype=np.uint8)


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.LANCZOS))


def _write_video(output: str, fps: int, frames_iter) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for frame in frames_iter:
            writer.append_data(frame)
    finally:
        writer.close()


def _frames_for_spec(
    spec: CCASpec,
    frames: int,
    grid_w: int,
    grid_h: int,
    out_w: int,
    out_h: int,
    label: str,
):
    grid = _init_grid(spec, grid_w, grid_h)
    grid = _advance(grid, spec.n, spec.threshold, spec.burn_in)
    palette = _hsv_palette(spec.n)
    for _ in range(frames):
        frame = _colorize(grid, palette)
        frame = _resize_frame(frame, out_w, out_h)
        yield _label_frame(frame, label)
        grid = _advance(grid, spec.n, spec.threshold, spec.steps_per_frame)


def _render_evolve(output: str, frames: int, fps: int, width: int, height: int) -> None:
    spec = CCASpec("N=14  threshold=1", n=14, threshold=1, seed=1401, steps_per_frame=3)
    _write_video(output, fps, _frames_for_spec(spec, frames, GRID_W * SS, GRID_H * SS, width, height, "cyclic cellular automaton"))
    _print_done("evolve", output, spec, "grid=480x270 upscaled to 1280x720")


def _render_birth(output: str, frames: int, fps: int, width: int, height: int) -> None:
    spec = CCASpec("N=14  threshold=1", n=14, threshold=1, seed=1402, steps_per_frame=1)
    _write_video(output, fps, _frames_for_spec(spec, frames, GRID_W * SS, GRID_H * SS, width, height, "noise becomes wavefronts"))
    _print_done("birth", output, spec, "grid=480x270 upscaled to 1280x720")


def _render_still(output: str, frames: int, fps: int, width: int, height: int) -> None:
    spec = CCASpec("N=14  threshold=1", n=14, threshold=1, seed=1403, burn_in=720, steps_per_frame=0)
    _write_video(output, fps, _frames_for_spec(spec, frames, GRID_W * SS, GRID_H * SS, width, height, "fully developed spiral waves"))
    _print_done("still", output, spec, "burn_in=720, held frame")


def _render_gallery(output: str, frames: int, fps: int, width: int, height: int) -> None:
    specs = (
        CCASpec("N=8 chunky", n=8, threshold=1, seed=801, burn_in=360, steps_per_frame=1),
        CCASpec("N=14 fine", n=14, threshold=1, seed=1404, burn_in=480, steps_per_frame=2),
        CCASpec("N=20 wispy", n=20, threshold=1, seed=2001, burn_in=620, steps_per_frame=2),
    )
    panel_w = width // 3
    panel_widths = (panel_w, panel_w, width - panel_w * 2)
    grids = [_advance(_init_grid(s, GRID_W // 3 * SS, GRID_H * SS), s.n, s.threshold, s.burn_in) for s in specs]
    palettes = [_hsv_palette(s.n) for s in specs]

    def frame_iter():
        local_grids = list(grids)
        for _ in range(frames):
            panels = []
            for idx, spec in enumerate(specs):
                frame = _colorize(local_grids[idx], palettes[idx])
                frame = _resize_frame(frame, panel_widths[idx], height)
                frame = _label_frame(frame, spec.label, x=16, y=16)
                panels.append(frame)
                local_grids[idx] = _advance(local_grids[idx], spec.n, spec.threshold, spec.steps_per_frame)
            yield np.concatenate(panels, axis=1)

    _write_video(output, fps, frame_iter())
    _print_done("gallery", output, specs[1], "panels: N=8, N=14, N=20")


def _print_done(mode: str, output: str, spec: CCASpec, note: str) -> None:
    print(
        f"[OK] modes=evolve,still,birth,gallery mode={mode} output={output} "
        f"grid={GRID_W}x{GRID_H} N={spec.n} threshold={spec.threshold} {note}"
    )
    print("Cost note: CA steps are cheap and vectorized with np.roll over the 8 Moore-neighborhood shifts.")


def render(mode: str, output: str, seconds: float, fps: int = FPS, width: int = WIDTH, height: int = HEIGHT) -> None:
    frames = max(2, int(seconds * fps))
    if mode == "evolve":
        _render_evolve(output, frames, fps, width, height)
    elif mode == "still":
        _render_still(output, frames, fps, width, height)
    elif mode == "birth":
        _render_birth(output, frames, fps, width, height)
    elif mode == "gallery":
        _render_gallery(output, frames, fps, width, height)
    else:
        raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["evolve", "still", "birth", "gallery"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=18.0)
    args = ap.parse_args()
    render(args.mode, args.output, args.seconds)


if __name__ == "__main__":
    main()
