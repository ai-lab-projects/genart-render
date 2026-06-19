"""Greenberg-Hastings excitable medium mp4 generator for Agentic Pixels.

Each cell is RESTING, EXCITED, or REFRACTORY. Resting cells ignite when at
least one excited Moore-neighbor is nearby; excited cells enter a refractory
tail; refractory cells recover back to rest. Random initial grids reliably
self-organize into rotating spiral waves like a Belousov-Zhabotinsky reaction.

Modes:
  evolve   - random noise into mature rotating spiral waves
  still    - one held hero frame after burn-in
  birth    - early phase from noise to the first rotating cores
  gallery  - three side-by-side K variants with different spiral spacing
  walkthrough - kamishibai rule walkthrough on a small readable grid

Usage:
  python greenberg_hastings_simulator.py --mode evolve --output outputs/ghm.mp4 --seconds 18
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BG = np.array([2, 5, 14], dtype=np.float32)
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
class GHMSpec:
    label: str
    k: int
    threshold: int
    seed: int
    burn_in: int = 0
    steps_per_frame: int = 2


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, height: int) -> None:
    font = _font(max(15 * SS, width // 64))
    margin = 10 * SS
    tw = int(draw.textlength(text, font=font))
    bbox = draw.textbbox((0, 0), text, font=font)
    th = int(bbox[3] - bbox[1])
    x = max(margin, min(x, width - tw - margin))
    y = max(margin, min(y, height - th - margin))
    draw.text((x + SS, y + SS), text, font=font, fill=(0, 0, 0, 200))
    draw.text((x, y), text, font=font, fill=(226, 240, 246, 238))


def _palette(k: int) -> np.ndarray:
    """Return RGB colors for states 0..K-1: rest, hot front, fading refractory tail."""
    if k < 3:
        raise ValueError("Greenberg-Hastings requires K >= 3")

    palette = np.zeros((k, 3), dtype=np.float32)
    palette[0] = BG
    palette[1] = (255, 250, 220)

    hot = np.array([255, 126, 50], dtype=np.float32)
    rose = np.array([218, 42, 112], dtype=np.float32)
    violet = np.array([83, 43, 158], dtype=np.float32)
    deep = np.array([8, 14, 46], dtype=np.float32)

    refractory_count = k - 2
    for state in range(2, k):
        t = (state - 2) / max(1, refractory_count - 1)
        if t < 0.34:
            u = t / 0.34
            color = hot * (1.0 - u) + rose * u
        elif t < 0.72:
            u = (t - 0.34) / 0.38
            color = rose * (1.0 - u) + violet * u
        else:
            u = (t - 0.72) / 0.28
            color = violet * (1.0 - u) + deep * u
        brightness = 0.95 - 0.38 * t
        palette[state] = color * brightness
    return palette


def _init_grid(spec: GHMSpec, width: int, height: int) -> np.ndarray:
    # 粗い domain 初期化(小グリッドを nearest 拡大): fine-random+T1 は過剰点火で乱流, 高Tは全滅。
    # 大きな coherent domain にすると front が立ち spiral へ自己組織化する(#034 検証)。
    rng = np.random.default_rng(spec.seed)
    fac = 22
    small = rng.integers(0, spec.k, size=(height // fac + 2, width // fac + 2), dtype=np.uint8)
    big = np.array(Image.fromarray(small).resize((width, height), Image.NEAREST))
    return big.astype(np.uint8)


def _step(grid: np.ndarray, k: int, threshold: int) -> np.ndarray:
    excited_neighbors = np.zeros(grid.shape, dtype=np.uint8)
    for dy, dx in NEIGHBOR_SHIFTS:
        excited_neighbors += np.roll(np.roll(grid, dy, axis=0), dx, axis=1) == 1

    resting = grid == 0
    ignite = resting & (excited_neighbors >= threshold)

    nxt = grid + 1
    nxt = np.where(grid == k - 1, 0, nxt)
    nxt = np.where(ignite, 1, nxt)
    nxt = np.where(resting & ~ignite, 0, nxt)
    return nxt.astype(np.uint8, copy=False)


def _advance(grid: np.ndarray, k: int, threshold: int, steps: int) -> np.ndarray:
    for _ in range(max(0, steps)):
        grid = _step(grid, k, threshold)
    return grid


def _colorize(grid: np.ndarray, palette: np.ndarray, glow: bool = True) -> np.ndarray:
    frame = palette[grid].astype(np.float32)
    excited = grid == 1
    if excited.any():
        frame[excited] = frame[excited] * 0.62 + np.array([255, 255, 245], dtype=np.float32) * 0.38

    h, w = grid.shape
    yy, xx = np.ogrid[:h, :w]
    dx = (xx - w * 0.5) / max(1.0, w * 0.5)
    dy = (yy - h * 0.5) / max(1.0, h * 0.5)
    vignette = np.clip(1.0 - 0.48 * (dx * dx + dy * dy), 0.50, 1.0)
    frame = BG + frame * vignette[..., None]

    if glow and excited.any():
        front = np.zeros_like(frame)
        front[excited] = (255, 210, 120)
        src = Image.fromarray(np.clip(front, 0, 255).astype(np.uint8), "RGB")
        bloom = np.asarray(src.filter(ImageFilter.GaussianBlur(radius=max(2, 3 * SS))), dtype=np.float32)
        frame = frame * 0.90 + bloom * 0.42

    return np.clip(frame, 0, 255).astype(np.uint8)


def _label_frame(frame: np.ndarray, label: str, x: int = 18, y: int = 16) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    _draw_label(draw, label, x, y, frame.shape[1], frame.shape[0])
    return np.asarray(img, dtype=np.uint8)


def _walkthrough_palette(k: int) -> np.ndarray:
    """Teaching palette: rest, fire, and one dimming recovery hue."""
    palette = np.zeros((k, 3), dtype=np.float32)
    palette[0] = (4, 8, 17)
    palette[1] = (255, 246, 168)
    recovery_hue = np.array([42, 184, 232], dtype=np.float32)
    for state in range(2, k):
        t = (state - 2) / max(1, k - 3)
        brightness = 0.90 - 0.58 * t
        palette[state] = recovery_hue * brightness
    return palette


def _init_walkthrough_grid(grid_n: int, k: int) -> np.ndarray:
    grid = np.zeros((grid_n, grid_n), dtype=np.uint8)
    mid_y = grid_n // 2
    start_x = 4
    tip_x = grid_n // 2

    grid[mid_y, start_x : tip_x + 1] = 1
    for offset, state in enumerate(range(2, min(k, 6)), start=1):
        y = mid_y + offset
        if y >= grid_n:
            break
        grid[y, start_x : tip_x + 1] = state
    return grid


def _draw_walkthrough_legend(draw: ImageDraw.ImageDraw, palette: np.ndarray, width: int, height: int) -> None:
    font = _font(max(11 * SS, width // 104))
    swatch = max(10 * SS, width // 96)
    gap = max(7 * SS, width // 150)
    pad = max(8 * SS, width // 128)
    labels = ("RESTING", "FIRING", "RECOVERING")
    colors = (palette[0], palette[1], palette[2] if len(palette) > 2 else palette[1])
    text_w = max(int(draw.textlength(label, font=font)) for label in labels)
    text_h = max(draw.textbbox((0, 0), label, font=font)[3] - draw.textbbox((0, 0), label, font=font)[1] for label in labels)
    row_h = max(swatch, int(text_h)) + gap
    box_w = pad * 2 + swatch + gap + text_w
    box_h = pad * 2 + row_h * len(labels) - gap
    margin = max(14 * SS, width // 80)
    x = min(max(margin, width - box_w - margin), width - box_w - 1)
    y = min(max(margin, height - box_h - margin), height - box_h - 1)

    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=4 * SS, fill=(1, 5, 13, 188), outline=(42, 55, 72, 210))
    for idx, (label, color) in enumerate(zip(labels, colors)):
        yy = y + pad + idx * row_h
        rgb = tuple(np.clip(color, 0, 255).astype(np.uint8).tolist())
        draw.rectangle((x + pad, yy, x + pad + swatch, yy + swatch), fill=rgb, outline=(78, 92, 112))
        draw.text((x + pad + swatch + gap, yy - 1), label, font=font, fill=(218, 229, 235, 238))


def _walkthrough_frame(grid: np.ndarray, palette: np.ndarray, width: int, height: int, label: str) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = BG.astype(np.uint8)

    grid_n = grid.shape[0]
    cell = max(12, min((width - 210) // grid_n, (height - 118) // grid_n))
    board_w = cell * grid_n
    board_h = cell * grid_n
    x0 = (width - board_w) // 2
    y0 = max(76, (height - board_h) // 2 + 24)

    cells = np.repeat(np.repeat(palette[grid], cell, axis=0), cell, axis=1)
    yy, xx = np.ogrid[:board_h, :board_w]
    dx = (xx - board_w * 0.5) / max(1.0, board_w * 0.5)
    dy = (yy - board_h * 0.5) / max(1.0, board_h * 0.5)
    vignette = np.clip(1.0 - 0.18 * (dx * dx + dy * dy), 0.74, 1.0)
    board = np.clip(cells * vignette[..., None], 0, 255).astype(np.uint8)

    line = np.array([30, 43, 62], dtype=np.uint8)
    for pos in range(0, board_w + 1, cell):
        x = min(pos, board_w - 1)
        board[:, x : min(x + 1, board_w)] = line
    for pos in range(0, board_h + 1, cell):
        y = min(pos, board_h - 1)
        board[y : min(y + 1, board_h), :] = line

    frame[y0 : y0 + board_h, x0 : x0 + board_w] = board
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    _draw_label(draw, label, 24, 18, width, height)
    _draw_walkthrough_legend(draw, palette, width, height)
    return np.asarray(img, dtype=np.uint8)


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.LANCZOS), dtype=np.uint8)


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
    spec: GHMSpec,
    frames: int,
    grid_w: int,
    grid_h: int,
    out_w: int,
    out_h: int,
    label: str,
):
    grid = _init_grid(spec, grid_w, grid_h)
    grid = _advance(grid, spec.k, spec.threshold, spec.burn_in)
    palette = _palette(spec.k)
    for _ in range(frames):
        frame = _colorize(grid, palette)
        frame = _resize_frame(frame, out_w, out_h)
        yield _label_frame(frame, label)
        grid = _advance(grid, spec.k, spec.threshold, spec.steps_per_frame)


def _render_evolve(output: str, frames: int, fps: int, width: int, height: int, k: int, threshold: int, seed: int) -> None:
    spec = GHMSpec(f"K={k} threshold={threshold}", k=k, threshold=threshold, seed=seed, steps_per_frame=2)
    _write_video(
        output,
        fps,
        _frames_for_spec(spec, frames, GRID_W * SS, GRID_H * SS, width, height, "excitable medium: spiral wave self-organization"),
    )
    _print_done("evolve", output, spec, "random init, no burn-in")


def _render_birth(output: str, frames: int, fps: int, width: int, height: int, k: int, threshold: int, seed: int) -> None:
    spec = GHMSpec(f"K={k} threshold={threshold}", k=k, threshold=threshold, seed=seed, steps_per_frame=1)
    _write_video(
        output,
        fps,
        _frames_for_spec(spec, frames, GRID_W * SS, GRID_H * SS, width, height, "noise ignites the first rotating cores"),
    )
    _print_done("birth", output, spec, "early phase, steps_per_frame=1")


def _render_still(output: str, frames: int, fps: int, width: int, height: int, k: int, threshold: int, seed: int) -> None:
    spec = GHMSpec(f"K={k} threshold={threshold}", k=k, threshold=threshold, seed=seed, burn_in=620, steps_per_frame=0)
    _write_video(
        output,
        fps,
        _frames_for_spec(spec, frames, GRID_W * SS, GRID_H * SS, width, height, "self-organizing spiral waves"),
    )
    _print_done("still", output, spec, "burn_in=620, held frame")


def _render_gallery(output: str, frames: int, fps: int, width: int, height: int, threshold: int, seed: int) -> None:
    specs = (
        GHMSpec("K=6 tight", k=6, threshold=threshold, seed=seed + 61, burn_in=460, steps_per_frame=1),
        GHMSpec("K=8 classic", k=8, threshold=threshold, seed=seed + 83, burn_in=620, steps_per_frame=1),
        GHMSpec("K=11 broad", k=11, threshold=threshold, seed=seed + 117, burn_in=780, steps_per_frame=1),
    )
    panel_w = width // 3
    panel_widths = (panel_w, panel_w, width - panel_w * 2)
    grid_w = GRID_W // 3 * SS
    grids = [_advance(_init_grid(s, grid_w, GRID_H * SS), s.k, s.threshold, s.burn_in) for s in specs]
    palettes = [_palette(s.k) for s in specs]

    def frame_iter():
        local_grids = list(grids)
        for _ in range(frames):
            panels = []
            for idx, spec in enumerate(specs):
                frame = _colorize(local_grids[idx], palettes[idx])
                frame = _resize_frame(frame, panel_widths[idx], height)
                frame = _label_frame(frame, spec.label, x=16, y=16)
                panels.append(frame)
                local_grids[idx] = _advance(local_grids[idx], spec.k, spec.threshold, spec.steps_per_frame)
            yield np.concatenate(panels, axis=1)

    _write_video(output, fps, frame_iter())
    _print_done("gallery", output, specs[1], "panels: K=6, K=8, K=11")


def _render_walkthrough(output: str, frames: int, fps: int, width: int, height: int, k: int, threshold: int) -> None:
    grid_n = 31
    min_steps = 16 if frames >= fps * 16 else 1
    step_count = max(min_steps, min(20, int(round(frames / max(1, int(round(1.35 * fps)))))))
    hold_frames = max(1, frames // step_count)
    remainder = max(0, frames - hold_frames * step_count)
    grid = _init_walkthrough_grid(grid_n, k)
    palette = _walkthrough_palette(k)
    spec = GHMSpec("walkthrough", k=k, threshold=threshold, seed=0, steps_per_frame=1)

    def frame_iter():
        nonlocal grid
        for step in range(step_count):
            repeat = hold_frames + (1 if step < remainder else 0)
            frame = _walkthrough_frame(grid, palette, width, height, f"Step {step}")
            for _ in range(repeat):
                yield frame
            grid = _step(grid, k, threshold)

    _write_video(output, fps, frame_iter())
    _print_done(
        "walkthrough",
        output,
        spec,
        f"rewritten: broken wavefront seed, 3-category countdown colors, legend; {grid_n}x{grid_n} grid, {step_count} held steps",
    )


def _print_done(mode: str, output: str, spec: GHMSpec, note: str) -> None:
    print(
        f"[OK] modes=evolve,still,birth,gallery,walkthrough mode={mode} output={output} "
        f"grid={GRID_W}x{GRID_H} K={spec.k} threshold={spec.threshold} seed={spec.seed} {note}"
    )
    print("Cost note: synchronous CA updates are vectorized with np.roll over 8 Moore-neighbor shifts per step.")


def render(
    mode: str,
    output: str,
    seconds: float,
    fps: int = FPS,
    width: int = WIDTH,
    height: int = HEIGHT,
    k: int = 8,
    threshold: int = 1,
    seed: int = 8808,
) -> None:
    if not 3 <= k <= 255:
        raise ValueError("--k must be in the range 3..255")
    if not 1 <= threshold <= 8:
        raise ValueError("--threshold must be in the range 1..8")

    frames = max(2, int(seconds * fps))
    if mode == "evolve":
        _render_evolve(output, frames, fps, width, height, k, threshold, seed)
    elif mode == "still":
        _render_still(output, frames, fps, width, height, k, threshold, seed)
    elif mode == "birth":
        _render_birth(output, frames, fps, width, height, k, threshold, seed)
    elif mode == "gallery":
        _render_gallery(output, frames, fps, width, height, threshold, seed)
    elif mode == "walkthrough":
        _render_walkthrough(output, frames, fps, width, height, k, threshold)
    else:
        raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["evolve", "still", "birth", "gallery", "walkthrough"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--threshold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=8808)
    args = ap.parse_args()
    render(args.mode, args.output, args.seconds, k=args.k, threshold=args.threshold, seed=args.seed)


if __name__ == "__main__":
    main()
