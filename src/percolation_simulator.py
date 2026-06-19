"""Site percolation simulator for procedural-art videos.

Square-grid site percolation opens cells in one fixed random permutation
(`numpy.default_rng(seed=7)`). Open cells are joined with 4-neighbor union-find
using path compression and union by size. TOP/BOTTOM virtual nodes make spanning
detection O(alpha(N)): the tipping point is the first frame where they share a
root.

Cost model:
  sweep: each cell opens once, each neighbor union is attempted once, so the
         simulation update is O(N alpha(N)) total. Per frame recoloring uses
         vectorized root lookup over the open cells and image upscale.
  threshold/still: build the needed state once per panel, no scipy required.

Usage:
  python percolation_simulator.py --mode sweep --output out.mp4 --duration 18
  python percolation_simulator.py --mode threshold --output out.mp4
  python percolation_simulator.py --mode still --output out.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops


GW = 200
GH = 112
PC = 0.5927
SEED = 7
BG = np.array([4, 6, 13], dtype=np.uint8)
CLOSED = np.array([9, 12, 22], dtype=np.uint8)
GOLD = np.array([255, 209, 76], dtype=np.uint8)
FLASH = np.array([255, 246, 176], dtype=np.uint8)
TEXT = (222, 235, 246)
MUTED = (122, 142, 166)
BAR_BG = (28, 34, 48)
BAR_FG = (255, 209, 76)

JEWELS = np.array(
    [
        [40, 184, 219],
        [80, 218, 162],
        [110, 140, 255],
        [190, 105, 255],
        [255, 92, 145],
        [255, 133, 72],
        [238, 218, 93],
        [55, 207, 118],
        [78, 168, 255],
        [210, 90, 220],
        [245, 115, 115],
        [137, 232, 215],
    ],
    dtype=np.uint8,
)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find_one(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def find_many(self, xs: np.ndarray) -> np.ndarray:
        roots = xs.astype(np.int32, copy=True)
        parent = self.parent
        while True:
            nxt = parent[roots]
            changed = nxt != roots
            if not np.any(changed):
                return roots
            roots[changed] = nxt[changed]

    def union(self, a: int, b: int) -> int:
        ra = self.find_one(a)
        rb = self.find_one(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


class PercolationState:
    def __init__(self, gw: int = GW, gh: int = GH, seed: int = SEED) -> None:
        self.gw = gw
        self.gh = gh
        self.n = gw * gh
        self.top = self.n
        self.bottom = self.n + 1
        self.uf = UnionFind(self.n + 2)
        self.open_mask = np.zeros(self.n, dtype=bool)
        self.order = np.random.default_rng(seed).permutation(self.n).astype(np.int32)
        self.open_count = 0
        self.first_span_count: int | None = None

    @property
    def p(self) -> float:
        return self.open_count / self.n

    @property
    def spans(self) -> bool:
        return self.uf.find_one(self.top) == self.uf.find_one(self.bottom)

    def open_to(self, count: int) -> None:
        count = int(np.clip(count, 0, self.n))
        while self.open_count < count:
            idx = int(self.order[self.open_count])
            self.open_mask[idx] = True
            y, x = divmod(idx, self.gw)
            if y == 0:
                self.uf.union(idx, self.top)
            if y == self.gh - 1:
                self.uf.union(idx, self.bottom)
            if x > 0 and self.open_mask[idx - 1]:
                self.uf.union(idx, idx - 1)
            if x + 1 < self.gw and self.open_mask[idx + 1]:
                self.uf.union(idx, idx + 1)
            if y > 0 and self.open_mask[idx - self.gw]:
                self.uf.union(idx, idx - self.gw)
            if y + 1 < self.gh and self.open_mask[idx + self.gw]:
                self.uf.union(idx, idx + self.gw)
            self.open_count += 1
            if self.first_span_count is None and self.spans:
                self.first_span_count = self.open_count


def state_at_fraction(p: float, gw: int, gh: int, seed: int) -> PercolationState:
    state = PercolationState(gw, gh, seed)
    state.open_to(round(p * state.n))
    return state


def first_spanning_state(gw: int, gh: int, seed: int) -> PercolationState:
    state = PercolationState(gw, gh, seed)
    state.open_to(state.n)
    count = state.first_span_count or round(PC * state.n)
    state = PercolationState(gw, gh, seed)
    state.open_to(count)
    return state


def cluster_rgb(state: PercolationState) -> np.ndarray:
    rgb = np.empty((state.n, 3), dtype=np.uint8)
    rgb[:] = CLOSED
    open_ids = np.flatnonzero(state.open_mask).astype(np.int32)
    if open_ids.size == 0:
        return rgb.reshape(state.gh, state.gw, 3)
    roots = state.uf.find_many(open_ids)
    colors = JEWELS[(roots.astype(np.int64) * 2654435761 % len(JEWELS))]
    rgb[open_ids] = colors
    if state.spans:
        span_root = state.uf.find_one(state.top)
        rgb[open_ids[roots == span_root]] = GOLD
    return rgb.reshape(state.gh, state.gw, 3)


def fit_grid(grid_rgb: np.ndarray, width: int, height: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    img = Image.fromarray(grid_rgb, "RGB")
    scale = min(width / img.width, height / img.height)
    gw = max(1, int(round(img.width * scale)))
    gh = max(1, int(round(img.height * scale)))
    resized = img.resize((gw, gh), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (width, height), tuple(int(v) for v in BG))
    ox = (width - gw) // 2
    oy = (height - gh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, (ox, oy, gw, gh)


def font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], size: int = 16) -> None:
    x, y = xy
    fnt = font(size)
    draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=fnt)
    draw.text((x, y), text, fill=TEXT, font=fnt)


def draw_progress(draw: ImageDraw.ImageDraw, p: float, width: int) -> None:
    x0, y0 = 20, 46
    x1 = min(width - 20, x0 + 210)
    y1 = y0 + 5
    draw.rounded_rectangle((x0, y0, x1, y1), radius=2, fill=BAR_BG)
    draw.rounded_rectangle((x0, y0, x0 + int((x1 - x0) * min(p / 0.75, 1.0)), y1), radius=2, fill=BAR_FG)


def render_state_frame(state: PercolationState, width: int, height: int, label: str = "", flash: float = 0.0) -> np.ndarray:
    grid = cluster_rgb(state)
    if flash > 0:
        grid = np.clip(grid.astype(np.float32) * (1.0 - flash) + FLASH * flash, 0, 255).astype(np.uint8)
    canvas, _ = fit_grid(grid, width, height)
    # bloom/glow: 格子ブロックを発光させて attractor 風に映えさせる (#029 user: アート底上げ)
    bloom = canvas.filter(ImageFilter.GaussianBlur(max(2, width // 200)))
    canvas = ImageChops.add(canvas, bloom, scale=1.5)
    draw = ImageDraw.Draw(canvas)
    caption = label or f"p = {state.p:.3f}"
    # ラベルは下端へ (上は section_title overlay 用に空ける, F24-2)
    fs = max(14, width // 80)
    draw_label(draw, caption, (20, height - fs - 22), size=fs)
    draw_progress(draw, state.p, width)
    return np.asarray(canvas)


def compose_threshold_frame(states: list[PercolationState], labels: list[str], width: int, height: int) -> np.ndarray:
    canvas = Image.new("RGB", (width, height), tuple(int(v) for v in BG))
    draw = ImageDraw.Draw(canvas)
    margin = max(18, width // 60)
    gap = max(10, width // 120)
    panel_w = (width - margin * 2 - gap * 2) // 3
    panel_h = int(panel_w * GH / GW)
    max_panel_h = height - margin * 2 - 50
    if panel_h > max_panel_h:
        panel_h = max_panel_h
        panel_w = int(panel_h * GW / GH)
    y = (height - panel_h) // 2 + 18
    for i, (state, label) in enumerate(zip(states, labels, strict=True)):
        x = margin + i * (panel_w + gap)
        img = Image.fromarray(cluster_rgb(state), "RGB").resize((panel_w, panel_h), Image.Resampling.NEAREST)
        canvas.paste(img, (x, y))
        draw_label(draw, label, (x, max(12, y - 34)), size=max(13, width // 95))
        draw.text((x, y + panel_h + 8), f"p = {state.p:.3f}", fill=MUTED, font=font(max(12, width // 110)))
    return np.asarray(canvas)


def render_sweep(output: Path, duration: float, fps: int, width: int, height: int, gw: int, gh: int, seed: int) -> None:
    frames = max(2, int(round(duration * fps)))
    target = round(0.75 * gw * gh)
    counts = np.linspace(0, target, frames).round().astype(np.int32)
    state = PercolationState(gw, gh, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for count in counts:
            before = state.first_span_count
            state.open_to(int(count))
            flash = 0.0
            if before is None and state.first_span_count is not None:
                flash = 0.55
            label = f"site percolation  p = {state.p:.3f}"
            writer.append_data(render_state_frame(state, width, height, label=label, flash=flash))
    finally:
        writer.close()


def render_threshold(output: Path, duration: float, fps: int, width: int, height: int, gw: int, gh: int, seed: int) -> None:
    below = state_at_fraction(0.50, gw, gh, seed)
    critical = first_spanning_state(gw, gh, seed)
    above = state_at_fraction(0.68, gw, gh, seed)
    labels = ["below: isolated islands", "critical: it just barely spans", "above: one connected whole"]
    frame = compose_threshold_frame([below, critical, above], labels, width, height)
    write_static(output, frame, duration, fps)


def render_still(output: Path, duration: float, fps: int, width: int, height: int, gw: int, gh: int, seed: int) -> None:
    state = first_spanning_state(gw, gh, seed)
    frame = render_state_frame(state, width, height, label=f"critical cluster  p = {state.p:.3f}")
    write_static(output, frame, duration, fps)


def write_static(output: Path, frame: np.ndarray, duration: float, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(round(duration * fps)))
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for _ in range(frames):
            writer.append_data(frame)
    finally:
        writer.close()


def render(mode: str, output: Path, duration: float, fps: int, width: int, height: int, gw: int, gh: int, seed: int) -> None:
    if mode == "sweep":
        render_sweep(output, duration, fps, width, height, gw, gh, seed)
    elif mode == "threshold":
        render_threshold(output, duration, fps, width, height, gw, gh, seed)
    elif mode == "still":
        render_still(output, duration, fps, width, height, gw, gh, seed)
    else:
        raise ValueError(f"unknown mode: {mode}")
    print(f"[OK] {mode} -> {output} ({width}x{height}, {duration:.1f}s @ {fps}fps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sweep", choices=["sweep", "threshold", "still"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=18.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gw", type=int, default=GW)
    parser.add_argument("--gh", type=int, default=GH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    render(args.mode, args.output, args.duration, args.fps, args.width, args.height, args.gw, args.gh, args.seed)


if __name__ == "__main__":
    main()
