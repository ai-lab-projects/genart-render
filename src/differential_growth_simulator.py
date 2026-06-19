"""Differential-growth closed-curve simulator for Agentic Pixels.

A closed chain grows faster than the available plane can comfortably hold it.
Neighbor springs keep the curve cohesive, grid-hashed repulsion prevents
self-overlap, and midpoint insertion keeps adding material until the loop
buckles into brain/coral/kale-like folds.

Usage:
  python differential_growth_simulator.py --mode grow --output /tmp/dg_grow.mp4 --seconds 10
  python differential_growth_simulator.py --mode walkthrough --output /tmp/dg_wt.mp4 --seconds 24
  python differential_growth_simulator.py --mode gallery --output /tmp/dg_gal.mp4 --seconds 14
  python differential_growth_simulator.py --mode still --output /tmp/dg_still.png
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 24
SS = 1

BG = np.array([4, 6, 13], dtype=np.float32)
DEEP = np.array([5, 14, 31], dtype=np.float32)
CYAN = np.array([36, 228, 232], dtype=np.float32)
TEAL = np.array([34, 176, 156], dtype=np.float32)
VIOLET = np.array([157, 92, 255], dtype=np.float32)
AMBER = np.array([255, 186, 92], dtype=np.float32)
WHITE_HOT = np.array([234, 250, 255], dtype=np.float32)


@dataclass(frozen=True)
class DGSpec:
    label: str = "differential growth"
    d: float = 0.015
    repulsion: float = 0.0030
    attraction: float = 0.20
    center_pull: float = 0.0006   # ほぼ0: 中心へ引かない(放射状の星型を防ぐ)。soft boundary で在圏維持し自由に蛇行・空間充填
    growth_interval: int = 6
    max_nodes: int = 4500
    seed: int = 7721
    radius: float = 0.112
    initial_nodes: int = 120


VARIANTS = (
    DGSpec("tight folds", d=0.0120, repulsion=0.0030, attraction=0.20, center_pull=0.0006, growth_interval=5, seed=1011),
    DGSpec("loose waves", d=0.0190, repulsion=0.0024, attraction=0.17, center_pull=0.0005, growth_interval=8, seed=2017),
    DGSpec("dense brain", d=0.0100, repulsion=0.0032, attraction=0.22, center_pull=0.0008, growth_interval=4, seed=3019),
    DGSpec("loose folds", d=0.0165, repulsion=0.0027, attraction=0.18, center_pull=0.0006, growth_interval=6, seed=4013),
)


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


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, height: int | None = None) -> None:
    font = _font(max(15 * SS, width // 64))
    margin = 12 * SS
    tw = int(draw.textlength(text, font=font))
    bbox = draw.textbbox((0, 0), text, font=font)
    th = int(bbox[3] - bbox[1])
    x = max(margin, min(x, width - tw - margin))
    if height is not None:
        y = max(margin, min(y, height - th - margin))
    draw.text((x + SS, y + SS), text, font=font, fill=(0, 0, 0, 190))
    draw.text((x, y), text, font=font, fill=(224, 239, 245, 232))


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _vignette(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xx / max(1, width - 1) - 0.5) * 2.0
    y = (yy / max(1, height - 1) - 0.5) * 2.0
    r = np.sqrt(x * x + y * y)
    return (1.0 - 0.42 * _smoothstep(np.clip((r - 0.34) / 0.86, 0.0, 1.0))).astype(np.float32)


def _background(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gx = xx / max(1, width - 1)
    gy = yy / max(1, height - 1)
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[:] = BG + DEEP * 0.78
    frame += CYAN * (0.018 * (1.0 - gy))[..., None]
    frame += VIOLET * (0.015 * gx)[..., None]
    frame *= _vignette(width, height)[..., None]
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))


def _write_video(output: str, fps: int, frames_iter) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for frame in frames_iter:
            writer.append_data(frame)
    finally:
        writer.close()


class DifferentialGrowth:
    def __init__(self, spec: DGSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        theta = np.linspace(0.0, 2.0 * np.pi, spec.initial_nodes, endpoint=False, dtype=np.float32)
        wobble = 1.0 + 0.020 * np.sin(theta * 5.0 + 0.7) + 0.014 * self.rng.normal(size=spec.initial_nodes)
        self.points = np.stack(
            [0.5 + np.cos(theta) * spec.radius * wobble, 0.5 + np.sin(theta) * spec.radius * wobble],
            axis=1,
        ).astype(np.float32)
        self.step_count = 0
        self.last_step_ms = 0.0

    def step(self, substeps: int = 1) -> None:
        for _ in range(substeps):
            started = time.perf_counter()
            p = self.points
            n = len(p)
            prev = np.roll(p, 1, axis=0)
            nxt = np.roll(p, -1, axis=0)
            forces = self._spring_force(p, prev) + self._spring_force(p, nxt)
            forces += ((prev + nxt) * np.float32(0.5) - p) * np.float32(0.16)  # Laplacian 平滑: 高周波ジャギを抑え大きく滑らかな折りに
            forces += self._repulsion_force(p)
            forces += (np.array([0.5, 0.5], dtype=np.float32) - p) * np.float32(self.spec.center_pull)
            forces += self._soft_boundary_force(p)

            disp = np.clip(forces, -0.018, 0.018)
            self.points = np.clip(p + disp.astype(np.float32), 0.035, 0.965)
            self.step_count += 1
            if self.step_count % self.spec.growth_interval == 0 and n < self.spec.max_nodes:
                self._grow()
            self.last_step_ms = (time.perf_counter() - started) * 1000.0

    def _spring_force(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        diff = q - p
        dist = np.sqrt(np.sum(diff * diff, axis=1, keepdims=True)) + np.float32(1e-6)
        return diff / dist * ((dist - np.float32(self.spec.d)) * np.float32(self.spec.attraction))

    def _repulsion_force(self, p: np.ndarray) -> np.ndarray:
        n = len(p)
        radius = np.float32(self.spec.d * 2.6)
        r2 = float(radius * radius)
        inv_r = np.float32(1.0) / radius
        cell = float(radius)
        coords = np.floor(p / cell).astype(np.int32)
        grid: dict[tuple[int, int], list[int]] = {}
        for idx, (cx, cy) in enumerate(coords):
            grid.setdefault((int(cx), int(cy)), []).append(idx)

        force = np.zeros_like(p, dtype=np.float32)
        for i, (cx, cy) in enumerate(coords):
            pi = p[i]
            for gx in (int(cx) - 1, int(cx), int(cx) + 1):
                for gy in (int(cy) - 1, int(cy), int(cy) + 1):
                    for j in grid.get((gx, gy), ()):
                        if j == i or abs(j - i) <= 2 or abs(j - i) >= n - 2:
                            continue
                        diff = pi - p[j]
                        dsq = float(diff[0] * diff[0] + diff[1] * diff[1])
                        if 1e-10 < dsq < r2:
                            dist = math.sqrt(dsq)
                            strength = (1.0 - dist * float(inv_r)) ** 2
                            force[i] += diff * np.float32(self.spec.repulsion * strength / dist)
        return force

    def _soft_boundary_force(self, p: np.ndarray) -> np.ndarray:
        f = np.zeros_like(p, dtype=np.float32)
        margin = np.float32(0.075)
        low = p < margin
        high = p > (1.0 - margin)
        f += low * ((margin - p) * np.float32(0.018))
        f -= high * ((p - (1.0 - margin)) * np.float32(0.018))
        return f

    def _curvature(self) -> np.ndarray:
        prev = np.roll(self.points, 1, axis=0)
        nxt = np.roll(self.points, -1, axis=0)
        return np.sqrt(np.sum((prev - 2.0 * self.points + nxt) ** 2, axis=1))

    def _grow(self) -> None:
        p = self.points
        n = len(p)
        if n >= self.spec.max_nodes:
            return
        nxt = np.roll(p, -1, axis=0)
        edge = np.sqrt(np.sum((nxt - p) ** 2, axis=1))
        curvature = self._curvature()
        threshold = self.spec.d * 0.72
        split = edge > threshold
        if n > 420:
            hot = curvature > np.quantile(curvature, 0.45)
            split &= hot | (edge > self.spec.d * 0.95)
        if not split.any():
            split[int(self.rng.integers(0, n))] = True

        room = min(self.spec.max_nodes - n, max(2, int(n * 0.045)))
        idxs = np.flatnonzero(split)
        if len(idxs) > room:
            idxs = idxs[np.argsort(edge[idxs] + curvature[idxs] * 2.0)[-room:]]
            split[:] = False
            split[idxs] = True

        new_points = []
        for i in range(n):
            new_points.append(p[i])
            if split[i]:
                q = p[(i + 1) % n]
                mid = (p[i] + q) * 0.5
                tangent = q - p[i]
                normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
                norm = float(np.linalg.norm(normal)) + 1e-6
                normal /= norm
                jitter = normal * np.float32(self.rng.normal(0.0, self.spec.d * 0.020))
                new_points.append((mid + jitter).astype(np.float32))
        self.points = np.asarray(new_points, dtype=np.float32)


def _world_to_pixels(points: np.ndarray, width: int, height: int, zoom: float = 1.0, center: tuple[float, float] = (0.5, 0.5)) -> list[tuple[float, float]]:
    scale = min(width, height) * zoom
    cx, cy = center
    x = width * 0.5 + (points[:, 0] - cx) * scale
    y = height * 0.5 + (points[:, 1] - cy) * scale
    return list(zip(x.astype(float), y.astype(float)))


def _curvature_colors(points: np.ndarray) -> list[tuple[int, int, int]]:
    curv = np.sqrt(np.sum((np.roll(points, 1, axis=0) - 2.0 * points + np.roll(points, -1, axis=0)) ** 2, axis=1))
    hot = np.clip((curv - np.percentile(curv, 40)) / max(np.percentile(curv, 96) - np.percentile(curv, 40), 1e-6), 0.0, 1.0)
    phase = np.linspace(0.0, 1.0, len(points), endpoint=False, dtype=np.float32)
    colors = []
    for h, ph in zip(hot, phase):
        base = CYAN * (0.70 + 0.18 * math.sin(float(ph) * math.tau)) + TEAL * 0.22
        color = base * (1.0 - h) + (VIOLET * 0.72 + AMBER * 0.36) * h
        colors.append(tuple(np.clip(color, 0, 255).astype(np.uint8).tolist()))
    return colors


def render_curve_frame(
    points: np.ndarray,
    width: int,
    height: int,
    label: str = "",
    nodes: bool = False,
    zoom: float = 1.0,
    center: tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    base = _background(width, height).convert("RGBA")
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    cd = ImageDraw.Draw(core, "RGBA")
    pts = _world_to_pixels(points, width, height, zoom, center)
    closed = pts + pts[:1]
    colors = _curvature_colors(points)

    for w, a in ((18, 18), (8, 46), (4, 92)):
        gd.line(closed, fill=(39, 226, 232, a), width=max(1, w * SS), joint="curve")
    for i in range(len(pts)):
        cd.line((pts[i], pts[(i + 1) % len(pts)]), fill=(*colors[i], 218), width=max(1, 2 * SS), joint="curve")
    if nodes:
        for i, (x, y) in enumerate(pts):
            r = 6 * SS if len(pts) <= 22 else 3 * SS
            fill = (236, 250, 255, 235) if i % 3 == 0 else (43, 226, 232, 220)
            cd.ellipse((x - r, y - r, x + r, y + r), fill=fill)

    bloom = glow.filter(ImageFilter.GaussianBlur(radius=7 * SS))
    img = Image.alpha_composite(base, bloom)
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, core).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    arr = np.maximum(arr, np.asarray(img.filter(ImageFilter.GaussianBlur(radius=2 * SS)), dtype=np.float32) * 0.26)
    frame = np.clip(arr, 0, 255).astype(np.uint8)
    if label:
        img = Image.fromarray(frame).convert("RGB")
        _draw_label(ImageDraw.Draw(img), label, 18, 16, width, height)
        frame = np.asarray(img, dtype=np.uint8)
    return frame


def _burn_to_nodes(sim: DifferentialGrowth, min_nodes: int, max_steps: int = 2200, substeps: int = 2) -> None:
    steps = 0
    while len(sim.points) < min_nodes and steps < max_steps:
        sim.step(substeps)
        steps += substeps


def render_grow(args: argparse.Namespace) -> None:
    frames = max(2, int(args.seconds * args.fps))
    spec = DGSpec(seed=args.seed, max_nodes=args.max_nodes)
    sim = DifferentialGrowth(spec)
    steps_per_frame = max(1, args.steps_per_frame)

    def frame_iter():
        for i in range(frames):
            yield render_curve_frame(sim.points, args.width, args.height, label="")
            sim.step(steps_per_frame)

    _write_video(args.output, args.fps, frame_iter())
    print(
        f"[OK] mode=grow output={args.output} frames={frames} nodes={len(sim.points)} "
        f"steps_per_frame={steps_per_frame} last_step={sim.last_step_ms:.2f}ms"
    )
    print("Cost note: repulsion uses a cell grid with fixed-radius neighbor buckets, expected O(N) per simulation step.")


def render_gallery(args: argparse.Namespace) -> None:
    frames = max(len(VARIANTS), int(args.seconds * args.fps))
    sims = [DifferentialGrowth(DGSpec(**{**v.__dict__, "max_nodes": min(args.max_nodes, 1150)})) for v in VARIANTS]
    for sim in sims:
        _burn_to_nodes(sim, min_nodes=720, max_steps=420, substeps=2)

    def frame_iter():
        for i in range(frames):
            vi = min(len(sims) - 1, int(i * len(sims) / frames))
            sim = sims[vi]
            yield render_curve_frame(sim.points, args.width, args.height, label=sim.spec.label)
            if i % 2 == 0:
                sim.step(max(1, args.steps_per_frame))

    _write_video(args.output, args.fps, frame_iter())
    node_counts = ",".join(str(len(s.points)) for s in sims)
    print(f"[OK] mode=gallery output={args.output} variants={len(sims)} nodes={node_counts} last_step={sims[-1].last_step_ms:.2f}ms")
    print("Cost note: each variant uses grid-hashed local repulsion; no all-pairs pass.")


def render_still(args: argparse.Namespace) -> None:
    spec = DGSpec(seed=args.seed, max_nodes=args.max_nodes, growth_interval=4, d=0.0115, repulsion=0.0030, center_pull=0.0043)
    sim = DifferentialGrowth(spec)
    _burn_to_nodes(sim, min_nodes=min(args.max_nodes, 3200), max_steps=1700, substeps=2)
    frame = render_curve_frame(sim.points, args.width, args.height, label="differential growth: final buckled fold field")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out)
    print(f"[OK] mode=still output={args.output} nodes={len(sim.points)} last_step={sim.last_step_ms:.2f}ms")
    print("Cost note: burn-in uses O(N) grid repulsion per step; PNG render is a single held frame.")


def _walkthrough_initial() -> DifferentialGrowth:
    spec = DGSpec(
        "walkthrough",
        d=0.058,
        repulsion=0.0065,
        attraction=0.24,
        center_pull=0.0005,
        growth_interval=1000,
        max_nodes=80,
        seed=5151,
        radius=0.120,
        initial_nodes=14,
    )
    sim = DifferentialGrowth(spec)
    sim.points[:, 0] = 0.5 + (sim.points[:, 0] - 0.5) * 1.55
    sim.points[:, 1] = 0.5 + (sim.points[:, 1] - 0.5) * 0.78
    sim.points[2:5] += np.array([-0.020, 0.030], dtype=np.float32)
    sim.points[8:11] += np.array([0.018, -0.026], dtype=np.float32)
    return sim


def _walkthrough_stage_points(sim: DifferentialGrowth, stage: int) -> None:
    if stage == 0:
        for _ in range(8):
            p = sim.points
            forces = sim._spring_force(p, np.roll(p, 1, axis=0)) + sim._spring_force(p, np.roll(p, -1, axis=0))
            sim.points = np.clip(p + forces * 1.35, 0.05, 0.95)
    elif stage == 1:
        for _ in range(10):
            p = sim.points
            sim.points = np.clip(p + sim._repulsion_force(p) * 2.8, 0.05, 0.95)
    elif stage == 2:
        p = sim.points
        nxt = np.roll(p, -1, axis=0)
        edge = np.sqrt(np.sum((nxt - p) ** 2, axis=1))
        split = edge > np.quantile(edge, 0.56)
        new_points = []
        for i in range(len(p)):
            new_points.append(p[i])
            if split[i]:
                mid = (p[i] + p[(i + 1) % len(p)]) * 0.5
                new_points.append(mid.astype(np.float32))
        sim.points = np.asarray(new_points, dtype=np.float32)
    else:
        sim.step(5)


def _wt_lerp(a, b, t):
    t = float(_smoothstep(t))
    return [(ax + (bx - ax) * t, ay + (by - ay) * t) for (ax, ay), (bx, by) in zip(a, b)]


def _wt_draw_chain(d, pts, hl=(), newidx=None, new_alpha=1.0, closed=False):
    seq = [(float(x), float(y)) for x, y in pts]
    line = seq + seq[:1] if (closed and len(seq) > 2) else seq
    for w, a in ((17, 22), (8, 58), (3, 150)):
        d.line(line, fill=(43, 226, 232, a), width=w, joint="curve")
    for i, (x, y) in enumerate(seq):
        if i == newidx:
            col = (120, 252, 154); r = 13; al = int(255 * new_alpha)
        elif i in hl:
            col = (236, 250, 255); r = 13; al = 255
        else:
            col = (120, 205, 220); r = 9; al = 255
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*col, al))


def _wt_arrow(d, x1, y1, x2, y2, col=(255, 150, 90)):
    d.line((x1, y1, x2, y2), fill=(*col, 240), width=5)
    ang = math.atan2(y2 - y1, x2 - x1); ln = 16
    for s in (0.5, -0.5):
        d.line((x2, y2, x2 - ln * math.cos(ang - s), y2 - ln * math.sin(ang - s)), fill=(*col, 240), width=5)


def render_walkthrough(args: argparse.Namespace) -> None:
    """Didactic kamishibai matching the REAL free-loop dynamics (#036 v5).
    NO box, NO pinned ends, NO confinement (v3 pins / v4 box were both fabrications that didn't
    match the sim — user rightly flagged that the real sim is a free, unconfined loop). The honest
    mechanism the CODE actually uses: springs keep even spacing, and repulsion reaches a bit FARTHER
    than one step (REP > 2*d), so points two-apart push each other -> a straight/smooth run is
    UNSTABLE and kinks. Growth keeps adding length, so an ever-longer line that must stay evenly
    spaced AND keep away from itself has no smooth shape left -> it buckles and folds (self-repulsion
    buckling, no container). Behavior emerges from relaxation; inserts timed to narration.
    (Note: nature's brain folds via growth+confinement — a *related theme* but a different cause;
    this walkthrough explains THIS sim's cause, self-repulsion, honestly.)"""
    W, H, fps = args.width, args.height, args.fps
    frames = max(2, int(args.seconds * fps))
    bg = _background(W, H).convert("RGBA")

    def compose(layer):
        img = Image.alpha_composite(bg.copy(), layer.filter(ImageFilter.GaussianBlur(6 * SS)))
        img = Image.alpha_composite(img, layer).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    CX, CY = 645.0, 380.0
    d = 32.0
    REP = 74.0          # > 2*d  ->  points two steps apart repel  ->  straight is unstable (the real cause)
    rng = np.random.default_rng(7)
    N0 = 16
    th = np.linspace(0, 2 * np.pi, N0, endpoint=False)
    pts = np.stack([CX + 80.0 * np.cos(th), CY + 80.0 * np.sin(th)], axis=1).astype(float)

    def relax(steps=3):
        nonlocal pts
        for _ in range(steps):
            n = len(pts)
            prev = np.roll(pts, 1, 0); nxt = np.roll(pts, -1, 0)
            F = np.zeros_like(pts)
            for q in (prev, nxt):
                dv = q - pts; dist = np.hypot(dv[:, 0], dv[:, 1])[:, None] + 1e-6
                F += dv / dist * (dist - d) * 0.5
            F += ((prev + nxt) * 0.5 - pts) * 0.06            # mild smoothing
            for i in range(n):                                 # self-repulsion (non-adjacent), REP > 2d
                for j in range(n):
                    if min((i - j) % n, (j - i) % n) > 1:
                        dv = pts[i] - pts[j]; dd = math.hypot(dv[0], dv[1]) + 1e-6
                        if dd < REP:
                            F[i] += dv / dd * (REP - dd) * 0.05
            cen = pts.mean(0)                                  # weak cohesion: the blob holds together
            F += (cen - pts) * 0.010                           # (NOT a box; like the hero's faint pull) -> repulsion+growth must fold, not fly apart
            F += rng.normal(0, 0.4, pts.shape)
            pts = pts + np.clip(F, -4, 4)
            pts = pts - (pts.mean(0) - np.array([CX, CY]))     # recenter (display only)
            pts[:, 0] = np.clip(pts[:, 0], 70, 1210); pts[:, 1] = np.clip(pts[:, 1], 96, 640)

    def insert():
        nonlocal pts
        nxt = np.roll(pts, -1, 0); seg = nxt - pts
        k = int(np.argmax(np.hypot(seg[:, 0], seg[:, 1])))
        a = pts[k]; b = pts[(k + 1) % len(pts)]
        mid = (a + b) / 2.0
        nx, ny = -(b[1] - a[1]), (b[0] - a[0]); nn = math.hypot(nx, ny) + 1e-6
        mid = mid + np.array([nx, ny]) / nn * 4.0
        pts = np.insert(pts, k + 1, mid, axis=0)

    # "now grow it" is the 5th sentence (~0.50 of seg); inserts start then.
    insert_frames = set(int(f * frames) for f in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90))

    def caption(fr):
        f = fr / frames
        if f < 0.20: return "a ring · springs keep even spacing"
        if f < 0.48: return "each bead pushes nearby ones — reaching past its neighbor  →  straight can't survive"
        if f < 0.80: return "grow it  →  ever more line, all repelling itself  →  it folds"
        return "it keeps away from itself as it grows  →  winds into folds, never crossing"

    def two_apart_closest():
        n = len(pts); best = None; bd = 1e9
        for i in range(n):
            j = (i + 2) % n
            dd = math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
            if dd < bd: bd = dd; best = (i, j)
        return best, bd

    def closest_nonadjacent():
        n = len(pts); best = None; bd = 1e9
        for i in range(n):
            for j in range(n):
                if min((i - j) % n, (j - i) % n) > 1 and i < j:
                    dd = math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                    if dd < bd: bd = dd; best = (i, j)
        return best, bd

    def frame_iter():
        for fr in range(frames):
            if fr in insert_frames and len(pts) < 42:
                insert()
            relax(3)
            f = fr / frames
            hl = set(); arrows = []
            n = len(pts)
            if 0.20 <= f < 0.48 and n > 4:                     # show the 2-apart push that kinks a straight run
                pair, bd = two_apart_closest()
                if pair:
                    i, j = pair; mid_i = (i + 1) % n; hl = {i, j, mid_i}
                    pi, pj = pts[i], pts[j]
                    dx, dy = pi - pj; dn = math.hypot(dx, dy) + 1e-6
                    ux, uy = dx / dn, dy / dn
                    arrows = [(pi[0], pi[1], pi[0] + ux * 30, pi[1] + uy * 30),
                              (pj[0], pj[1], pj[0] - ux * 30, pj[1] - uy * 30)]
            elif 0.80 <= f and n > 4:                          # repulsion keeps folds from crossing
                pair, bd = closest_nonadjacent()
                if pair and bd < REP * 1.2:
                    i, j = pair; hl = {i, j}
                    pi, pj = pts[i], pts[j]
                    dx, dy = pi - pj; dn = math.hypot(dx, dy) + 1e-6
                    ux, uy = dx / dn, dy / dn
                    mx, my = (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2
                    arrows = [(mx, my, mx + ux * 26, my + uy * 26), (mx, my, mx - ux * 26, my - uy * 26)]
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dr = ImageDraw.Draw(layer, "RGBA")
            _wt_draw_chain(dr, pts.tolist(), hl=hl, closed=True)
            for (x1, y1, x2, y2) in arrows:
                _wt_arrow(dr, x1, y1, x2, y2)
            frame = compose(layer)
            img = Image.fromarray(frame)
            _draw_label(ImageDraw.Draw(img), caption(fr), 18, 16, W, H)
            yield np.asarray(img, dtype=np.uint8)

    _write_video(args.output, args.fps, frame_iter())
    print(f"[OK] mode=walkthrough (free loop, self-repulsion buckling, no box) output={args.output} final_nodes={len(pts)}")


def render(mode: str, args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_nodes < 120:
        raise ValueError("--max-nodes must be at least 120")

    if mode == "grow":
        render_grow(args)
    elif mode == "walkthrough":
        render_walkthrough(args)
    elif mode == "gallery":
        render_gallery(args)
    elif mode == "still":
        render_still(args)
    else:
        raise ValueError(f"unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["grow", "walkthrough", "gallery", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--seed", type=int, default=7721)
    ap.add_argument("--max-nodes", type=int, default=4500)
    ap.add_argument("--steps-per-frame", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    render(args.mode, args)


if __name__ == "__main__":
    main()
