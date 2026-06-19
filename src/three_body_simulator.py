"""Three-body choreography simulator for Agentic Pixels.

Newtonian gravity with leapfrog integration, no damping or attractor terms.
The artwork comes from glowing recent trails: figure-8 choreography,
rotating equal-mass polygons, unstable dances drifting into chaos, and
restricted three-body Trojan islands around L4/L5.

Usage:
  python three_body_simulator.py --mode evolve --output /tmp/tb_e.mp4 --seconds 10
  python three_body_simulator.py --mode gallery --output /tmp/tb_g.mp4 --seconds 18
  python three_body_simulator.py --mode walkthrough --output /tmp/tb_w.mp4 --seconds 20
  python three_body_simulator.py --mode lagrange --output /tmp/tb_l.mp4 --seconds 12
  python three_body_simulator.py --mode still --output /tmp/tb_s.png
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 24
BG = np.array([4, 6, 13], dtype=np.float32)
DEEP = np.array([5, 14, 31], dtype=np.float32)
CYAN = np.array([36, 228, 232], dtype=np.float32)
TEAL = np.array([34, 176, 156], dtype=np.float32)
VIOLET = np.array([157, 92, 255], dtype=np.float32)
AMBER = np.array([255, 186, 92], dtype=np.float32)
WHITE = np.array([235, 250, 255], dtype=np.float32)
PALETTE = np.vstack([CYAN, TEAL, VIOLET, AMBER, WHITE]).astype(np.float32)


@dataclass(frozen=True)
class DanceSpec:
    label: str
    kind: str
    n: int = 3
    radius: float = 1.0
    perturb: float = 0.0
    dt: float = 0.004
    substeps: int = 9
    half_height: float = 1.65
    trail: int = 170


GALLERY = (
    DanceSpec("figure-8", "figure8", dt=0.0038, substeps=10, half_height=1.35, trail=210),
    DanceSpec("triangle", "polygon", n=3, radius=0.82, dt=0.0055, substeps=9, half_height=1.25, trail=180),
    DanceSpec("square", "polygon", n=4, radius=0.78, dt=0.0048, substeps=9, half_height=1.28, trail=180),
    DanceSpec("pentagon", "polygon", n=5, radius=0.77, dt=0.0043, substeps=9, half_height=1.30, trail=180),
    DanceSpec("Euler line", "euler", radius=0.86, dt=0.0042, substeps=9, half_height=1.35, trail=190),
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


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _background(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gx = xx / max(1, width - 1)
    gy = yy / max(1, height - 1)
    r = np.sqrt(((gx - 0.5) * 2.0) ** 2 + ((gy - 0.5) * 2.0) ** 2)
    vignette = 1.0 - 0.48 * _smoothstep((r - 0.25) / 0.95)
    frame = np.zeros((height, width, 3), dtype=np.float32)
    frame[:] = BG + DEEP * 0.68
    frame += CYAN * (0.018 * (1.0 - gy))[..., None]
    frame += VIOLET * (0.018 * gx)[..., None]
    frame += AMBER * (0.012 * (1.0 - gx) * gy)[..., None]
    frame *= vignette[..., None]
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))


def _draw_label(img: Image.Image, text: str, width: int, height: int, y: int = 16) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    font = _font(max(18, width // 58))
    draw.text((19, y + 1), text, font=font, fill=(0, 0, 0, 190))
    draw.text((18, y), text, font=font, fill=(226, 242, 248, 235))


def _write_video(output: str, fps: int, frames_iter) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for frame in frames_iter:
            writer.append_data(frame)
    finally:
        writer.close()


class GravitySystem:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, dt: float, soft: float = 0.006):
        self.pos = pos.astype(np.float32, copy=True)
        self.vel = vel.astype(np.float32, copy=True)
        self.mass = mass.astype(np.float32, copy=True)
        self.dt = np.float32(dt)
        self.soft = np.float32(soft)
        self.last_step_ms = 0.0

    def acceleration(self) -> np.ndarray:
        diff = self.pos[None, :, :] - self.pos[:, None, :]
        dist2 = np.sum(diff * diff, axis=2) + self.soft * self.soft
        inv_r3 = np.power(dist2, -1.5, dtype=np.float32)
        np.fill_diagonal(inv_r3, 0.0)
        return (diff * (self.mass[None, :, None] * inv_r3[:, :, None])).sum(axis=1).astype(np.float32)

    def step(self, substeps: int) -> None:
        started = time.perf_counter()
        for _ in range(substeps):
            a0 = self.acceleration()
            vh = self.vel + np.float32(0.5) * self.dt * a0
            self.pos = (self.pos + self.dt * vh).astype(np.float32, copy=False)
            a1 = self.acceleration()
            self.vel = (vh + np.float32(0.5) * self.dt * a1).astype(np.float32, copy=False)
        self.last_step_ms = (time.perf_counter() - started) * 1000.0 / max(1, substeps)


def _figure8(dt: float, perturb: float = 0.0) -> GravitySystem:
    pos = np.array(
        [[-0.97000436, 0.24308753], [0.97000436, -0.24308753], [0.0, 0.0]],
        dtype=np.float32,
    )
    vel = np.array(
        [[0.4662036850, 0.4323657300], [0.4662036850, 0.4323657300], [-0.93240737, -0.86473146]],
        dtype=np.float32,
    )
    if perturb:
        vel[0, 0] += np.float32(perturb)
        pos[2, 1] += np.float32(perturb * 0.5)
    return GravitySystem(pos, vel, np.ones(3, dtype=np.float32), dt=dt, soft=0.0035)


def _polygon(n: int, radius: float, dt: float, perturb: float = 0.0) -> GravitySystem:
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32) + np.float32(np.pi / 2)
    pos = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1).astype(np.float32)
    mass = np.ones(n, dtype=np.float32)
    probe = GravitySystem(pos, np.zeros_like(pos), mass, dt=dt, soft=0.0035)
    acc = probe.acceleration()
    radial = np.sum(acc * (-pos / radius), axis=1)
    speed = math.sqrt(max(0.0, float(np.mean(radial)) * radius))
    tangent = np.stack([-np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)
    vel = tangent * np.float32(speed)
    if perturb:
        vel[0] += np.array([perturb, -perturb * 0.55], dtype=np.float32)
        pos[-1] += np.array([perturb * 0.6, perturb * 0.3], dtype=np.float32)
    vel -= np.average(vel, axis=0, weights=mass).astype(np.float32)
    return GravitySystem(pos, vel, mass, dt=dt, soft=0.0035)


def _euler(radius: float, dt: float, perturb: float = 0.0) -> GravitySystem:
    a = np.float32(radius)
    pos = np.array([[-a, 0.0], [0.0, 0.0], [a, 0.0]], dtype=np.float32)
    omega = math.sqrt(1.25 / float(a**3))
    vel = np.array([[0.0, -omega * a], [0.0, 0.0], [0.0, omega * a]], dtype=np.float32)
    if perturb:
        pos[1, 1] += np.float32(perturb)
        vel[2, 0] += np.float32(perturb * 0.7)
    return GravitySystem(pos, vel, np.ones(3, dtype=np.float32), dt=dt, soft=0.0035)


def _make_dance(spec: DanceSpec) -> GravitySystem:
    if spec.kind == "figure8":
        return _figure8(spec.dt, spec.perturb)
    if spec.kind == "polygon":
        return _polygon(spec.n, spec.radius, spec.dt, spec.perturb)
    if spec.kind == "euler":
        return _euler(spec.radius, spec.dt, spec.perturb)
    raise ValueError(spec.kind)


def _to_pixels(points: np.ndarray, width: int, height: int, center: np.ndarray, half_height: float) -> np.ndarray:
    half_width = half_height * width / max(1, height)
    x = (points[:, 0] - center[0] + half_width) / (2.0 * half_width) * width
    y = (center[1] + half_height - points[:, 1]) / (2.0 * half_height) * height
    return np.stack([x, y], axis=1).astype(np.float32)


def _colors(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        c = PALETTE[i % len(PALETTE)] * (0.88 + 0.16 * math.sin(i * 1.37))
        out.append(tuple(np.clip(c, 0, 255).astype(np.uint8).tolist()))
    return out


def _draw_dance(
    sim: GravitySystem,
    history: deque[np.ndarray],
    width: int,
    height: int,
    label: str,
    half_height: float,
    center: np.ndarray | None = None,
    subtitle: str = "",
) -> np.ndarray:
    center = np.zeros(2, dtype=np.float32) if center is None else center.astype(np.float32)
    base = _background(width, height).convert("RGBA")
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    cd = ImageDraw.Draw(core, "RGBA")
    colors = _colors(len(sim.mass))
    hist = list(history)
    for body in range(len(sim.mass)):
        pts = []
        for h in hist:
            p = _to_pixels(h[body : body + 1], width, height, center, half_height)[0]
            if -120 <= p[0] <= width + 120 and -120 <= p[1] <= height + 120:
                pts.append((float(p[0]), float(p[1])))
        if len(pts) > 1:
            col = colors[body]
            gd.line(pts, fill=col + (70,), width=3, joint="curve")
            cd.line(pts[-40:], fill=col + (105,), width=2, joint="curve")
    px = _to_pixels(sim.pos, width, height, center, half_height)
    for i, (x, y) in enumerate(px):
        col = colors[i]
        gd.ellipse((x - 20, y - 20, x + 20, y + 20), fill=col + (44,))
        cd.ellipse((x - 5, y - 5, x + 5, y + 5), fill=col + (235,))
        cd.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(245, 252, 255, 245))
    bloom = glow.filter(ImageFilter.GaussianBlur(radius=7))
    img = Image.alpha_composite(base, bloom)
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, core).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    arr = np.maximum(arr, np.asarray(img.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32) * 0.19)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    if label:
        _draw_label(img, label if not subtitle else f"{label}  |  {subtitle}", width, height)
    return np.asarray(img, dtype=np.uint8)


def render_evolve(args: argparse.Namespace) -> None:
    spec = DanceSpec("figure-8 choreography", "figure8", dt=0.0038, substeps=args.steps_per_frame, half_height=1.35, trail=args.trail)
    sim = _make_dance(spec)
    frames = max(2, int(args.seconds * args.fps))
    history: deque[np.ndarray] = deque(maxlen=spec.trail)

    def frames_iter():
        for _ in range(frames):
            history.append(sim.pos.copy())
            yield _draw_dance(sim, history, args.width, args.height, spec.label, spec.half_height)
            sim.step(spec.substeps)

    _write_video(args.output, args.fps, frames_iter())
    print(f"[OK] mode=evolve output={args.output} frames={frames} figure8=closed-looking last_step={sim.last_step_ms:.3f}ms")
    print("Cost note: one leapfrog substep builds an NxN displacement matrix, so gravity is O(N^2); here N=3.")


def render_gallery(args: argparse.Namespace) -> None:
    frames = max(2, int(args.seconds * args.fps))
    per = max(1, frames // len(GALLERY))
    sims = [_make_dance(s) for s in GALLERY]
    histories = [deque(maxlen=s.trail) for s in GALLERY]

    def frames_iter():
        for i in range(frames):
            idx = min(len(GALLERY) - 1, i // per)
            spec = GALLERY[idx]
            sim = sims[idx]
            histories[idx].append(sim.pos.copy())
            yield _draw_dance(sim, histories[idx], args.width, args.height, spec.label, spec.half_height)
            sim.step(spec.substeps)

    _write_video(args.output, args.fps, frames_iter())
    print(f"[OK] mode=gallery output={args.output} variants={len(GALLERY)} closed_curves=figure8/triangle/square/pentagon/euler last_step={sims[-1].last_step_ms:.3f}ms")
    print("Cost note: each dance uses vectorized O(N^2) leapfrog gravity; gallery N ranges from 3 to 5.")


def _walk_caption(f: float) -> str:
    if f < 0.28:
        return "a perfect rotating square"
    if f < 0.52:
        return "a tiny nudge"
    if f < 0.76:
        return "it drifts off the dance"
    return "chaos braids the trails"


def render_walkthrough(args: argparse.Namespace) -> None:
    spec = DanceSpec("unstable square", "polygon", n=4, radius=0.80, perturb=0.008, dt=0.0048, substeps=args.steps_per_frame, half_height=1.62, trail=args.trail)
    sim = _make_dance(spec)
    frames = max(2, int(args.seconds * args.fps))
    history: deque[np.ndarray] = deque(maxlen=spec.trail)
    d0 = sim.pos[:, None, :] - sim.pos[None, :, :]
    initial_pairs = np.sort(np.sqrt(np.sum(d0 * d0, axis=2))[np.triu_indices(len(sim.mass), 1)])
    final_error = 0.0
    running_half = spec.half_height

    frozen = False

    def frames_iter():
        nonlocal final_error, running_half, frozen
        for i in range(frames):
            f = i / max(1, frames - 1)
            if not frozen:
                history.append(sim.pos.copy())
                d = sim.pos[:, None, :] - sim.pos[None, :, :]
                pairs = np.sort(np.sqrt(np.sum(d * d, axis=2))[np.triu_indices(len(sim.mass), 1)])
                final_error = float(np.linalg.norm(pairs - initial_pairs) / max(float(np.linalg.norm(initial_pairs)), 1e-6))
                allpts = np.concatenate(list(history) + [sim.pos], axis=0)
                ext = float(np.max(np.sqrt(np.sum(allpts * allpts, axis=1))))
                running_half = max(running_half, ext * 1.15)
                # once the chaos has spread enough, FREEZE: hold the braided tangle in frame
                if running_half >= 3.0:
                    frozen = True
            yield _draw_dance(sim, history, args.width, args.height, "walkthrough", running_half, subtitle=_walk_caption(f))
            if not frozen:
                sim.step(spec.substeps)

    _write_video(args.output, args.fps, frames_iter())
    broke = final_error > 0.06
    print(f"[OK] mode=walkthrough output={args.output} perturbed_square_broke={broke} pair_distance_error={final_error:.3f} last_step={sim.last_step_ms:.3f}ms")
    print("Cost note: no keyframing; the visible drift is the same O(N^2) Newtonian ODE with a tiny initial perturbation.")


class RestrictedThreeBody:
    def __init__(self, seed: int = 7):
        self.mu = np.float32(0.001)
        self.dt = np.float32(0.010)
        self.soft = np.float32(0.004)
        self.t = np.float32(0.0)
        self.last_step_ms = 0.0
        rng = np.random.default_rng(seed)
        mu = float(self.mu)
        rh = (mu / 3.0) ** (1.0 / 3.0)
        centers = {
            "L1": np.array([1.0 - mu - rh, 0.0], dtype=np.float32),
            "L2": np.array([1.0 - mu + rh, 0.0], dtype=np.float32),
            "L3": np.array([-1.0 - 5.0 * mu / 12.0, 0.0], dtype=np.float32),
            "L4": np.array([0.5 - mu, math.sqrt(3.0) / 2.0], dtype=np.float32),
            "L5": np.array([0.5 - mu, -math.sqrt(3.0) / 2.0], dtype=np.float32),
        }
        pts = []
        groups = []
        for gi, name in enumerate(("L1", "L2", "L3", "L4", "L5")):
            count = 18 if name in ("L4", "L5") else 12
            scale = 0.007 if name in ("L4", "L5") else 0.032
            pts.append(centers[name] + rng.normal(0.0, scale, (count, 2)).astype(np.float32))
            groups.extend([gi] * count)
        self.rot_pos = np.vstack(pts).astype(np.float32)
        self.groups = np.array(groups, dtype=np.int32)
        self.pos = self.rot_pos.copy()
        self.vel = self._omega_cross(self.rot_pos)
        unstable = self.groups < 3
        kick = rng.normal(0.0, 0.015, (int(np.count_nonzero(unstable)), 2)).astype(np.float32)
        kick[:, 0] += np.sign(self.rot_pos[unstable, 0]) * np.float32(0.012)
        self.vel[unstable] += kick

    def _primaries(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        mu = float(self.mu)
        rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=np.float32)
        p = np.array([[-mu, 0.0], [1.0 - mu, 0.0]], dtype=np.float32) @ rot.T
        m = np.array([1.0 - mu, mu], dtype=np.float32)
        return p, m

    @staticmethod
    def _omega_cross(p: np.ndarray) -> np.ndarray:
        return np.stack([-p[:, 1], p[:, 0]], axis=1).astype(np.float32)

    def _accel(self, pos: np.ndarray, t: float) -> np.ndarray:
        prim, mass = self._primaries(t)
        diff = prim[None, :, :] - pos[:, None, :]
        dist2 = np.sum(diff * diff, axis=2) + self.soft * self.soft
        return (diff * (mass[None, :, None] * np.power(dist2, -1.5, dtype=np.float32)[:, :, None])).sum(axis=1).astype(np.float32)

    def step(self, substeps: int) -> None:
        started = time.perf_counter()
        for _ in range(substeps):
            t0 = float(self.t)
            a0 = self._accel(self.pos, t0)
            vh = self.vel + np.float32(0.5) * self.dt * a0
            mid = self.pos + self.dt * vh
            t1 = t0 + float(self.dt)
            a1 = self._accel(mid, t1)
            self.pos = mid.astype(np.float32, copy=False)
            self.vel = (vh + np.float32(0.5) * self.dt * a1).astype(np.float32, copy=False)
            self.t = np.float32(t1)
        self.last_step_ms = (time.perf_counter() - started) * 1000.0 / max(1, substeps)

    def rotating_positions(self) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(-float(self.t)), math.sin(-float(self.t))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        prim, _ = self._primaries(float(self.t))
        return self.pos @ rot.T, prim @ rot.T


def _draw_lagrange(sim: RestrictedThreeBody, history: deque[np.ndarray], width: int, height: int) -> np.ndarray:
    base = _background(width, height).convert("RGBA")
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    cd = ImageDraw.Draw(core, "RGBA")
    half = 1.30
    center = np.array([0.18, 0.0], dtype=np.float32)
    group_cols = [AMBER, VIOLET, TEAL, CYAN, CYAN]
    for j in range(len(sim.groups)):
        pts = []
        for h in history:
            p = _to_pixels(h[j : j + 1], width, height, center, half)[0]
            if -80 <= p[0] <= width + 80 and -80 <= p[1] <= height + 80:
                pts.append((float(p[0]), float(p[1])))
        if len(pts) > 1:
            col = tuple(np.clip(group_cols[int(sim.groups[j])] * (0.85 if sim.groups[j] < 3 else 1.0), 0, 255).astype(np.uint8).tolist())
            gd.line(pts, fill=col + (34 if sim.groups[j] < 3 else 65,), width=1, joint="curve")
    rot_pos, prim = sim.rotating_positions()
    px = _to_pixels(rot_pos, width, height, center, half)
    for j, (x, y) in enumerate(px):
        col = tuple(np.clip(group_cols[int(sim.groups[j])], 0, 255).astype(np.uint8).tolist())
        cd.ellipse((x - 2, y - 2, x + 2, y + 2), fill=col + (210,))
    ppx = _to_pixels(prim, width, height, center, half)
    cd.ellipse((ppx[0, 0] - 10, ppx[0, 1] - 10, ppx[0, 0] + 10, ppx[0, 1] + 10), fill=(245, 250, 255, 240))
    cd.ellipse((ppx[1, 0] - 5, ppx[1, 1] - 5, ppx[1, 0] + 5, ppx[1, 1] + 5), fill=(255, 186, 92, 240))
    lag = np.array(
        [
            [1.0 - float(sim.mu) - (float(sim.mu) / 3.0) ** (1.0 / 3.0), 0.0],
            [1.0 - float(sim.mu) + (float(sim.mu) / 3.0) ** (1.0 / 3.0), 0.0],
            [-1.0 - 5.0 * float(sim.mu) / 12.0, 0.0],
            [0.5 - float(sim.mu), math.sqrt(3.0) / 2.0],
            [0.5 - float(sim.mu), -math.sqrt(3.0) / 2.0],
        ],
        dtype=np.float32,
    )
    lpx = _to_pixels(lag, width, height, center, half)
    font = _font(14)
    for i, (x, y) in enumerate(lpx):
        col = (72, 232, 238, 210) if i >= 3 else (255, 186, 92, 170)
        cd.ellipse((x - 4, y - 4, x + 4, y + 4), outline=col, width=1)
        cd.text((x + 6, y - 8), f"L{i+1}", font=font, fill=col)
    img = Image.alpha_composite(base, glow.filter(ImageFilter.GaussianBlur(radius=5)))
    img = Image.alpha_composite(img, glow)
    img = Image.alpha_composite(img, core).convert("RGB")
    _draw_label(img, "restricted three-body  |  L4/L5 Trojans stay, L1-L3 leak", width, height)
    return np.asarray(img, dtype=np.uint8)


def render_lagrange(args: argparse.Namespace) -> None:
    sim = RestrictedThreeBody(seed=args.seed)
    frames = max(2, int(args.seconds * args.fps))
    history: deque[np.ndarray] = deque(maxlen=args.trail)
    stable0 = sim.rot_pos[sim.groups >= 3].copy()
    unstable0 = sim.rot_pos[sim.groups < 3].copy()
    stable_d = unstable_d = 0.0

    def frames_iter():
        nonlocal stable_d, unstable_d
        for _ in range(frames):
            rot_pos, _ = sim.rotating_positions()
            history.append(rot_pos.copy())
            stable_d = float(np.median(np.linalg.norm(rot_pos[sim.groups >= 3] - stable0, axis=1)))
            unstable_d = float(np.median(np.linalg.norm(rot_pos[sim.groups < 3] - unstable0, axis=1)))
            yield _draw_lagrange(sim, history, args.width, args.height)
            sim.step(args.steps_per_frame)

    _write_video(args.output, args.fps, frames_iter())
    print(f"[OK] mode=lagrange output={args.output} L4L5_median_drift={stable_d:.3f} L1L2L3_median_drift={unstable_d:.3f} trojans_stay={stable_d < unstable_d} last_step={sim.last_step_ms:.3f}ms")
    print("Cost note: restricted mode is O(P) for P test particles because only the two massive primaries gravitate.")


def render_still(args: argparse.Namespace) -> None:
    spec = DanceSpec("figure-8 choreography", "figure8", dt=0.0038, substeps=args.steps_per_frame, half_height=1.35, trail=max(args.trail, 260))
    sim = _make_dance(spec)
    history: deque[np.ndarray] = deque(maxlen=spec.trail)
    for _ in range(spec.trail):
        history.append(sim.pos.copy())
        sim.step(spec.substeps)
    frame = _draw_dance(sim, history, args.width, args.height, "three-body figure-8 choreography", spec.half_height)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out)
    print(f"[OK] mode=still output={args.output} figure8=closed-looking last_step={sim.last_step_ms:.3f}ms")
    print("Cost note: still uses the same O(N^2) three-body leapfrog integration.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["evolve", "gallery", "walkthrough", "lagrange", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps-per-frame", type=int, default=9)
    ap.add_argument("--trail", type=int, default=180)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "evolve":
        render_evolve(args)
    elif args.mode == "gallery":
        render_gallery(args)
    elif args.mode == "walkthrough":
        render_walkthrough(args)
    elif args.mode == "lagrange":
        render_lagrange(args)
    elif args.mode == "still":
        render_still(args)


if __name__ == "__main__":
    main()
