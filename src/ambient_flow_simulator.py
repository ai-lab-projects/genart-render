"""Seamless ambient curl-flow particle loop generator.

Thousands of points drift through a smooth, time-periodic 2D flow field and
leave soft glowing trails on a dark background. The loop mode is designed for
long-form relaxing background videos: frame N is mathematically the same state
as frame 0 because particle positions, trail history, palette, and field all
depend on time through a unit circle.

Usage:
  python ambient_flow_simulator.py --mode loop --output outputs/ambient_flow.mp4 --seconds 20
  python ambient_flow_simulator.py --mode still --output outputs/ambient_flow.png
  python ambient_flow_simulator.py --mode field --output outputs/ambient_flow_field.mp4 --seconds 18
  python ambient_flow_simulator.py --mode gallery --output outputs/ambient_flow_gallery.mp4 --seconds 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 24
SS = 1
BG = np.array([2, 5, 13], dtype=np.float32)
DEEP = np.array([4, 12, 28], dtype=np.float32)
BLUE = np.array([38, 116, 255], dtype=np.float32)
TEAL = np.array([32, 226, 220], dtype=np.float32)
VIOLET = np.array([160, 80, 255], dtype=np.float32)
WARM = np.array([255, 184, 86], dtype=np.float32)
WHITE_HOT = np.array([232, 248, 255], dtype=np.float32)


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
    margin = 12 * SS
    tw = int(draw.textlength(text, font=font))
    x = max(margin, min(x, width - tw - margin))
    draw.text((x + SS, y + SS), text, font=font, fill=(0, 0, 0, 180))
    draw.text((x, y), text, font=font, fill=(221, 237, 242, 222))


def _smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ease_pulse(x: np.ndarray) -> np.ndarray:
    """Periodic birth/death envelope: soft fade in, long hold, soft fade out."""
    rise = _smoothstep(np.clip(x / 0.16, 0.0, 1.0))
    fall = 1.0 - _smoothstep(np.clip((x - 0.84) / 0.16, 0.0, 1.0))
    return (rise * fall).astype(np.float32, copy=False)


def _make_particles(count: int, seed: int = 8107) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "x": rng.random(count, dtype=np.float32),
        "y": rng.random(count, dtype=np.float32),
        "phase": rng.random(count, dtype=np.float32),
        "scale": rng.uniform(0.74, 1.24, count).astype(np.float32),
        "lane": rng.integers(0, 4, count, dtype=np.int16),
    }


FIELD_VARIANTS = {
    "default": {
        "terms": (
            (1.0, 1.0, 0.030, 0.0, 0.9),
            (2.0, -1.0, 0.018, 1.7, -1.2),
            (-1.0, 2.0, 0.020, 3.1, 1.5),
            (3.0, 1.0, 0.010, 4.4, -0.7),
            (1.0, -3.0, 0.008, 2.2, 1.9),
        ),
        "swirl": 0.020,
        "swirl_radius": 0.18,
        "scale": 1.0,
    },
    "tight eddies": {
        "terms": (
            (3.0, 2.0, 0.018, 0.2, 1.2),
            (-4.0, 3.0, 0.016, 1.6, -1.0),
            (5.0, -2.0, 0.012, 2.5, 1.8),
            (-2.0, -5.0, 0.011, 4.1, -1.4),
            (6.0, 1.0, 0.008, 5.0, 0.8),
        ),
        "swirl": 0.044,
        "swirl_radius": 0.13,
        "scale": 1.24,
    },
    "broad streams": {
        "terms": (
            (1.0, 0.0, 0.030, 0.1, 0.4),
            (0.0, 1.0, 0.024, 1.5, -0.5),
            (1.0, -1.0, 0.014, 2.8, 0.7),
        ),
        "swirl": 0.008,
        "swirl_radius": 0.32,
        "scale": 1.55,
    },
    "many vortices": {
        "terms": (
            (2.0, 3.0, 0.018, 0.0, 1.1),
            (-3.0, 4.0, 0.015, 1.1, -1.5),
            (5.0, -4.0, 0.012, 2.2, 1.7),
            (-5.0, -3.0, 0.012, 3.0, -0.9),
            (7.0, 2.0, 0.007, 4.8, 1.2),
            (-2.0, 7.0, 0.007, 5.6, -1.3),
        ),
        "swirl": 0.026,
        "swirl_radius": 0.19,
        "scale": 1.18,
    },
    "silk lanes": {
        "terms": (
            (1.0, -1.0, 0.026, 0.5, 0.6),
            (2.0, 1.0, 0.014, 2.0, -0.7),
            (-1.0, 2.0, 0.012, 3.4, 0.8),
            (3.0, -1.0, 0.006, 4.6, -0.6),
        ),
        "swirl": 0.012,
        "swirl_radius": 0.26,
        "scale": 1.35,
    },
    # multi-scale woven field: big lanes + mid eddies + fine ripples -> rich, never-flat motion
    "woven silk": {
        "terms": (
            (1.0, -1.0, 0.028, 0.5, 0.6),
            (1.0, 1.0, 0.022, 1.4, -0.5),
            (2.0, 1.0, 0.016, 2.0, -0.8),
            (-1.0, 2.0, 0.014, 3.4, 0.9),
            (3.0, -2.0, 0.010, 4.6, -0.7),
            (-3.0, 1.0, 0.009, 0.9, 1.3),
            (4.0, 3.0, 0.006, 2.7, -1.1),
            (-2.0, 5.0, 0.005, 5.1, 1.0),
            (6.0, -1.0, 0.0035, 3.8, -0.6),
        ),
        "swirl": 0.020,
        "swirl_radius": 0.30,
        "scale": 1.30,
    },
}


def _curl_displacement(
    x: np.ndarray,
    y: np.ndarray,
    u: float,
    variant: str = "default",
) -> tuple[np.ndarray, np.ndarray]:
    """Periodic curl-like warp sampled on a time circle.

    This is a compact analytic flow field, not random-frame noise. All temporal
    phases are functions of cos/sin(2*pi*u), so u=1 and u=0 match exactly.
    """
    tau = np.float32(2.0 * np.pi)
    c = np.float32(np.cos(tau * u))
    s = np.float32(np.sin(tau * u))
    dx = np.zeros_like(x, dtype=np.float32)
    dy = np.zeros_like(y, dtype=np.float32)
    preset = FIELD_VARIANTS.get(variant, FIELD_VARIANTS["default"])
    for kx, ky, amp, ph, wob in preset["terms"]:
        p = tau * (kx * x + ky * y) + np.float32(ph) + np.float32(wob) * c + np.float32(0.55 * wob) * s
        q = np.cos(p).astype(np.float32, copy=False)
        dx += np.float32(amp * ky) * q
        dy -= np.float32(amp * kx) * q

    cx = x - np.float32(0.5)
    cy = y - np.float32(0.5)
    r2 = cx * cx + cy * cy
    swirl = np.float32(preset["swirl"]) * np.exp(-r2 / np.float32(preset["swirl_radius"]))
    dx += -cy * swirl * (np.float32(0.55) + np.float32(0.45) * c)
    dy += cx * swirl * (np.float32(0.55) + np.float32(0.45) * s)
    scale = np.float32(preset["scale"])
    return dx * scale, dy * scale


def _particle_positions(
    particles: dict[str, np.ndarray],
    u: float,
    variant: str = "default",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0 = particles["x"]
    y0 = particles["y"]
    phase = particles["phase"]
    scale = particles["scale"]
    lane = particles["lane"]

    dx, dy = _curl_displacement(x0, y0, u, variant)
    tau = np.float32(2.0 * np.pi)
    drift = (u + phase) % 1.0
    ribbon = tau * (drift + lane.astype(np.float32) * np.float32(0.173))
    slow_x = np.float32(0.020) * np.cos(ribbon).astype(np.float32, copy=False)
    slow_x += np.float32(0.010) * np.sin(ribbon * 2.0).astype(np.float32, copy=False)
    slow_y = np.float32(0.018) * np.sin(ribbon).astype(np.float32, copy=False)
    slow_y += np.float32(0.008) * np.cos(ribbon * 1.5).astype(np.float32, copy=False)

    x = (x0 + dx * scale + slow_x) % np.float32(1.0)
    y = (y0 + dy * scale + slow_y) % np.float32(1.0)
    age = (u + phase * np.float32(0.73)) % np.float32(1.0)
    alpha = _ease_pulse(age)
    return x, y, alpha, lane


def _palette_for_time(u: float) -> np.ndarray:
    tau = 2.0 * np.pi
    c = 0.5 + 0.5 * np.cos(tau * u)
    s = 0.5 + 0.5 * np.sin(tau * u)
    blue = BLUE * (0.72 + 0.18 * c) + TEAL * (0.10 + 0.10 * s)
    teal = TEAL * (0.78 + 0.12 * s) + WARM * (0.08 + 0.06 * c)
    violet = VIOLET * (0.72 + 0.16 * (1.0 - c)) + BLUE * 0.18
    warm = WARM * (0.62 + 0.18 * c) + TEAL * 0.12
    return np.vstack([blue, teal, violet, warm]).astype(np.float32)


def _deposit_trails(
    particles: dict[str, np.ndarray],
    u: float,
    width: int,
    height: int,
    trail_steps: int,
    variant: str = "default",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist = np.zeros((height, width), dtype=np.float32)
    channels = np.zeros((4, height, width), dtype=np.float32)
    total = max(1, trail_steps - 1)
    for j in range(trail_steps):
        tail = j / total
        sample_u = (u - 0.020 * tail) % 1.0
        x, y, alpha, lane = _particle_positions(particles, sample_u, variant)
        fade = np.float32((1.0 - tail) ** 1.7)
        weights = alpha * fade
        px = x * width
        py = y * height
        h, _, _ = np.histogram2d(py, px, bins=(height, width), range=((0, height), (0, width)), weights=weights)
        hist += h.astype(np.float32, copy=False)
        for k in range(4):
            mask = lane == k
            if mask.any():
                hk, _, _ = np.histogram2d(
                    py[mask],
                    px[mask],
                    bins=(height, width),
                    range=((0, height), (0, width)),
                    weights=weights[mask],
                )
                channels[k] += hk.astype(np.float32, copy=False)
    return hist, channels[0], channels[1], channels[2], channels[3]


def _colorize(hist: np.ndarray, channels: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], u: float) -> np.ndarray:
    if float(hist.max()) <= 0.0:
        frame = np.empty((*hist.shape, 3), dtype=np.uint8)
        frame[:] = BG.astype(np.uint8)
        return frame

    logv = np.log1p(hist)
    brightness = logv / max(float(logv.max()), 1e-6)
    brightness = np.clip(brightness ** 0.38, 0.0, 1.0).astype(np.float32)

    src = Image.fromarray((brightness * 255).astype(np.uint8), "L")
    glow_wide = np.asarray(src.filter(ImageFilter.GaussianBlur(radius=14 * SS)), dtype=np.float32) / 255.0
    glow_mid = np.asarray(src.filter(ImageFilter.GaussianBlur(radius=5 * SS)), dtype=np.float32) / 255.0

    frame = np.empty((*hist.shape, 3), dtype=np.float32)
    frame[:] = BG
    frame += DEEP * (0.62 + 0.12 * np.sin(2.0 * np.pi * u))
    frame += np.array([2.0, 5.0, 9.0], dtype=np.float32)
    frame += TEAL * (0.34 * glow_wide[..., None])
    frame += VIOLET * (0.24 * glow_wide[..., None])
    frame += WARM * (0.20 * glow_mid[..., None])

    pal = _palette_for_time(u)
    csum = np.maximum(sum(channels), 1e-6)
    color = np.zeros_like(frame)
    for i, ch in enumerate(channels):
        color += pal[i] * (ch / csum)[..., None]
    frame = frame * (1.0 - 0.98 * brightness[..., None]) + color * (1.10 * brightness[..., None])

    core = np.clip((brightness - 0.56) / 0.44, 0.0, 1.0) ** 1.45
    frame = frame * (1.0 - 0.62 * core[..., None]) + WHITE_HOT * (0.74 * core[..., None])
    vignette = _vignette(hist.shape[1], hist.shape[0])
    frame *= vignette[..., None]
    return np.clip(frame, 0, 255).astype(np.uint8)


def _vignette(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (xx / max(1, width - 1) - 0.5) * 2.0
    y = (yy / max(1, height - 1) - 0.5) * 2.0
    r = np.sqrt(x * x + y * y)
    return (1.0 - 0.20 * _smoothstep(np.clip((r - 0.45) / 0.95, 0.0, 1.0))).astype(np.float32)


def render_frame(
    particles: dict[str, np.ndarray],
    u: float,
    width: int = WIDTH,
    height: int = HEIGHT,
    trail_steps: int = 16,
    label: str = "",
    variant: str = "default",
) -> np.ndarray:
    hist, c0, c1, c2, c3 = _deposit_trails(particles, u % 1.0, width * SS, height * SS, trail_steps, variant)
    frame = _colorize(hist, (c0, c1, c2, c3), u % 1.0)
    if SS != 1:
        frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.LANCZOS), dtype=np.uint8)
    if label:
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        _draw_label(draw, label, 18, 16, width)
        frame = np.asarray(img, dtype=np.uint8)
    return frame


def _write_video(output: str, fps: int, frames_iter) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for frame in frames_iter:
            writer.append_data(frame)
    finally:
        writer.close()


def render_loop(args: argparse.Namespace) -> None:
    frames = max(2, int(args.seconds * args.fps))
    particles = _make_particles(args.particles, args.seed)

    def frame_iter():
        for i in range(frames):
            yield render_frame(particles, i / frames, args.width, args.height, args.trail_steps)

    _write_video(args.output, args.fps, frame_iter())
    print(
        f"[OK] seamless loop via time-circle field/particles: u=i/{frames}, u=1==0; "
        f"cost ~= {args.trail_steps} x {args.particles:,} vectorized deposits/frame -> {Path(args.output)}"
    )


def _field_background(width: int, height: int) -> Image.Image:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gx = xx / max(1, width - 1)
    gy = yy / max(1, height - 1)
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[:] = BG + DEEP * 0.82 + np.array([1.5, 5.0, 8.0], dtype=np.float32)
    frame += TEAL * (0.018 * (1.0 - gy))[..., None]
    frame += VIOLET * (0.014 * gx)[..., None]
    frame *= _vignette(width, height)[..., None]
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8), "RGB")


def _make_field_grid(width: int, height: int, cols: int = 12, rows: int = 8) -> dict[str, np.ndarray]:
    xmin, xmax = 0.18, 0.82
    ymin, ymax = 0.18, 0.78
    xs = np.linspace(xmin, xmax, cols, dtype=np.float32)
    ys = np.linspace(ymin, ymax, rows, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = _curl_displacement(gx, gy, 0.18, "silk lanes")
    mag = np.sqrt(dx * dx + dy * dy) + np.float32(1e-6)
    return {
        "x": (gx * width).reshape(-1),
        "y": (gy * height).reshape(-1),
        "dx": (dx / mag).reshape(-1),
        "dy": (dy / mag).reshape(-1),
    }


def _make_dense_field(size: int = 180, variant: str = "silk lanes") -> dict[str, np.ndarray]:
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, size, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = _curl_displacement(gx, gy, 0.18, variant)
    mag = np.sqrt(dx * dx + dy * dy) + np.float32(1e-6)
    return {"dx": dx / mag, "dy": dy / mag, "size": np.array(size, dtype=np.int32)}


# ---- flowing full-screen streamlines (the silk hero; deposit/loop fog is replaced by this) ----
_STREAM_COLORS = [(42, 232, 222), (166, 105, 255), (255, 196, 108), (120, 200, 255)]


def _draw_stream_comet(draw, path, head, win, width, height, color, env, ss):
    """Draw a moving comet window along an integrated streamline: faint tail -> bright head.
    env (0..1) is a birth/death envelope so heads fade in/out -> seamless loop (no teleport pop).
    ss = supersample factor; line widths scale with ss so they stay THIN after downscale."""
    if env <= 0.01:
        return
    lo = max(0, head - win)
    seg = path[lo:head + 1]
    if len(seg) < 2:
        return
    pts = [(float(x * width), float(y * height)) for x, y in seg]
    m = len(pts)
    head_lo = max(0, m - max(2, win // 3))  # brightest near the head
    # base widths are deliberately delicate; *ss keeps them ~1px after the LANCZOS downscale
    layers = ((3.4, 16), (1.8, 48), (0.9, 130))
    for gw, ga in layers:                    # whole faint trail
        draw.line(pts, fill=(*color, int(ga * env)), width=max(1, int(round(gw * ss))), joint="curve")
    for gw, ga in layers:                    # brighter near the head
        draw.line(pts[head_lo:], fill=(*color, int(ga * 1.7 * env)), width=max(1, int(round(gw * ss))), joint="curve")
    hx, hy = pts[-1]
    r = 2.6 * ss
    draw.ellipse((hx - r, hy - r, hx + r, hy + r), fill=(236, 250, 255, int(235 * env)))


def _render_stream_frame(paths, phase, u, width, height, win, bg_rgba, ss=2):
    """Render at width*ss (anti-alias), bloom, then LANCZOS downscale -> smooth thin silk."""
    W2, H2 = width * ss, height * ss
    img = bg_rgba.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    steps = paths.shape[1] - 1
    for pi in range(len(paths)):
        h = (u + float(phase[pi])) % 1.0
        env = float(_ease_pulse(np.array([h], dtype=np.float32))[0])
        head = int(h * steps)
        _draw_stream_comet(draw, paths[pi], head, win, W2, H2, _STREAM_COLORS[pi % 4], env, ss)
    rgb = img.convert("RGB")
    bloom = rgb.filter(ImageFilter.GaussianBlur(radius=5 * ss))
    arr = np.maximum(np.asarray(rgb, np.float32), 0.58 * np.asarray(bloom, np.float32))
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if ss != 1:
        out = out.resize((width, height), Image.LANCZOS)
    return np.asarray(out, np.uint8)


def render_streams(args: argparse.Namespace) -> None:
    frames = max(2, int(args.seconds * args.fps))
    variant = getattr(args, "variant", None) or "woven silk"
    ss = 2
    dense = _make_dense_field(size=240, variant=variant)
    bg = _field_background(args.width * ss, args.height * ss).convert("RGBA")
    n = int(np.clip(args.particles, 30, 480))
    rng = np.random.default_rng(args.seed)
    # full-bleed starts (slightly off-screen) so streamlines fill corners/edges, no black border
    starts = np.stack([rng.uniform(-0.05, 1.05, n), rng.uniform(-0.05, 1.05, n)], axis=1).astype(np.float32)
    # more, finer steps + full-bleed clip -> smoother curves that run off the edges
    paths = _integrate_field_paths(starts, 260, dense, step_size=0.016, lo=-0.06, hi=1.06)
    phase = rng.random(n).astype(np.float32)
    win = 64
    label = getattr(args, "label", "") or ""

    def frame_iter():
        for i in range(frames):
            frame = _render_stream_frame(paths, phase, i / frames, args.width, args.height, win, bg, ss)
            if label:
                img = Image.fromarray(frame)
                _draw_label(ImageDraw.Draw(img), label, 18, 16, args.width)
                frame = np.asarray(img, np.uint8)
            yield frame

    _write_video(args.output, args.fps, frame_iter())
    print(f"[OK] streams: {n} thin silk streamlines @ss{ss} ({variant}), full-bleed seamless -> {Path(args.output)}")


def render_streams_gallery(args: argparse.Namespace) -> None:
    variants = ["tight eddies", "broad streams", "many vortices", "silk lanes"]
    counts = {"tight eddies": 220, "broad streams": 90, "many vortices": 260, "silk lanes": 150}
    frames = max(len(variants), int(args.seconds * args.fps))
    rng = np.random.default_rng(args.seed)
    bg = _field_background(args.width, args.height).convert("RGBA")
    built = []
    for name in variants:
        dense = _make_dense_field(variant=name)
        n = counts[name]
        starts = np.stack([rng.uniform(0.05, 0.95, n), rng.uniform(0.08, 0.90, n)], axis=1).astype(np.float32)
        built.append((name, _integrate_field_paths(starts, 120, dense), rng.random(n).astype(np.float32)))

    def frame_iter():
        for i in range(frames):
            vi = min(len(variants) - 1, int(i * len(variants) / frames))
            name, paths, phase = built[vi]
            lo = int(vi * frames / len(variants))
            hi = max(lo + 1, int((vi + 1) * frames / len(variants)))
            u = (i - lo) / (hi - lo)
            frame = _render_stream_frame(paths, phase, u, args.width, args.height, 30, bg)
            img = Image.fromarray(frame)
            _draw_label(ImageDraw.Draw(img), name, 18, 16, args.width)
            yield np.asarray(img, np.uint8)

    _write_video(args.output, args.fps, frame_iter())
    print(f"[OK] streams_gallery: {len(variants)} field characters as flowing streamlines -> {Path(args.output)}")


def _sample_dense_field(field: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size = int(field["size"])
    fx = np.clip(x, 0.0, 0.9999) * (size - 1)
    fy = np.clip(y, 0.0, 0.9999) * (size - 1)
    ix = fx.astype(np.int32)
    iy = fy.astype(np.int32)
    return field["dx"][iy, ix], field["dy"][iy, ix]


def _integrate_field_paths(starts: np.ndarray, steps: int, dense_field: dict[str, np.ndarray],
                           step_size: float = 0.030, lo: float = 0.08, hi: float = 0.92) -> np.ndarray:
    """Integrate streamlines. lo/hi widen to full-bleed (e.g. -0.05/1.05) so lines reach screen edges."""
    paths = np.zeros((len(starts), steps + 1, 2), dtype=np.float32)
    paths[:, 0, :] = starts
    pos = starts.copy()
    for step in range(steps):
        dx, dy = _sample_dense_field(dense_field, np.clip(pos[:, 0], 0, 1), np.clip(pos[:, 1], 0, 1))
        pos[:, 0] = np.clip(pos[:, 0] + dx * step_size, lo, hi)
        pos[:, 1] = np.clip(pos[:, 1] + dy * step_size, lo, hi)
        paths[:, step + 1, :] = pos
    return paths


def _draw_arrow_grid(draw: ImageDraw.ImageDraw, grid: dict[str, np.ndarray], alpha: float) -> None:
    fill = (70, 230, 236, int(138 * alpha))
    head = (180, 250, 255, int(160 * alpha))
    length = 31 * SS
    for x, y, dx, dy in zip(grid["x"], grid["y"], grid["dx"], grid["dy"]):
        x0 = float(x - dx * length * 0.48)
        y0 = float(y - dy * length * 0.48)
        x1 = float(x + dx * length * 0.48)
        y1 = float(y + dy * length * 0.48)
        draw.line((x0, y0, x1, y1), fill=fill, width=max(1, 2 * SS))
        nx, ny = -float(dy), float(dx)
        hx = float(x1 - dx * 8 * SS)
        hy = float(y1 - dy * 8 * SS)
        draw.polygon(
            ((x1, y1), (hx + nx * 4 * SS, hy + ny * 4 * SS), (hx - nx * 4 * SS, hy - ny * 4 * SS)),
            fill=head,
        )


def _draw_paths(draw: ImageDraw.ImageDraw, paths: np.ndarray, upto: int, width: int, height: int, alpha: float = 1.0) -> None:
    for pi, path in enumerate(paths):
        pts = [(float(x * width), float(y * height)) for x, y in path[: upto + 1]]
        if len(pts) >= 2:
            for glow_width, glow_alpha in ((10, 36), (5, 86), (2, 210)):
                color = (42, 232, 222, int(glow_alpha * alpha))
                if pi % 3 == 1:
                    color = (166, 105, 255, int(glow_alpha * alpha))
                elif pi % 3 == 2:
                    color = (255, 196, 108, int(glow_alpha * alpha))
                draw.line(pts, fill=color, width=max(1, glow_width * SS), joint="curve")
        x, y = pts[-1]
        r = 5 * SS
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(236, 250, 255, int(238 * alpha)))


def render_field(args: argparse.Namespace) -> None:
    frames = max(2, int(args.seconds * args.fps))
    grid = _make_field_grid(args.width, args.height)
    dense = _make_dense_field()
    background = _field_background(args.width, args.height).convert("RGBA")
    single = _integrate_field_paths(np.array([[0.235, 0.505]], dtype=np.float32), 8, dense)
    starts = np.array(
        [[0.20, 0.34], [0.22, 0.39], [0.24, 0.44], [0.26, 0.49], [0.28, 0.54], [0.30, 0.59],
         [0.32, 0.64], [0.36, 0.36], [0.38, 0.43], [0.40, 0.50], [0.42, 0.57], [0.44, 0.64]],
        dtype=np.float32,
    )
    many = _integrate_field_paths(starts, 24, dense)
    stage_a = int(frames * 0.18)
    stage_b = int(frames * 0.64)

    def frame_iter():
        for i in range(frames):
            img = background.copy()
            draw = ImageDraw.Draw(img, "RGBA")
            if i < stage_a:
                arrow_alpha = _smoothstep(i / max(1, stage_a - 1))
                _draw_arrow_grid(draw, grid, float(arrow_alpha))
                _draw_label(draw, "flow field: every point has an arrow", 18, 16, args.width)
            elif i < stage_b:
                _draw_arrow_grid(draw, grid, 1.0)
                local = (i - stage_a) / max(1, stage_b - stage_a - 1)
                step = min(8, int(local * 8.999))
                _draw_paths(draw, single, step, args.width, args.height, 1.0)
                _draw_label(draw, f"Step {step}", 18, 16, args.width)
            else:
                _draw_arrow_grid(draw, grid, 0.48)
                local = (i - stage_b) / max(1, frames - stage_b - 1)
                upto = min(24, max(1, int(local * 24.999)))
                _draw_paths(draw, many, upto, args.width, args.height, 1.0)
                _draw_label(draw, "nearby particles read nearby arrows", 18, 16, args.width)
            yield np.asarray(img.convert("RGB"), dtype=np.uint8)

    _write_video(args.output, args.fps, frame_iter())
    print(f"[OK] field mode precomputed arrow grid + dense guide field once -> {Path(args.output)}")


def render_gallery(args: argparse.Namespace) -> None:
    variants = ["tight eddies", "broad streams", "many vortices", "silk lanes"]
    frames = max(len(variants), int(args.seconds * args.fps))
    particles = {
        name: _make_particles(max(4_000, args.particles + idx * 1_000), args.seed + 97 * idx)
        for idx, name in enumerate(variants)
    }

    def frame_iter():
        for i in range(frames):
            vi = min(len(variants) - 1, int(i * len(variants) / frames))
            name = variants[vi]
            local_i = i - int(vi * frames / len(variants))
            local_frames = max(1, int((vi + 1) * frames / len(variants)) - int(vi * frames / len(variants)))
            u = local_i / local_frames
            yield render_frame(
                particles[name],
                u,
                args.width,
                args.height,
                max(10, args.trail_steps - 2 + vi),
                label=name,
                variant=name,
            )

    _write_video(args.output, args.fps, frame_iter())
    print(f"[OK] gallery mode rendered {len(variants)} flow-field variants -> {Path(args.output)}")


def render_still(args: argparse.Namespace) -> None:
    particles = _make_particles(args.particles_still, args.seed)
    frame = render_frame(particles, 0.37, args.width, args.height, args.trail_steps_still)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        Image.fromarray(frame).save(out)
    else:
        _write_video(args.output, args.fps, (frame for _ in range(max(1, int(args.seconds * args.fps)))))
    print(
        f"[OK] still uses the same periodic curl field; "
        f"cost ~= {args.trail_steps_still} x {args.particles_still:,} vectorized deposits/frame -> {out}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop",
                    choices=["loop", "still", "field", "gallery", "streams", "streams_gallery"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--particles", type=int, default=9_000)
    ap.add_argument("--particles-still", type=int, default=18_000)
    ap.add_argument("--trail-steps", type=int, default=16)
    ap.add_argument("--trail-steps-still", type=int, default=24)
    ap.add_argument("--seed", type=int, default=8107)
    ap.add_argument("--variant", default=None,
                    help="streams field character: 'silk lanes'|'tight eddies'|'broad streams'|'many vortices'")
    ap.add_argument("--label", default=None, help="optional on-screen label for streams mode")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "loop":
        render_loop(args)
    elif args.mode == "still":
        render_still(args)
    elif args.mode == "field":
        render_field(args)
    elif args.mode == "gallery":
        render_gallery(args)
    elif args.mode == "streams":
        render_streams(args)
    elif args.mode == "streams_gallery":
        render_streams_gallery(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
