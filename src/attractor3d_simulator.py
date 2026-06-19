"""3D iterated-map strange-attractor mp4 generator.

This renderer uses a Pickover/de-Jong-style 3D map, not an ODE flow:

  x' = sin(a*y) - z*cos(b*x)
  y' = z*sin(c*x) - cos(d*y)
  z' = sin(e*x)

Cost rule:
  The 3D point cloud is generated once per preset, after a short transient.
  Camera motion only re-projects the same stored points with a rotation matrix
  and perspective divide. The map is never re-iterated per frame.

Usage:
  python attractor3d_simulator.py --mode build --output outputs/attractor3d_build.mp4
  python attractor3d_simulator.py --mode orbit --output outputs/attractor3d_orbit.mp4
  python attractor3d_simulator.py --mode gallery --output outputs/attractor3d_gallery.mp4
  python attractor3d_simulator.py --mode still --output outputs/attractor3d_still.mp4
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BG = np.array([2, 4, 11], dtype=np.float32)
CYAN = np.array([42, 219, 255], dtype=np.float32)
GOLD = np.array([255, 199, 92], dtype=np.float32)
MAGENTA = np.array([255, 78, 180], dtype=np.float32)
WHITE_HOT = np.array([255, 244, 220], dtype=np.float32)
SS = 1  # 点群+bloom は blur 自体が AA。SS=2 は per-frame GaussianBlur が重く orbit が run_capped KILL されるため 1 に (重い計算ガード)


@dataclass(frozen=True)
class AttractorSpec:
    name: str
    params: tuple[float, float, float, float, float]
    seed: tuple[float, float, float]


# Bounded by construction after the first step: z is in [-1, 1], and x/y stay
# inside roughly [-2, 2]. These presets were chosen for layered, dense clouds.
CURATED: tuple[AttractorSpec, ...] = (
    AttractorSpec("sine glass nebula", (1.78, 1.46, 2.08, 1.18, 1.71), (0.12, -0.31, 0.27)),
    AttractorSpec("folded cosine veil", (2.21, -1.62, 1.34, 2.46, -1.27), (-0.19, 0.23, -0.11)),
    AttractorSpec("warm orbit lace", (-1.91, 2.37, 1.73, -2.08, 1.53), (0.31, 0.07, -0.21)),
    AttractorSpec("layered sine cathedral", (2.64, 1.27, -2.18, 1.86, 2.03), (-0.27, -0.16, 0.35)),
)


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


def _ease(t: float | np.ndarray) -> float | np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int) -> None:
    # F16-3 (#030 v2視聴 user「ラベル大きすぎ」): SS二重掛けを是正し width//52 に縮小
    font = _font(max(13 * SS, width // 52))
    # フレーム端でラベルが見切れないよう右端をクランプ (#030 QA: "120 circles" 見切れ対策)
    margin = 10 * SS
    tw = int(draw.textlength(text, font=font))
    x = max(margin, min(x, width - tw - margin))
    draw.text((x + SS, y + SS), text, font=font, fill=(0, 0, 0, 190))
    draw.text((x, y), text, font=font, fill=(222, 236, 230, 230))


def _map_step(
    x: float,
    y: float,
    z: float,
    params: tuple[float, float, float, float, float],
) -> tuple[float, float, float]:
    a, b, c, d, e = params
    nx = np.sin(a * y) - z * np.cos(b * x)
    ny = z * np.sin(c * x) - np.cos(d * y)
    nz = np.sin(e * x)
    return float(nx), float(ny), float(nz)


def generate_points(
    spec: AttractorSpec,
    total_points: int = 600_000,
    transient_steps: int = 1_000,
) -> np.ndarray:
    """Generate one sequential 3D orbit and validate that it stays finite."""
    x, y, z = spec.seed
    for _ in range(transient_steps):
        x, y, z = _map_step(x, y, z, spec.params)
        if not np.isfinite((x, y, z)).all():
            raise ValueError(f"{spec.name} became non-finite during transient")

    pts = np.empty((total_points, 3), dtype=np.float32)
    for i in range(total_points):
        x, y, z = _map_step(x, y, z, spec.params)
        if not np.isfinite((x, y, z)).all():
            raise ValueError(f"{spec.name} became non-finite at point {i}")
        pts[i] = (x, y, z)

    max_abs = float(np.max(np.abs(pts)))
    if max_abs > 3.25:
        raise ValueError(f"{spec.name} escaped expected bounds: max |p|={max_abs:.3f}")
    return pts


def _normalizer(points: np.ndarray) -> tuple[np.ndarray, float]:
    lo = np.percentile(points, 0.4, axis=0)
    hi = np.percentile(points, 99.6, axis=0)
    center = ((lo + hi) * 0.5).astype(np.float32)
    radius = float(np.max(hi - lo) * 0.5)
    return center, max(radius, 1e-3)


def _project(
    points: np.ndarray,
    width: int,
    height: int,
    azimuth: float,
    center: np.ndarray,
    radius: float,
    elev: float = 0.34,
) -> tuple[np.ndarray, np.ndarray]:
    p = (points - center) / radius
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ce, se = np.cos(elev), np.sin(elev)

    x = ca * p[:, 0] - sa * p[:, 1]
    y = sa * p[:, 0] + ca * p[:, 1]
    z = p[:, 2]
    vy = ce * y - se * z
    vz = se * y + ce * z

    distance = 3.2
    perspective = distance / np.maximum(0.8, distance - vy)
    scale = min(width, height) * SS * 0.47
    px = width * SS * 0.5 + x * perspective * scale
    py = height * SS * 0.51 - vz * perspective * scale
    return px, py


def density_image(
    points: np.ndarray,
    width: int,
    height: int,
    azimuth: float,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    px, py = _project(points, width, height, azimuth, center, radius)
    xi = px.astype(np.int32)
    yi = py.astype(np.int32)
    w = width * SS
    h = height * SS
    keep = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    hist = np.zeros((h, w), dtype=np.float32)
    np.add.at(hist, (yi[keep], xi[keep]), 1.0)
    return hist


def _gradient(values: np.ndarray) -> np.ndarray:
    t = np.clip(values, 0.0, 1.0).astype(np.float32)
    split = 0.58
    lower = np.clip(t / split, 0.0, 1.0)
    upper = np.clip((t - split) / (1.0 - split), 0.0, 1.0)
    lower = _ease(lower)[..., None]
    upper = _ease(upper)[..., None]
    first = CYAN * (1.0 - lower) + GOLD * lower
    second = GOLD * (1.0 - upper) + MAGENTA * upper
    return np.where((t < split)[..., None], first, second)


def colorize_density(hist: np.ndarray, bloom_radius: int = 10) -> np.ndarray:
    if hist.max() <= 0:
        frame = np.empty((*hist.shape, 3), dtype=np.uint8)
        frame[:] = BG.astype(np.uint8)
        return frame

    logv = np.log1p(hist)
    brightness = logv / max(float(logv.max()), 1e-6)
    brightness = np.clip(brightness ** 0.58, 0.0, 1.0).astype(np.float32)

    glow_src = Image.fromarray((brightness * 255).astype(np.uint8), "L")
    glow_wide = np.asarray(
        glow_src.filter(ImageFilter.GaussianBlur(radius=bloom_radius * SS)),
        dtype=np.float32,
    ) / 255.0
    glow_tight = np.asarray(
        glow_src.filter(ImageFilter.GaussianBlur(radius=max(2, bloom_radius // 3) * SS)),
        dtype=np.float32,
    ) / 255.0

    frame = np.empty((*hist.shape, 3), dtype=np.float32)
    frame[:] = BG
    color = _gradient(brightness)
    frame += CYAN * (0.22 * glow_wide[..., None])
    frame += MAGENTA * (0.10 * glow_wide[..., None])
    frame += GOLD * (0.20 * glow_tight[..., None])
    frame = frame * (1.0 - 0.95 * brightness[..., None]) + color * (0.95 * brightness[..., None])

    core = np.clip((brightness - 0.70) / 0.30, 0.0, 1.0) ** 2
    frame = frame * (1.0 - 0.56 * core[..., None]) + WHITE_HOT * (0.56 * core[..., None])
    return np.clip(frame, 0, 255).astype(np.uint8)


def _finish_frame(frame: np.ndarray, label: str, width: int, height: int) -> np.ndarray:
    img = Image.fromarray(frame, "RGB")
    draw = ImageDraw.Draw(img)
    _draw_label(draw, label, 24 * SS, 22 * SS, width * SS)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def _render_frame(
    points: np.ndarray,
    azimuth: float,
    label: str,
    width: int,
    height: int,
    center: np.ndarray,
    radius: float,
    bloom: int,
) -> np.ndarray:
    hist = density_image(points, width, height, azimuth, center, radius)
    frame = colorize_density(hist, bloom)
    return _finish_frame(frame, label, width, height)


def _write_video(output: str, fps: int, frames_iter) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=8,
    )
    try:
        for frame in frames_iter:
            writer.append_data(frame)
    finally:
        writer.close()


def _render_build(args: argparse.Namespace) -> None:
    frames = max(1, int(args.seconds * args.fps))
    spec = CURATED[0]
    pts = generate_points(spec, args.points, args.transient_steps)
    center, radius = _normalizer(pts)

    def frame_iter():
        for i in range(frames):
            frac = _ease((i + 1) / frames)
            k = max(2_000, int(len(pts) * frac))
            az = -0.35 + 0.35 * frac
            yield _render_frame(pts[:k], az, "a 3D map builds a cloud", args.width, args.height, center, radius, args.bloom)

    _write_video(args.output, args.fps, frame_iter())


def _render_orbit(args: argparse.Namespace) -> None:
    frames = max(1, int(args.seconds * args.fps))
    spec = CURATED[0]
    pts = generate_points(spec, args.points, args.transient_steps)
    center, radius = _normalizer(pts)

    def frame_iter():
        for i in range(frames):
            az = 2.0 * np.pi * i / max(1, frames)
            yield _render_frame(pts, az, "the same points, seen from every angle", args.width, args.height, center, radius, args.bloom)

    _write_video(args.output, args.fps, frame_iter())


def _render_gallery(args: argparse.Namespace) -> None:
    frames = max(1, int(args.seconds * args.fps))
    clouds = []
    for spec in CURATED:
        pts = generate_points(spec, args.points_gallery, args.transient_steps)
        center, radius = _normalizer(pts)
        clouds.append((spec, pts, center, radius))

    def frame_iter():
        for i in range(frames):
            pos = (i / max(1, frames)) * len(clouds)
            idx = min(int(pos), len(clouds) - 1)
            local = pos - idx
            spec, pts, center, radius = clouds[idx]
            az = 2.0 * np.pi * local
            yield _render_frame(pts, az, spec.name, args.width, args.height, center, radius, args.bloom)

    _write_video(args.output, args.fps, frame_iter())


def _render_still(args: argparse.Namespace) -> None:
    frames = max(1, int(args.seconds * args.fps))
    spec = CURATED[0]
    pts = generate_points(spec, args.points, args.transient_steps)
    center, radius = _normalizer(pts)
    frame = _render_frame(pts, 0.72, "Agentic Pixels", args.width, args.height, center, radius, args.bloom)
    _write_video(args.output, args.fps, (frame for _ in range(frames)))


def render(args: argparse.Namespace) -> None:
    if args.mode == "build":
        _render_build(args)
    elif args.mode == "orbit":
        _render_orbit(args)
    elif args.mode == "gallery":
        _render_gallery(args)
    elif args.mode == "still":
        _render_still(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="orbit", choices=["build", "orbit", "gallery", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--points", type=int, default=600_000)
    ap.add_argument("--points-gallery", type=int, default=420_000)
    ap.add_argument("--transient-steps", type=int, default=1_000)
    ap.add_argument("--bloom", type=int, default=10)
    return ap.parse_args()


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
