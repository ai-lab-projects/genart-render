"""2D strange-attractor mp4 generator: Clifford and de Jong maps.

This simulator renders an "infinite art gallery" of 2D strange attractors.
Each image is made by iterating a point map with four parameters (a, b, c, d),
accumulating visited coordinates into a density histogram, then log-coloring
the density with a dark background, jewel gradient, and soft bloom.

Cost rule:
  Attractor maps are sequential along one orbit, so this file uses many
  independent seeds in parallel. For example, 20,000 seeds x 100 visible steps
  gives 2,000,000 plotted hops while only looping 100 times in Python, with
  each step vectorized over all seeds. The still/gallery/build modes compute
  each orbit once. Only morph recomputes per frame, and it uses a modest
  default of about 400,000 visible points per frame.

Usage:
  python strange2d_simulator.py --mode still --output outputs/strange2d.mp4
  python strange2d_simulator.py --mode gallery --output outputs/gallery.mp4
  python strange2d_simulator.py --mode build --output outputs/build.mp4
  python strange2d_simulator.py --mode morph --output outputs/morph.mp4
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BG = np.array([2, 4, 9], dtype=np.float32)
CYAN = np.array([38, 224, 255], dtype=np.float32)
GOLD = np.array([255, 203, 82], dtype=np.float32)
MAGENTA = np.array([255, 64, 190], dtype=np.float32)
WHITE_HOT = np.array([255, 244, 214], dtype=np.float32)


@dataclass(frozen=True)
class AttractorSpec:
    name: str
    family: str
    params: tuple[float, float, float, float]
    seed: int


# Curated, well-known attractive parameter regions. The list intentionally
# mixes Clifford and de Jong maps so gallery/morph shots change structure hard.
CURATED: tuple[AttractorSpec, ...] = (
    AttractorSpec("Clifford nebula", "clifford", (1.7, 1.7, 0.6, 1.2), 1101),
    AttractorSpec("Clifford orchid", "clifford", (1.5, -1.8, 1.6, 0.9), 1102),
    AttractorSpec("Clifford lace", "clifford", (-1.4, 1.6, 1.0, 0.7), 1103),
    AttractorSpec("Clifford crown", "clifford", (-1.7, 1.8, -1.9, -0.4), 1104),
    AttractorSpec("Clifford shell", "clifford", (1.4, 1.6, 1.0, 0.7), 1105),
    AttractorSpec("Clifford flame", "clifford", (1.9, 1.9, 1.2, 0.7), 1106),
    AttractorSpec("de Jong veil", "dejong", (1.4, -2.3, 2.4, -2.1), 2101),
    AttractorSpec("de Jong comet", "dejong", (2.01, -2.53, 1.61, -0.33), 2102),
    AttractorSpec("de Jong blossom", "dejong", (-2.7, -0.09, -0.86, -2.2), 2103),
    AttractorSpec("de Jong aurora", "dejong", (-0.827, -1.637, 1.659, -0.943), 2104),
    AttractorSpec("de Jong filament", "dejong", (1.641, 1.902, 0.316, 1.525), 2105),
    AttractorSpec("de Jong cathedral", "dejong", (2.879, -0.765, -0.966, -2.879), 2106),
)

MORPH_KEYS = (CURATED[0], CURATED[1], CURATED[3], CURATED[4])


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


def _map_step(
    family: str,
    x: np.ndarray,
    y: np.ndarray,
    params: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d = params
    if family == "clifford":
        nx = np.sin(a * y) + c * np.cos(a * x)
        ny = np.sin(b * x) + d * np.cos(b * y)
    elif family == "dejong":
        nx = np.sin(a * y) - np.cos(b * x)
        ny = np.sin(c * x) - np.cos(d * y)
    else:
        raise ValueError(f"unknown attractor family: {family}")
    return nx.astype(np.float32, copy=False), ny.astype(np.float32, copy=False)


def generate_points(
    spec: AttractorSpec,
    total_points: int,
    batch_size: int = 20_000,
    transient_steps: int = 64,
) -> np.ndarray:
    """Generate visible points by iterating many independent seeds in parallel."""
    visible_steps = max(1, int(np.ceil(total_points / batch_size)))
    rng = np.random.default_rng(spec.seed)
    x = rng.uniform(-0.5, 0.5, batch_size).astype(np.float32)
    y = rng.uniform(-0.5, 0.5, batch_size).astype(np.float32)

    for _ in range(transient_steps):
        x, y = _map_step(spec.family, x, y, spec.params)

    pts = np.empty((visible_steps * batch_size, 2), dtype=np.float32)
    for i in range(visible_steps):
        x, y = _map_step(spec.family, x, y, spec.params)
        start = i * batch_size
        pts[start : start + batch_size, 0] = x
        pts[start : start + batch_size, 1] = y
    return pts[:total_points]


def _bounds(points: np.ndarray, aspect: float = 16.0 / 9.0) -> tuple[tuple[float, float], tuple[float, float]]:
    lo = np.percentile(points, 0.35, axis=0)
    hi = np.percentile(points, 99.65, axis=0)
    center = (lo + hi) * 0.5
    span = np.maximum(hi - lo, 1e-3)
    span *= 1.10
    if span[0] / span[1] > aspect:
        span[1] = span[0] / aspect
    else:
        span[0] = span[1] * aspect
    xlim = (float(center[0] - span[0] * 0.5), float(center[0] + span[0] * 0.5))
    ylim = (float(center[1] - span[1] * 0.5), float(center[1] + span[1] * 0.5))
    return xlim, ylim


def density_image(
    points: np.ndarray,
    width: int,
    height: int,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> np.ndarray:
    if bounds is None:
        bounds = _bounds(points, width / max(1, height))
    hist, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=(width, height),
        range=(bounds[0], bounds[1]),
    )
    return hist.T[::-1].astype(np.float32, copy=False)


def _gradient(values: np.ndarray) -> np.ndarray:
    t = np.clip(values, 0.0, 1.0).astype(np.float32)
    split = 0.56
    lower = np.clip(t / split, 0.0, 1.0)
    upper = np.clip((t - split) / (1.0 - split), 0.0, 1.0)
    lower = _ease(lower)[..., None]
    upper = _ease(upper)[..., None]
    first = CYAN * (1.0 - lower) + GOLD * lower
    second = GOLD * (1.0 - upper) + MAGENTA * upper
    return np.where((t < split)[..., None], first, second)


def colorize_density(hist: np.ndarray, bloom_radius: int = 11) -> np.ndarray:
    if hist.max() <= 0:
        frame = np.empty((*hist.shape, 3), dtype=np.uint8)
        frame[:] = BG.astype(np.uint8)
        return frame

    logv = np.log1p(hist)
    brightness = logv / max(float(logv.max()), 1e-6)
    brightness = np.clip(brightness ** 0.62, 0.0, 1.0).astype(np.float32)

    glow_src = Image.fromarray((brightness * 255).astype(np.uint8), "L")
    glow_wide = np.asarray(
        glow_src.filter(ImageFilter.GaussianBlur(radius=bloom_radius)),
        dtype=np.float32,
    ) / 255.0
    glow_tight = np.asarray(
        glow_src.filter(ImageFilter.GaussianBlur(radius=max(2, bloom_radius // 3))),
        dtype=np.float32,
    ) / 255.0

    frame = np.empty((*hist.shape, 3), dtype=np.float32)
    frame[:] = BG
    color = _gradient(brightness)
    frame += CYAN * (0.24 * glow_wide[..., None])
    frame += MAGENTA * (0.12 * glow_wide[..., None])
    frame += GOLD * (0.18 * glow_tight[..., None])
    frame = frame * (1.0 - 0.96 * brightness[..., None]) + color * (0.96 * brightness[..., None])

    core = np.clip((brightness - 0.72) / 0.28, 0.0, 1.0) ** 2
    frame = frame * (1.0 - 0.55 * core[..., None]) + WHITE_HOT * (0.55 * core[..., None])
    return np.clip(frame, 0, 255).astype(np.uint8)


def _add_label(frame: np.ndarray, text: str, width: int, height: int) -> np.ndarray:
    if not text:
        return frame
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _font(max(17, width // 62))
    pad = max(18, width // 60)
    fill = (226, 232, 230)
    shadow = (0, 0, 0)
    # ラベルは左上に (チャンネル共通の配置。#028 user: 左下でなく左上に)
    y = pad
    draw.text((pad + 1, y + 1), text, font=font, fill=shadow)
    draw.text((pad, y), text, font=font, fill=fill)
    return np.asarray(img, dtype=np.uint8)


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


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    t = float(_ease(alpha))
    return np.clip(a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t, 0, 255).astype(np.uint8)


def _render_still(args: argparse.Namespace) -> None:
    frames = max(1, int(args.duration * args.fps))
    spec = CURATED[0]
    pts = generate_points(spec, args.points_still, args.batch_size, args.transient_steps)
    hist = density_image(pts, args.width, args.height)
    frame = colorize_density(hist, args.bloom)
    frame = _add_label(frame, spec.name, args.width, args.height)
    _write_video(args.output, args.fps, (frame for _ in range(frames)))


def _render_gallery(args: argparse.Namespace) -> None:
    frames = max(1, int(args.duration * args.fps))
    rendered = []
    for spec in CURATED:
        pts = generate_points(spec, args.points_gallery, args.batch_size, args.transient_steps)
        hist = density_image(pts, args.width, args.height)
        frame = colorize_density(hist, args.bloom)
        rendered.append(_add_label(frame, "same formula, four different numbers", args.width, args.height))

    def frame_iter():
        transition = min(0.38, 1.0 / max(2, len(rendered)))
        for i in range(frames):
            pos = (i / max(1, frames)) * len(rendered)
            idx = min(int(pos), len(rendered) - 1)
            local = pos - idx
            if idx < len(rendered) - 1 and local > 1.0 - transition:
                yield _blend(rendered[idx], rendered[idx + 1], (local - (1.0 - transition)) / transition)
            else:
                yield rendered[idx]

    _write_video(args.output, args.fps, frame_iter())


def _render_build(args: argparse.Namespace) -> None:
    frames = max(1, int(args.duration * args.fps))
    spec = CURATED[6]
    pts = generate_points(spec, args.points_build, args.batch_size, args.transient_steps)
    bounds = _bounds(pts, args.width / max(1, args.height))

    def frame_iter():
        for i in range(frames):
            frac = (i + 1) / frames
            frac = max(0.018, float(_ease(frac)))
            k = max(1000, int(len(pts) * frac))
            hist = density_image(pts[:k], args.width, args.height, bounds)
            frame = colorize_density(hist, args.bloom)
            yield _add_label(frame, "millions of hops", args.width, args.height)

    _write_video(args.output, args.fps, frame_iter())


def _interpolated_spec(t: float) -> AttractorSpec:
    # 異なる curated 間を線形補間すると途中で attractor が退化(スパース)する。
    # 代わりに「密な1セット(Clifford nebula)の近傍を小さく正弦波で揺らす」=全編 dense を保つ
    # "breathing" な変形にする(F18-1: 常に映える)。各ダイヤルを位相違いで微小に動かす。
    base = np.array(CURATED[0].params, dtype=np.float32)  # dense Clifford nebula
    amp = np.array([0.22, 0.22, 0.30, 0.30], dtype=np.float32)
    phase = np.array([0.0, 1.6, 3.1, 4.7], dtype=np.float32)
    wob = amp * np.sin(2.0 * np.pi * (t % 1.0) + phase)
    params = tuple(float(v) for v in (base + wob))
    return AttractorSpec("turning the four dials", "clifford", params, 3100)


def _render_morph(args: argparse.Namespace) -> None:
    frames = max(1, int(args.duration * args.fps))

    def frame_iter():
        for i in range(frames):
            spec = _interpolated_spec(i / max(1, frames))
            pts = generate_points(spec, args.points_morph, args.batch_size, args.transient_steps)
            hist = density_image(pts, args.width, args.height)
            frame = colorize_density(hist, args.bloom)
            yield _add_label(frame, "turning the four dials", args.width, args.height)

    _write_video(args.output, args.fps, frame_iter())


def render(args: argparse.Namespace) -> None:
    if args.mode == "still":
        _render_still(args)
    elif args.mode == "gallery":
        _render_gallery(args)
    elif args.mode == "build":
        _render_build(args)
    elif args.mode == "morph":
        _render_morph(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="still", choices=["still", "gallery", "build", "morph"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--batch-size", type=int, default=20_000)
    ap.add_argument("--transient-steps", type=int, default=64)
    ap.add_argument("--points-still", type=int, default=2_000_000)
    ap.add_argument("--points-gallery", type=int, default=1_200_000)
    ap.add_argument("--points-build", type=int, default=2_000_000)
    ap.add_argument("--points-morph", type=int, default=400_000)
    ap.add_argument("--bloom", type=int, default=11)
    return ap.parse_args()


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
