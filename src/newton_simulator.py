"""Newton fractal mp4 generator. Basins of attraction for f(z)=z^n-1.

Each pixel is an initial value in the complex plane. Newton's method
z -> z - (z^n - 1) / (n z^(n-1)) is iterated, then pixels are colored by
the root they converge to and shaded by the iteration count.

modes:
  reveal  — cubic z^3-1; iteration cap rises frame by frame so basins appear
  zoom    — cubic z^3-1; slow zoom into a basin boundary
  art     — cubic z^3-1; subtle rotation and micro-zoom beauty beat
  quartic — z^4-1 variant with four-fold symmetry
  still   — cubic z^3-1; honest static basins held as a still clip
  compare — cubic z^3-1; Newton, Halley, and Householder side by side
  trajectory — cubic z^3-1; sample Newton paths over static basins

Usage:
  python newton_simulator.py --mode reveal --output /tmp/newton.mp4 --duration 18
"""
from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = np.array([4, 6, 13], dtype=np.float32)
MAX_ITER = 40
TOL = 1e-6
SS = 2


def _roots(n: int) -> np.ndarray:
    return np.exp(2j * np.pi * np.arange(n, dtype=np.float32) / n).astype(np.complex64)


def _palette(n: int) -> np.ndarray:
    base = np.array(
        [
            (80, 200, 255),   # cyan
            (230, 90, 200),   # magenta
            (255, 200, 90),   # gold
            (120, 255, 160),  # mint
            (145, 120, 255),  # violet
            (255, 120, 110),  # coral
        ],
        dtype=np.float32,
    )
    if n <= len(base):
        return base[:n]
    extra = []
    for k in range(n - len(base)):
        a = k / max(1, n - len(base))
        extra.append((120 + 100 * np.sin(2 * np.pi * a),
                      150 + 90 * np.sin(2 * np.pi * a + 2.1),
                      190 + 60 * np.sin(2 * np.pi * a + 4.2)))
    return np.vstack([base, np.array(extra, dtype=np.float32)])


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _grid(w: int, h: int, center: complex, half_height: float, angle: float = 0.0) -> np.ndarray:
    half_width = half_height * w / h
    xs = np.linspace(-half_width, half_width, w, dtype=np.float32)
    ys = np.linspace(half_height, -half_height, h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    z = (gx + 1j * gy).astype(np.complex64)
    if angle:
        z *= np.complex64(np.exp(1j * angle))
    return z + np.complex64(center)


def _poly_f(z: np.ndarray, n: int) -> np.ndarray:
    if n == 3:
        return z * z * z - 1.0
    if n == 4:
        z2 = z * z
        return z2 * z2 - 1.0
    return z ** n - 1.0


def _derivative(z: np.ndarray, n: int) -> np.ndarray:
    if n == 3:
        return 3.0 * z * z
    if n == 4:
        return 4.0 * z * z * z
    return n * (z ** (n - 1))


def newton_step(z: np.ndarray, n: int) -> np.ndarray:
    f = _poly_f(z, n)
    fp = _derivative(z, n)
    return z - f / fp


def halley_step(z: np.ndarray, n: int) -> np.ndarray:
    if n != 3:
        raise ValueError("halley_step is implemented for z^3-1 only")
    f = z * z * z - 1.0
    fp = 3.0 * z * z
    fpp = 6.0 * z
    return z - (2.0 * f * fp) / (2.0 * fp * fp - f * fpp)


def householder3_step(z: np.ndarray, n: int) -> np.ndarray:
    if n != 3:
        raise ValueError("householder3_step is implemented for z^3-1 only")
    f = z * z * z - 1.0
    fp = 3.0 * z * z
    fpp = 6.0 * z
    return z - (f / fp) * (1.0 + (f * fpp) / (2.0 * fp * fp))


def newton_field(
    n: int,
    w: int,
    h: int,
    center: complex,
    half_height: float,
    angle: float = 0.0,
    max_iter: int = MAX_ITER,
    step_fn=newton_step,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (root_idx, iter_count). root_idx is -1 for non-converged pixels."""
    z = _grid(w, h, center, half_height, angle)
    root_idx = np.full((h, w), -1, dtype=np.int16)
    iter_count = np.full((h, w), max_iter, dtype=np.uint8)
    active = np.ones((h, w), dtype=bool)

    for it in range(1, max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            f = _poly_f(z, n)
            finite = (
                np.isfinite(f.real)
                & np.isfinite(f.imag)
            )
            active &= finite
            conv = active & ((f.real * f.real + f.imag * f.imag) < TOL * TOL)
        if conv.any():
            ang = np.mod(np.angle(z[conv]), 2 * np.pi)
            nearest = np.floor(ang / (2 * np.pi / n) + 0.5).astype(np.int16) % n
            root_idx[conv] = nearest
            iter_count[conv] = it
            active[conv] = False
        if not active.any():
            break

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            active_pos = np.nonzero(active)
            next_z = step_fn(z[active_pos], n)
            finite_next = np.isfinite(next_z.real) & np.isfinite(next_z.imag)
            z[active_pos] = np.where(finite_next, next_z, z[active_pos])
            still_active = np.zeros_like(active)
            still_active[active_pos] = finite_next
        active &= still_active

    return root_idx, iter_count


def colorize(root_idx: np.ndarray, iter_count: np.ndarray, n: int, reveal_iter: int | None = None) -> np.ndarray:
    img = np.zeros(root_idx.shape + (3,), dtype=np.float32)
    img[:] = BG
    shown = root_idx >= 0
    if reveal_iter is not None:
        shown &= iter_count <= reveal_iter
    if shown.any():
        pal = _palette(n)
        base = pal[np.clip(root_idx, 0, n - 1)]
        t = np.clip(iter_count.astype(np.float32) / MAX_ITER, 0.0, 1.0)
        # Slow convergence lives near fractal boundaries. Use soft stripes there
        # so the basin edge reads as dark-to-bright filigree instead of flat color.
        # #023 brand: dark jewel-tones. 内部を深く(暗め)、境界の filigree を光らせる。
        shade = 0.42 + 0.5 * t
        stripes = 0.20 * (0.5 + 0.5 * np.sin(iter_count.astype(np.float32) * 1.8))
        shade += stripes * (0.4 + 0.6 * t)
        img[shown] = base[shown] * shade[shown, None]
        edge = shown & (iter_count >= 18)
        img[edge] = img[edge] * 0.80 + np.array([200, 225, 255], dtype=np.float32) * 0.20
    return np.clip(img, 0, 255).astype(np.uint8)


def _write_frame(wr, frame: np.ndarray, w: int, h: int) -> None:
    wr.append_data(np.asarray(Image.fromarray(frame).resize((w, h), Image.LANCZOS)))


def _zoom_image(frame: np.ndarray, frac: float, max_zoom: float = 1.22) -> np.ndarray:
    img = Image.fromarray(frame)
    fw, fh = img.size
    zoom = 1.0 + (max_zoom - 1.0) * _smoothstep(frac)
    cw = max(1, int(fw / zoom))
    ch = max(1, int(fh / zoom))
    left = (fw - cw) // 2
    top = (fh - ch) // 2
    return np.asarray(img.crop((left, top, left + cw, top + ch)).resize((fw, fh), Image.LANCZOS))


def _rotate_image(frame: np.ndarray, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    img = Image.fromarray(frame)
    fw, fh = img.size
    if scale != 1.0:
        cw = max(1, int(fw / scale))
        ch = max(1, int(fh / scale))
        left = (fw - cw) // 2
        top = (fh - ch) // 2
        img = img.crop((left, top, left + cw, top + ch)).resize((fw, fh), Image.LANCZOS)
    return np.asarray(img.rotate(angle_deg, resample=Image.BICUBIC, fillcolor=tuple(BG.astype(np.uint8))))


def _label_panel(frame: np.ndarray, label: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # font size scaled to panel width so it stays legible after downscale.
    fsize = max(22, int(frame.shape[1] * 0.055))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
    except Exception:
        font = ImageFont.load_default()
    x, y = 22, 18
    bbox = draw.textbbox((x, y), label, font=font)
    pad = 10
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), label, fill=(245, 248, 255), font=font)
    return np.asarray(img)


def _complex_to_pixel(z: complex, w: int, h: int, center: complex, half_height: float) -> tuple[float, float]:
    half_width = half_height * w / h
    x = (z.real - center.real + half_width) / (2.0 * half_width) * (w - 1)
    y = (half_height - (z.imag - center.imag)) / (2.0 * half_height) * (h - 1)
    return float(x), float(y)


def _trajectory_path(start: complex, steps: int = 15) -> tuple[list[complex], int]:
    z = np.array([start], dtype=np.complex64)
    path = [complex(z[0])]
    for _ in range(steps):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z = newton_step(z, 3)
        if not np.isfinite(z[0].real) or not np.isfinite(z[0].imag):
            break
        path.append(complex(z[0]))
    root = int(np.floor((np.mod(np.angle(path[-1]), 2 * np.pi) / (2 * np.pi / 3)) + 0.5) % 3)
    return path, root


def _draw_trajectory_frame(
    background: np.ndarray,
    paths: list[list[complex]],
    root_ids: list[int],
    frac: float,
    center: complex,
    half_height: float,
) -> np.ndarray:
    img = Image.fromarray(background).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    h, w = background.shape[:2]
    pal = _palette(3).astype(np.uint8)
    roots = _roots(3)
    for idx, root in enumerate(roots):
        x, y = _complex_to_pixel(complex(root), w, h, center, half_height)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(*pal[idx].tolist(), 255), outline=(255, 255, 255, 230), width=2)

    for path, root_id in zip(paths, root_ids):
        if len(path) < 2:
            continue
        color = pal[root_id].tolist()
        pos = frac * (len(path) - 1)
        whole = int(np.floor(pos))
        local = pos - whole
        points: list[tuple[float, float]] = []
        for i in range(min(whole + 1, len(path))):
            points.append(_complex_to_pixel(path[i], w, h, center, half_height))
        if whole + 1 < len(path):
            a, b = path[whole], path[whole + 1]
            z = a + (b - a) * local
            points.append(_complex_to_pixel(z, w, h, center, half_height))
        start = max(0, len(points) - 8)
        trail = points[start:]
        for i in range(1, len(trail)):
            alpha = int(45 + 170 * i / max(1, len(trail) - 1))
            draw.line((trail[i - 1], trail[i]), fill=(*color, alpha), width=3)
        if points:
            x, y = points[-1]
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(*color, 255), outline=(255, 255, 255, 210), width=2)
    return np.asarray(img)


def _view(mode: str, frac: float) -> tuple[int, complex, float, float]:
    if mode == "quartic":
        # Four roots at 1, i, -1, -i. A slight drift keeps the symmetry alive.
        angle = 0.10 * np.sin(2 * np.pi * frac)
        zoom = 1.0 + 0.10 * _smoothstep(frac)
        return 4, 0.0 + 0.0j, 1.45 / zoom, angle
    if mode == "zoom":
        target = -0.18 + 0.64j
        z = 1.0 + 10.0 * _smoothstep(frac)
        center = target * _smoothstep(frac)
        return 3, center, 1.38 / z, 0.0
    if mode == "art":
        beat = np.sin(2 * np.pi * frac)
        return 3, 0.03 * beat + 0.02j * np.sin(2 * np.pi * frac + 1.2), 1.33 / (1.0 + 0.16 * _smoothstep(frac)), 0.18 * beat
    return 3, 0.0 + 0.0j, 1.38, 0.0


def render(mode: str, output: str, duration: float, fps: int, w: int, h: int) -> None:
    frames = max(2, int(duration * fps))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    W, H = w * SS, h * SS

    if mode == "reveal":
        # Full-grid computations: 1. The reveal is a mask over one static field.
        n, center, half_height, angle = _view(mode, 0.0)
        root_idx, iter_count = newton_field(n, W, H, center, half_height, angle)
        for f in range(frames):
            frac = f / max(1, frames - 1)
            cap = 1 + int(_smoothstep(frac) * (MAX_ITER - 1))
            frame = colorize(root_idx, iter_count, n, reveal_iter=cap)
            _write_frame(wr, frame, w, h)
    elif mode == "still":
        # Full-grid computations: 1. Honest cubic basins held as an unchanged still.
        n, center, half_height, angle = _view("reveal", 0.0)
        root_idx, iter_count = newton_field(n, W, H, center, half_height, angle)
        frame = colorize(root_idx, iter_count, n)
        for _ in range(frames):
            _write_frame(wr, frame, w, h)
    elif mode == "compare":
        # Full-grid computations: 3. One field each for Newton, Halley, Householder.
        labels = ["Newton", "Halley", "Householder"]
        steps = [newton_step, halley_step, householder3_step]
        panels = []
        panel_w = max(1, W // 3)
        for label, step in zip(labels, steps):
            root_idx, iter_count = newton_field(3, panel_w, H, 0.0 + 0.0j, 1.38, step_fn=step)
            panels.append(_label_panel(colorize(root_idx, iter_count, 3), label))
        frame = np.concatenate(panels, axis=1)
        for _ in range(frames):
            _write_frame(wr, frame, w, h)
    elif mode == "trajectory":
        # Full-grid computations: 1. Background basins are static; frames only draw paths.
        center, half_height = 0.0 + 0.0j, 1.38
        root_idx, iter_count = newton_field(3, W, H, center, half_height)
        bg = colorize(root_idx, iter_count, 3).astype(np.float32)
        bg = np.clip(bg * 0.33 + BG * 0.67, 0, 255).astype(np.uint8)
        starts = [
            -1.15 + 0.82j,
            -0.78 - 0.68j,
            -0.22 + 1.02j,
            0.12 - 0.92j,
            0.42 + 0.58j,
            0.78 - 0.18j,
            1.18 + 0.38j,
        ]
        path_pairs = [_trajectory_path(start) for start in starts]
        paths = [p for p, _ in path_pairs]
        root_ids = [r for _, r in path_pairs]
        for f in range(frames):
            frac = f / max(1, frames - 1)
            frame = _draw_trajectory_frame(bg, paths, root_ids, frac, center, half_height)
            _write_frame(wr, frame, w, h)
    else:
        # Full-grid computations: 1. Legacy animated modes transform one static field.
        n, center, half_height, angle = _view(mode, 0.0)
        root_idx, iter_count = newton_field(n, W, H, center, half_height, angle)
        base_frame = colorize(root_idx, iter_count, n)
        for f in range(frames):
            frac = f / max(1, frames - 1)
            if mode == "zoom":
                frame = _zoom_image(base_frame, frac, max_zoom=1.55)
            elif mode == "art":
                beat = np.sin(2 * np.pi * frac)
                frame = _rotate_image(base_frame, 10.0 * beat, scale=1.0 + 0.10 * _smoothstep(frac))
            elif mode == "quartic":
                frame = _rotate_image(base_frame, 5.0 * np.sin(2 * np.pi * frac), scale=1.0 + 0.06 * _smoothstep(frac))
            else:
                frame = base_frame
            _write_frame(wr, frame, w, h)

    wr.close()
    print(f"[OK] {mode} -> {out} ({frames}f @ {fps}fps, {w}x{h}, ss={SS}x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="reveal", choices=["reveal", "zoom", "art", "quartic", "still", "compare", "trajectory"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    render(a.mode, a.output, a.duration, a.fps, a.width, a.height)


if __name__ == "__main__":
    main()
