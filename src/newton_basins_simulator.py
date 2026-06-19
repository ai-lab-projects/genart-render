"""Newton basin mp4 generator for Agentic Pixels.

Each pixel is a starting value in the complex plane. Newton's method
z -> z - f(z) / f'(z) is iterated, then pixels are colored by which root
they converge to and shaded by iteration count so slow boundary regions
become visible as filigree.

modes:
  still      - one held hero basin still
  threshold  - 1-root, 2-root, 3-root comparison panels
  zoom       - recomputed zoom into a z^3-1 basin boundary
  gallery    - held stills for several equations

Usage:
  python newton_basins_simulator.py --mode threshold --output /tmp/basins.mp4 --seconds 12
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = np.array([4, 6, 13], dtype=np.float32)
MAX_ITER = 60
TOL = 1e-6
SS = 1
FPS = 24
WIDTH = 1280
HEIGHT = 720


ArrayFn = Callable[[np.ndarray], np.ndarray]
RootFn = Callable[[np.ndarray, float, float], tuple[np.ndarray, int]]


@dataclass(frozen=True)
class FunctionSpec:
    key: str
    label: str
    f: ArrayFn
    df: ArrayFn
    root_index: RootFn
    colors: int
    center: complex
    half_height: float


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
            (95, 235, 210),   # teal
            (255, 155, 85),   # ember
            (185, 235, 105),  # lime
        ],
        dtype=np.float32,
    )
    if n <= len(base):
        return base[:n]
    extra = []
    for k in range(n - len(base)):
        a = k / max(1, n - len(base))
        extra.append(
            (
                120 + 100 * np.sin(2 * np.pi * a),
                150 + 90 * np.sin(2 * np.pi * a + 2.1),
                190 + 60 * np.sin(2 * np.pi * a + 4.2),
            )
        )
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


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int) -> None:
    font = _font(max(15 * SS, width // 58))
    margin = 10 * SS
    tw = int(draw.textlength(text, font=font))
    x = max(margin, min(x, width - tw - margin))
    draw.text((x + SS, y + SS), text, font=font, fill=(0, 0, 0, 195))
    draw.text((x, y), text, font=font, fill=(224, 238, 244, 235))


def _poly_power(n: int) -> tuple[ArrayFn, ArrayFn, RootFn]:
    roots = _roots(n)

    def f(z: np.ndarray) -> np.ndarray:
        if n == 1:
            return z - 1.0
        if n == 2:
            return z * z - 1.0
        if n == 3:
            return z * z * z - 1.0
        if n == 4:
            z2 = z * z
            return z2 * z2 - 1.0
        return z**n - 1.0

    def df(z: np.ndarray) -> np.ndarray:
        if n == 1:
            return np.ones_like(z)
        if n == 2:
            return 2.0 * z
        if n == 3:
            return 3.0 * z * z
        if n == 4:
            return 4.0 * z * z * z
        return n * (z ** (n - 1))

    def root_index(z: np.ndarray, _view_min_x: float, _view_max_x: float) -> tuple[np.ndarray, int]:
        if n == 1:
            return np.zeros(z.shape, dtype=np.int16), 1
        d2 = np.abs(z[:, None] - roots[None, :]) ** 2
        return np.argmin(d2, axis=1).astype(np.int16), n

    return f, df, root_index


def _cubic_interesting() -> tuple[ArrayFn, ArrayFn, RootFn, int]:
    roots = np.roots([1.0, 0.0, -2.0, 2.0]).astype(np.complex64)

    def f(z: np.ndarray) -> np.ndarray:
        return z * z * z - 2.0 * z + 2.0

    def df(z: np.ndarray) -> np.ndarray:
        return 3.0 * z * z - 2.0

    def root_index(z: np.ndarray, _view_min_x: float, _view_max_x: float) -> tuple[np.ndarray, int]:
        d2 = np.abs(z[:, None] - roots[None, :]) ** 2
        return np.argmin(d2, axis=1).astype(np.int16), len(roots)

    return f, df, root_index, len(roots)


def _sin_family() -> tuple[ArrayFn, ArrayFn, RootFn, int]:
    def f(z: np.ndarray) -> np.ndarray:
        return np.sin(z)

    def df(z: np.ndarray) -> np.ndarray:
        return np.cos(z)

    def root_index(z: np.ndarray, view_min_x: float, view_max_x: float) -> tuple[np.ndarray, int]:
        k_min = int(np.floor((view_min_x - 0.5 * np.pi) / np.pi))
        k_max = int(np.ceil((view_max_x + 0.5 * np.pi) / np.pi))
        count = max(1, k_max - k_min + 1)
        k = np.rint(z.real / np.pi).astype(np.int16)
        idx = np.clip(k - k_min, 0, count - 1).astype(np.int16)
        root_x = k.astype(np.float32) * np.pi
        in_view_root = (root_x >= view_min_x - 0.5 * np.pi) & (root_x <= view_max_x + 0.5 * np.pi)
        idx = np.where(in_view_root, idx, -1).astype(np.int16)
        return idx, count

    return f, df, root_index, 13


def _registry() -> dict[str, FunctionSpec]:
    specs: dict[str, FunctionSpec] = {}
    for n in range(1, 6):
        f, df, root_index = _poly_power(n)
        specs[f"z{n}-1"] = FunctionSpec(
            key=f"z{n}-1",
            label=f"z^{n} - 1",
            f=f,
            df=df,
            root_index=root_index,
            colors=n,
            center=0.0 + 0.0j,
            half_height=1.38 if n >= 3 else 1.20,
        )
    f, df, root_index, colors = _cubic_interesting()
    specs["cubic"] = FunctionSpec(
        key="cubic",
        label="z^3 - 2z + 2",
        f=f,
        df=df,
        root_index=root_index,
        colors=colors,
        center=0.0 + 0.0j,
        half_height=1.75,
    )
    f, df, root_index, colors = _sin_family()
    specs["sin"] = FunctionSpec(
        key="sin",
        label="sin(z)",
        f=f,
        df=df,
        root_index=root_index,
        colors=colors,
        center=0.0 + 0.0j,
        half_height=3.15,
    )
    return specs


FUNCTIONS = _registry()


def newton_field(
    spec: FunctionSpec,
    w: int,
    h: int,
    center: complex,
    half_height: float,
    angle: float = 0.0,
    max_iter: int = MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (root_idx, iter_count, color_count). root_idx is -1 for non-converged pixels."""
    z0 = _grid(w, h, center, half_height, angle)
    z = z0.copy()
    view_min_x = float(z0.real.min())
    view_max_x = float(z0.real.max())
    root_idx = np.full((h, w), -1, dtype=np.int16)
    iter_count = np.full((h, w), max_iter, dtype=np.uint8)
    active = np.ones((h, w), dtype=bool)
    colors = spec.colors

    for it in range(1, max_iter + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            fz = spec.f(z)
            finite = np.isfinite(fz.real) & np.isfinite(fz.imag)
            active &= finite
            conv = active & ((fz.real * fz.real + fz.imag * fz.imag) < TOL * TOL)
        if conv.any():
            idx, colors = spec.root_index(z[conv].ravel(), view_min_x, view_max_x)
            valid = idx >= 0
            yy, xx = np.nonzero(conv)
            root_idx[yy[valid], xx[valid]] = idx[valid]
            iter_count[yy[valid], xx[valid]] = it
            active[conv] = False
        if not active.any():
            break

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            active_pos = np.nonzero(active)
            za = z[active_pos]
            denom = spec.df(za)
            next_z = za - spec.f(za) / denom
            finite_next = np.isfinite(next_z.real) & np.isfinite(next_z.imag)
            z[active_pos] = np.where(finite_next, next_z, za)
            still_active = np.zeros_like(active)
            still_active[active_pos] = finite_next
        active &= still_active

    return root_idx, iter_count, max(colors, int(root_idx.max()) + 1)


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
        shade = 0.42 + 0.5 * t
        stripes = 0.20 * (0.5 + 0.5 * np.sin(iter_count.astype(np.float32) * 1.8))
        shade += stripes * (0.4 + 0.6 * t)
        img[shown] = base[shown] * shade[shown, None]
        edge = shown & (iter_count >= 18)
        img[edge] = img[edge] * 0.80 + np.array([200, 225, 255], dtype=np.float32) * 0.20
    return np.clip(img, 0, 255).astype(np.uint8)


def _write_frame(wr, frame: np.ndarray, w: int, h: int) -> None:
    wr.append_data(np.asarray(Image.fromarray(frame).resize((w, h), Image.LANCZOS)))


def _label_frame(frame: np.ndarray, label: str, x: int = 18, y: int = 16) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    _draw_label(draw, label, x, y, frame.shape[1])
    return np.asarray(img)


def _field_frame(spec: FunctionSpec, w: int, h: int, center: complex | None = None, half_height: float | None = None) -> np.ndarray:
    root_idx, iter_count, colors = newton_field(
        spec,
        w,
        h,
        spec.center if center is None else center,
        spec.half_height if half_height is None else half_height,
    )
    return colorize(root_idx, iter_count, colors)


def _threshold_frame(w: int, h: int) -> np.ndarray:
    panel_w = w // 3
    panel_specs = [
        ("1 solution", FUNCTIONS["z1-1"]),
        ("2 solutions", FUNCTIONS["z2-1"]),
        ("3 solutions", FUNCTIONS["z3-1"]),
    ]
    panels = []
    for label, spec in panel_specs:
        frame = _field_frame(spec, panel_w, h)
        panels.append(_label_frame(frame, label))
    return np.concatenate(panels, axis=1)


def render(mode: str, output: str, seconds: float, fps: int = FPS, w: int = WIDTH, h: int = HEIGHT) -> None:
    frames = max(2, int(seconds * fps))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)

    if mode == "still":
        spec = FUNCTIONS["z5-1"]
        # hero は種明かし前なのでラベル無し (#032視聴 user: cold open に早すぎる数式を置かない)
        frame = _field_frame(spec, w * SS, h * SS)
        for _ in range(frames):
            _write_frame(wr, frame, w, h)
    elif mode == "threshold":
        frame = _threshold_frame(w * SS, h * SS)
        for _ in range(frames):
            _write_frame(wr, frame, w, h)
    elif mode == "zoom":
        spec = FUNCTIONS["z3-1"]
        target = 0.05 + 0.48j  # 境界上の detail-rich 点(中央に三色の交点が来る; 旧 -0.18+0.64j は basin 内部に入り平坦化した)
        # フラクタルズームは毎フレーム再計算が不可避 → half 解像度で計算し _write_frame(LANCZOS)で1280x720へ拡大
        # (重い計算ガード: per-frame コストを 1/4 に。glowなしの basin 着色なので拡大で破綻しない)
        zw, zh = (w * SS) // 2, (h * SS) // 2
        for f in range(frames):
            frac = f / max(1, frames - 1)
            mag = 1.0 + 22.0 * _smoothstep(frac)
            center = target * _smoothstep(frac)
            half_height = 1.38 / mag
            frame = _field_frame(spec, zw, zh, center=center, half_height=half_height)
            frame = _label_frame(frame, f"{spec.label}  zoom x{mag:0.1f}")
            _write_frame(wr, frame, w, h)
    elif mode == "gallery":
        keys = ["z4-1", "z5-1", "cubic", "sin"]
        hold = max(1, frames // len(keys))
        for i, key in enumerate(keys):
            spec = FUNCTIONS[key]
            frame = _label_frame(_field_frame(spec, w * SS, h * SS), spec.label)
            reps = hold if i < len(keys) - 1 else frames - hold * (len(keys) - 1)
            for _ in range(max(1, reps)):
                _write_frame(wr, frame, w, h)
    else:
        raise ValueError(f"unknown mode: {mode}")

    wr.close()
    print(f"[OK] modes=still,threshold,zoom,gallery registry={','.join(FUNCTIONS)} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["still", "threshold", "zoom", "gallery"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    a = ap.parse_args()
    render(a.mode, a.output, a.seconds)


if __name__ == "__main__":
    main()
