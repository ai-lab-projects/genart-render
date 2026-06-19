"""Magnetic pendulum mp4 generator for Agentic Pixels.

A damped bob is pulled by three magnets. The final magnet depends sensitively
on the release point, producing fractal basins of attraction.

modes:
  basins     - vectorized basin map from a full grid of starting positions
  swing      - several glowing paths settling onto labeled magnets
  sensitive  - two almost-identical starts diverging to different magnets
  zoom       - recomputed half-resolution zoom into a basin boundary
  still      - one held hero basin still

Usage:
  python magnetic_pendulum_simulator.py --mode basins --output /tmp/magnetic.mp4 --seconds 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BG = np.array([3, 5, 12], dtype=np.float32)
WHITE_HOT = np.array([238, 246, 235], dtype=np.float32)
SS = 1
FPS = 24
WIDTH = 1280
HEIGHT = 720

DT = 0.035
MAX_STEPS = 900
STRENGTH = 0.060
MAGNET_H = 0.085
SPRING_K = 0.18
FRICTION = 0.21
SETTLE_SPEED = 0.045
SETTLE_DIST = 0.16

MAGNETS = np.array(
    [
        (0.0, 0.82),
        (-0.710, -0.410),
        (0.710, -0.410),
    ],
    dtype=np.float32,
)
PALETTE = np.array(
    [
        (70, 218, 255),   # cyan
        (255, 92, 193),   # magenta
        (255, 204, 80),   # gold
    ],
    dtype=np.float32,
)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _smoothstep(x: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _draw_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int) -> None:
    font = _font(max(15 * SS, width // 62))
    margin = 10 * SS
    tw = int(draw.textlength(text, font=font))
    x = max(margin, min(x, width - tw - margin))
    draw.text((x + SS, y + SS), text, font=font, fill=(0, 0, 0, 205))
    draw.text((x, y), text, font=font, fill=(225, 238, 238, 235))


def _view_grid(w: int, h: int, center: tuple[float, float], half_height: float) -> np.ndarray:
    half_width = half_height * w / max(1, h)
    xs = np.linspace(center[0] - half_width, center[0] + half_width, w, dtype=np.float32)
    ys = np.linspace(center[1] + half_height, center[1] - half_height, h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1).astype(np.float32, copy=False)


def _world_to_pixel(
    p: np.ndarray,
    width: int,
    height: int,
    center: tuple[float, float] = (0.0, 0.0),
    half_height: float = 1.32,
) -> tuple[int, int]:
    half_width = half_height * width / max(1, height)
    x = int((float(p[0]) - center[0] + half_width) / (2.0 * half_width) * width)
    y = int((center[1] + half_height - float(p[1])) / (2.0 * half_height) * height)
    return x, y


def _acceleration(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(pos, dtype=np.float32)
    for magnet in MAGNETS:
        delta = magnet - pos
        r2 = np.sum(delta * delta, axis=-1, keepdims=True) + MAGNET_H * MAGNET_H
        acc += STRENGTH * delta / np.power(r2, 1.5)
    acc += -SPRING_K * pos - FRICTION * vel
    return np.clip(acc, -8.0, 8.0).astype(np.float32, copy=False)


def _verlet_step(pos: np.ndarray, vel: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    acc = _acceleration(pos, vel)
    vh = vel + 0.5 * dt * acc
    pn = pos + dt * vh
    an = _acceleration(pn, vh)
    vn = vh + 0.5 * dt * an
    return pn.astype(np.float32, copy=False), vn.astype(np.float32, copy=False)


def basin_field(
    w: int,
    h: int,
    center: tuple[float, float] = (0.0, 0.0),
    half_height: float = 1.32,
    max_steps: int = MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (winner, settle_step) for a whole start-position grid.

    The expensive part is vectorized: pos and vel are [H,W,2] arrays stepped
    together, with no per-pixel Python loop.
    """
    pos = _view_grid(w, h, center, half_height)
    vel = np.zeros_like(pos, dtype=np.float32)
    winner = np.full((h, w), -1, dtype=np.int16)
    settle = np.full((h, w), max_steps, dtype=np.uint16)
    active = np.ones((h, w), dtype=bool)

    for step in range(1, max_steps + 1):
        pos, vel = _verlet_step(pos, vel)
        d = pos[..., None, :] - MAGNETS[None, None, :, :]
        d2 = np.sum(d * d, axis=-1)
        nearest = np.argmin(d2, axis=-1).astype(np.int16)
        speed2 = np.sum(vel * vel, axis=-1)
        done = active & (speed2 < SETTLE_SPEED * SETTLE_SPEED) & (np.min(d2, axis=-1) < SETTLE_DIST * SETTLE_DIST)
        if done.any():
            winner[done] = nearest[done]
            settle[done] = step
            active[done] = False
        if not active.any():
            break

    if active.any():
        d = pos[..., None, :] - MAGNETS[None, None, :, :]
        nearest = np.argmin(np.sum(d * d, axis=-1), axis=-1).astype(np.int16)
        winner[active] = nearest[active]
    return winner, settle


def colorize_basin(winner: np.ndarray, settle: np.ndarray, max_steps: int = MAX_STEPS) -> np.ndarray:
    base = PALETTE[np.clip(winner, 0, 2)]
    t = np.clip(settle.astype(np.float32) / max(1, max_steps), 0.0, 1.0)
    shade = 0.30 + 0.54 * (t ** 0.55)
    stripes = 0.12 * (0.5 + 0.5 * np.sin(settle.astype(np.float32) * 0.45))
    frame = BG * (1.0 - shade[..., None]) + base * (shade + stripes)[..., None]

    slow = np.clip((t - 0.45) / 0.55, 0.0, 1.0)
    frame += WHITE_HOT * (0.18 * slow[..., None])

    glow_src = Image.fromarray((slow * 255).astype(np.uint8), "L")
    glow = np.asarray(glow_src.filter(ImageFilter.GaussianBlur(radius=9 * SS)), dtype=np.float32) / 255.0
    frame += (PALETTE.mean(axis=0) * 0.20 + WHITE_HOT * 0.10) * glow[..., None]
    return np.clip(frame, 0, 255).astype(np.uint8)


def _label_frame(frame: np.ndarray, label: str, x: int = 18, y: int = 16) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    _draw_label(draw, label, x, y, frame.shape[1])
    return np.asarray(img, dtype=np.uint8)


def _draw_magnets(draw: ImageDraw.ImageDraw, width: int, height: int, half_height: float = 1.32) -> None:
    for i, magnet in enumerate(MAGNETS):
        x, y = _world_to_pixel(magnet, width, height, half_height=half_height)
        color = tuple(int(c) for c in PALETTE[i])
        r = 9 * SS
        draw.ellipse((x - r * 3, y - r * 3, x + r * 3, y + r * 3), fill=color + (34,))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color + (235,))
        _draw_label(draw, f"M{i + 1}", x + 12 * SS, y - 11 * SS, width)


def _trace_points(start: tuple[float, float], steps: int, stride: int = 3) -> tuple[np.ndarray, int]:
    pos = np.array(start, dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32)
    pts = []
    winner = -1
    for step in range(steps):
        pos, vel = _verlet_step(pos, vel)
        if step % stride == 0:
            pts.append(pos.copy())
        d2 = np.sum((MAGNETS - pos) ** 2, axis=1)
        if np.dot(vel, vel) < SETTLE_SPEED * SETTLE_SPEED and float(d2.min()) < SETTLE_DIST * SETTLE_DIST:
            winner = int(np.argmin(d2))
            break
    if winner < 0:
        winner = int(np.argmin(np.sum((MAGNETS - pos) ** 2, axis=1)))
    return np.asarray(pts, dtype=np.float32), winner


def _find_sensitive_pair() -> tuple[tuple[float, float], tuple[float, float], tuple[np.ndarray, np.ndarray], tuple[int, int]]:
    explicit_pairs = [
        ((-0.006, 1.2866), (0.006, 1.2866)),
        ((-0.0166, 1.2866), (0.0166, 1.2866)),
    ]
    for a, b in explicit_pairs:
        ta, wa = _trace_points(a, MAX_STEPS, 2)
        tb, wb = _trace_points(b, MAX_STEPS, 2)
        if wa != wb:
            return a, b, (ta, tb), (wa, wb)

    candidates = [
        (-0.32, 0.36), (-0.18, 0.52), (0.06, 0.47), (0.24, 0.30),
        (-0.48, -0.05), (0.42, 0.08), (-0.05, -0.02), (0.18, -0.24),
    ]
    offsets = [(0.004, 0.0), (0.0, 0.004), (0.006, -0.003), (-0.004, 0.005)]
    for base in candidates:
        for off in offsets:
            a = base
            b = (base[0] + off[0], base[1] + off[1])
            ta, wa = _trace_points(a, MAX_STEPS, 2)
            tb, wb = _trace_points(b, MAX_STEPS, 2)
            if wa != wb:
                return a, b, (ta, tb), (wa, wb)
    a, b = explicit_pairs[-1]
    ta, wa = _trace_points(a, MAX_STEPS, 2)
    tb, wb = _trace_points(b, MAX_STEPS, 2)
    return a, b, (ta, tb), (wa, wb)


def _draw_glow_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: tuple[int, int, int], alpha: int) -> None:
    if len(pts) < 2:
        return
    draw.line(pts, fill=color + (max(20, alpha // 4),), width=12 * SS, joint="curve")
    draw.line(pts, fill=color + (max(45, alpha // 2),), width=5 * SS, joint="curve")
    draw.line(pts, fill=color + (alpha,), width=2 * SS, joint="curve")


def _trajectory_frame(
    traces: list[np.ndarray],
    winners: list[int],
    upto: int,
    label: str,
    width: int,
    height: int,
) -> np.ndarray:
    img = Image.new("RGB", (width, height), tuple(BG.astype(np.uint8)))
    glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(glow, "RGBA")

    for trace, winner in zip(traces, winners):
        shown = trace[: min(len(trace), upto)]
        pts = [_world_to_pixel(p, width, height) for p in shown]
        _draw_glow_polyline(draw, pts, tuple(int(c) for c in PALETTE[winner]), 210)
        if pts:
            x, y = pts[-1]
            r = 5 * SS
            draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(c) for c in PALETTE[winner]) + (245,))

    glow = glow.filter(ImageFilter.GaussianBlur(radius=1.1 * SS))
    img = Image.alpha_composite(img.convert("RGBA"), glow)
    draw = ImageDraw.Draw(img, "RGBA")
    _draw_magnets(draw, width, height)
    _draw_label(draw, label, 18 * SS, 16 * SS, width)
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _write_frame(wr, frame: np.ndarray, w: int, h: int) -> None:
    wr.append_data(np.asarray(Image.fromarray(frame).resize((w, h), Image.LANCZOS)))


def _write_static(wr, frame: np.ndarray, frames: int, w: int, h: int) -> None:
    for _ in range(frames):
        _write_frame(wr, frame, w, h)


def render(mode: str, output: str, seconds: float, fps: int = FPS, w: int = WIDTH, h: int = HEIGHT) -> None:
    frames = max(2, int(seconds * fps))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)

    try:
        if mode in {"basins", "still"}:
            # half解像度で計算→_write_frameがLANCZOS拡大 (重い計算ガード: full 1280x720×900step は cap超過KILL)
            winner, settle = basin_field((w * SS) // 2, (h * SS) // 2, max_steps=700)
            frame = colorize_basin(winner, settle, max_steps=700)
            label = "three magnets, one release point" if mode == "basins" else ""
            if label:
                frame = _label_frame(frame, label)
            _write_static(wr, frame, frames, w, h)
        elif mode == "zoom":
            # 磁石振り子の basin は1枚が重い(per-frame再計算は不可能) → 数レベルを事前計算し保持+クロスフェードで「段階ズーム」
            zw, zh = int(w * SS * 0.42), int(h * SS * 0.42)   # 中解像度で各レベル計算 (_write_frameがLANCZOS拡大)
            target = (0.0, -0.2)                              # 境界が全レベルで tangle する点(実測選定, x81は平坦化したため)
            n_levels = 4
            imgs = []
            for k in range(n_levels):
                mag = 2.3 ** k                                # 1, 2.3, 5.3, 12x (これ以上深いと basin 内部=平坦)
                winner, settle = basin_field(zw, zh, center=target, half_height=1.32 / mag, max_steps=480)
                f = colorize_basin(winner, settle, max_steps=480)
                f = _label_frame(f, f"basin boundary  zoom x{mag:.0f}")
                imgs.append(f)
            hold = max(1, frames // n_levels)
            fade = max(1, hold // 4)
            seq_idx = []
            for k in range(n_levels):
                seq_idx += [k] * hold
            seq_idx = (seq_idx + [n_levels - 1] * frames)[:frames]
            for i in range(frames):
                k = seq_idx[i]
                # レベル境界手前の fade フレームで次レベルへクロスフェード
                nxt = min(n_levels - 1, k + 1)
                local = i - (seq_idx.index(k) if k in seq_idx else 0)
                frame = imgs[k]
                # 単純化: hold の最後 fade 枚で次へブレンド
                pos_in_level = i - k * hold
                if k < n_levels - 1 and pos_in_level >= hold - fade:
                    t = (pos_in_level - (hold - fade)) / max(1, fade)
                    frame = (imgs[k].astype(np.float32) * (1 - t) + imgs[nxt].astype(np.float32) * t).astype(np.uint8)
                _write_frame(wr, frame, w, h)
        elif mode == "swing":
            # 全て off-axis の curvy な開始点 (対称軸上=直進する線/ sensitive pair の直線降下を避ける)
            starts = [(-0.86, 0.18), (0.82, 0.20), (0.30, -0.74), (-0.34, -0.62), (0.58, -0.30)]
            traces: list[np.ndarray] = []
            winners: list[int] = []
            for start in starts:
                trace, winner = _trace_points(start, MAX_STEPS, 2)
                traces.append(trace)
                winners.append(winner)
            max_len = max(len(t) for t in traces)
            for i in range(frames):
                frac = i / max(1, frames - 1)
                upto = max(2, int(max_len * float(_smoothstep(frac))))
                frame = _trajectory_frame(traces, winners, upto, "glowing paths settle into magnets", w * SS, h * SS)
                _write_frame(wr, frame, w, h)
        elif mode == "sensitive":
            _a, _b, traces_pair, winners_pair = _find_sensitive_pair()
            traces = [traces_pair[0], traces_pair[1]]
            winners = [winners_pair[0], winners_pair[1]]
            max_len = max(len(t) for t in traces)
            for i in range(frames):
                frac = i / max(1, frames - 1)
                upto = max(2, int(max_len * float(_smoothstep(frac))))
                frame = _trajectory_frame(traces, winners, upto, "a hair apart, different endings", w * SS, h * SS)
                _write_frame(wr, frame, w, h)
        else:
            raise ValueError(f"unknown mode: {mode}")
    finally:
        wr.close()

    print(
        f"[OK] modes=basins,swing,sensitive,zoom,still -> {out}\n"
        "note: basins/zoom step pos,vel as [H,W,2] numpy arrays; zoom uses half-resolution recompute to cap cost."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["basins", "swing", "sensitive", "zoom", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=16.0)
    args = ap.parse_args()
    render(args.mode, args.output, args.seconds)


if __name__ == "__main__":
    main()
