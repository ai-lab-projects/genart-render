"""2D rigid-body physics sandbox — the 'satisfying physics' video format, self-implemented (numpy).

No engine, no MCP: we integrate gravity + elastic ball-ball + ball-wall collisions ourselves. The
flagship scene = colorful balls bouncing inside a slowly ROTATING hexagon (the spinning walls keep
feeding energy so it never dies), with glow + motion trails. Seamless-ish, hypnotic, zero copyright.

Scenes:
  hexagon  — balls bouncing in a rotating hexagon (default)
  plinko   — balls cascading down a peg field (Galton board)

CLI:
  python physics_sim.py --scene hexagon --output out.mp4 --seconds 30
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import imageio.v2 as imageio

OUT_W, OUT_H = 1280, 720
FPS = 30
PALETTE = np.array([
    (255, 99, 132), (54, 200, 235), (255, 206, 86), (120, 230, 160),
    (180, 140, 255), (255, 160, 90), (120, 200, 255), (240, 120, 200),
], np.float32)


# ----------------------------------------------------------------- hexagon scene
def sim_hexagon(seconds, fps, seed):
    rng = np.random.default_rng(seed)
    N = 18
    R = 0.46                         # hexagon apothem (center->edge), in normalized units
    rad = 0.030                      # ball radius
    g = np.array([0.0, -0.7], np.float32)
    rest = 0.94                      # restitution (slightly lossy; spin re-adds energy)
    omega = 0.85                     # hexagon angular velocity (rad/s) - keeps it lively
    # init balls near center, random velocities
    ang = rng.uniform(0, 2 * np.pi, N)
    P = np.stack([0.16 * np.cos(ang), 0.16 * np.sin(ang)], 1).astype(np.float32)
    V = rng.uniform(-0.4, 0.4, (N, 2)).astype(np.float32)
    cols = PALETTE[np.arange(N) % len(PALETTE)]
    trails = [[] for _ in range(N)]

    steps = int(seconds * fps)
    sub = 8
    dt = 1.0 / (fps * sub)
    theta = 0.0
    frames = []
    for s in range(steps):
        for _ in range(sub):
            theta += omega * dt
            V += g * dt
            P += V * dt
            # ball-ball elastic collisions (equal mass)
            for i in range(N):
                for j in range(i + 1, N):
                    d = P[j] - P[i]; dist = np.hypot(*d)
                    if dist < 2 * rad and dist > 1e-6:
                        n = d / dist
                        overlap = 2 * rad - dist
                        P[i] -= n * overlap * 0.5; P[j] += n * overlap * 0.5
                        rv = V[j] - V[i]; vn = np.dot(rv, n)
                        if vn < 0:
                            imp = -(1 + rest) * vn * 0.5
                            V[i] -= imp * n; V[j] += imp * n
            # hexagon walls: 6 edges with outward normals rotating at theta
            for i in range(N):
                for k in range(6):
                    a = theta + k * np.pi / 3
                    n = np.array([np.cos(a), np.sin(a)], np.float32)
                    pen = np.dot(P[i], n) - (R - rad)
                    if pen > 0:
                        P[i] -= n * pen
                        vn = np.dot(V[i], n)
                        if vn > 0:
                            V[i] -= (1 + rest) * vn * n
                        # spinning wall imparts tangential velocity (energy feed)
                        tang = np.array([-n[1], n[0]], np.float32)
                        wall_v = omega * R                       # speed of wall surface
                        V[i] += tang * wall_v * 0.22
        for i in range(N):
            trails[i].append(P[i].copy())
            if len(trails[i]) > 14:
                trails[i].pop(0)
        frames.append(_draw_hex(P, V, cols, rad, R, theta, trails))
    return frames


SCALE = 0.72
def _to_px(p):
    return (OUT_W / 2 + p[0] * OUT_H * SCALE, OUT_H / 2 - p[1] * OUT_H * SCALE)


def _draw_hex(P, V, cols, rad, R, theta, trails):
    im = Image.new("RGB", (OUT_W, OUT_H), (8, 10, 18))
    d = ImageDraw.Draw(im)
    rpx = rad * OUT_H * SCALE
    # hexagon outline
    verts = []
    for k in range(6):
        a = theta + k * np.pi / 3 + np.pi / 6
        vx = R / np.cos(np.pi / 6) * np.cos(a); vy = R / np.cos(np.pi / 6) * np.sin(a)
        verts.append(_to_px(np.array([vx, vy])))
    d.polygon(verts, outline=(90, 110, 160))
    for k in range(6):
        d.line([verts[k], verts[(k + 1) % 6]], fill=(120, 150, 210), width=4)
    # trails
    for i in range(len(P)):
        if len(trails[i]) > 1:
            pts = [_to_px(p) for p in trails[i]]
            d.line(pts, fill=tuple(int(c * 0.4) for c in cols[i]), width=max(2, int(rpx * 0.5)))
    # balls (with a highlight for volume)
    for i in range(len(P)):
        x, y = _to_px(P[i])
        c = tuple(int(v) for v in cols[i])
        d.ellipse([x - rpx, y - rpx, x + rpx, y + rpx], fill=c)
        hl = tuple(min(255, int(v * 1.4) + 40) for v in cols[i])
        d.ellipse([x - rpx * 0.4 - rpx * 0.3, y - rpx * 0.4 - rpx * 0.3,
                   x - rpx * 0.4 + rpx * 0.25, y - rpx * 0.4 + rpx * 0.25], fill=hl)
    bloom = im.filter(ImageFilter.GaussianBlur(5))
    return np.clip(np.asarray(im, np.float32) + 0.4 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- plinko scene
def sim_plinko(seconds, fps, seed):
    rng = np.random.default_rng(seed)
    g = np.array([0.0, -1.1], np.float32)
    rad = 0.014
    # peg grid
    pegs = []
    rows = 11
    for r in range(rows):
        y = 0.34 - r * 0.052
        n = r + 3
        for c in range(n):
            x = (c - (n - 1) / 2) * 0.062
            pegs.append((x, y))
    pegs = np.array(pegs, np.float32)
    P = []; V = []; cols = []
    steps = int(seconds * fps); sub = 6; dt = 1.0 / (fps * sub)
    spawn_every = max(1, int(fps * 0.18))
    frames = []
    ci = 0
    for s in range(steps):
        if s % spawn_every == 0 and len(P) < 120:
            P.append([rng.uniform(-0.03, 0.03), 0.44]); V.append([rng.uniform(-0.05, 0.05), 0.0])
            cols.append(PALETTE[ci % len(PALETTE)]); ci += 1
        Pa = np.array(P, np.float32) if P else np.zeros((0, 2), np.float32)
        Va = np.array(V, np.float32) if V else np.zeros((0, 2), np.float32)
        for _ in range(sub):
            Va += g * dt; Pa += Va * dt
            # peg collisions
            for i in range(len(Pa)):
                dd = Pa[i] - pegs; dn = np.hypot(dd[:, 0], dd[:, 1])
                hit = np.where(dn < rad + 0.010)[0]
                for h in hit:
                    if dn[h] > 1e-6:
                        n = dd[h] / dn[h]
                        Pa[i] += n * (rad + 0.010 - dn[h])
                        vn = np.dot(Va[i], n)
                        if vn < 0:
                            Va[i] -= 1.4 * vn * n
                        Va[i] *= 0.86
            # side walls
            Pa[:, 0] = np.clip(Pa[:, 0], -0.40, 0.40)
            # floor: settle
            below = Pa[:, 1] < -0.40
            Pa[below, 1] = -0.40; Va[below] *= 0.0
        P = Pa.tolist(); V = Va.tolist()
        frames.append(_draw_plinko(Pa, np.array(cols[:len(Pa)]), pegs, rad))
    return frames


def _draw_plinko(P, cols, pegs, rad):
    im = Image.new("RGB", (OUT_W, OUT_H), (8, 10, 18))
    d = ImageDraw.Draw(im)
    for (px, py) in pegs:
        x, y = _to_px(np.array([px, py])); d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(90, 110, 150))
    rpx = rad * OUT_H * SCALE
    for i in range(len(P)):
        x, y = _to_px(P[i]); c = tuple(int(v) for v in cols[i])
        d.ellipse([x - rpx, y - rpx, x + rpx, y + rpx], fill=c)
    bloom = im.filter(ImageFilter.GaussianBlur(4))
    return np.clip(np.asarray(im, np.float32) + 0.35 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hexagon", choices=["hexagon", "plinko"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    frames = (sim_hexagon if a.scene == "hexagon" else sim_plinko)(a.seconds, a.fps, a.seed)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for f in frames:
            w.append_data(f)
    finally:
        w.close()
    print(f"[OK] physics '{a.scene}' {len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
