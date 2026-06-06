"""RL maze solver (tabular Q-learning) — watch the value function BLOOM from the goal and the
agent find the shortest path. Self-implemented (~Q-learning), VM-friendly, visually shows learning.

The 'wow' is the value heatmap: each cell glows by its learned value V=max_a Q. Early on only the
goal glows; as training proceeds the glow spreads backward along the maze and the greedy path
straightens — you literally watch the agent figure it out.

Modes:
  python maze_rl_simulator.py --mode loop --output out.mp4 --seconds 18
  python maze_rl_simulator.py --mode still --output out.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import imageio.v2 as imageio

OUT_W, OUT_H = 1280, 720
FPS = 24
MW, MH = 31, 17                      # maze grid (odd = walls between cells)


def gen_maze(rng):
    """Recursive-backtracker maze. grid[y,x]=1 wall, 0 path. Cells at odd coords."""
    g = np.ones((MH, MW), np.uint8)
    def carve(cx, cy):
        g[cy, cx] = 0
        for dx, dy in rng.permutation([(2, 0), (-2, 0), (0, 2), (0, -2)]):
            nx, ny = cx + dx, cy + dy
            if 0 < nx < MW - 1 and 0 < ny < MH - 1 and g[ny, nx] == 1:
                g[cy + dy // 2, cx + dx // 2] = 0
                carve(nx, ny)
    import sys; sys.setrecursionlimit(10000)
    carve(1, 1)
    return g


ACTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def train_episodes(g, Q, start, goal, rng, n, eps):
    H, W = g.shape
    for _ in range(n):
        x, y = start
        for _step in range(4 * W * H):
            s = y * W + x
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
            dx, dy = ACTS[a]; nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H) or g[ny, nx] == 1:
                nx, ny = x, y                       # bump wall -> stay
            r = 1.0 if (nx, ny) == goal else -0.01
            ns = ny * W + nx
            Q[s, a] += 0.5 * (r + 0.95 * Q[ns].max() - Q[s, a])
            x, y = nx, ny
            if (x, y) == goal:
                break


def _render(g, Q, start, goal):
    H, W = g.shape
    cw = OUT_W / W; ch = OUT_H / H
    V = Q.max(axis=1).reshape(H, W)
    Vn = V.copy(); Vn[g == 1] = np.nan
    lo = np.nanmin(Vn) if np.isfinite(np.nanmin(Vn)) else 0
    hi = np.nanmax(Vn) if np.isfinite(np.nanmax(Vn)) else 1
    Vn = (Vn - lo) / (hi - lo + 1e-6)
    img = Image.new("RGB", (OUT_W, OUT_H), (6, 8, 16))
    d = ImageDraw.Draw(img)
    for y in range(H):
        for x in range(W):
            x0, y0, x1, y1 = x * cw, y * ch, (x + 1) * cw, (y + 1) * ch
            if g[y, x] == 1:
                d.rectangle([x0, y0, x1, y1], fill=(14, 18, 34))     # wall
            else:
                v = 0.0 if np.isnan(Vn[y, x]) else float(Vn[y, x])
                c = (int(20 + 30 * v), int(40 + 180 * v), int(70 + 150 * v))   # value glow teal->bright
                d.rectangle([x0, y0, x1, y1], fill=c)
    # greedy path from start
    x, y = start; path = [(x, y)]
    for _ in range(W * H):
        s = y * W + x
        if V[y, x] <= -1e8: break
        a = int(np.argmax(Q[s])); dx, dy = ACTS[a]; nx, ny = x + dx, y + dy
        if not (0 <= nx < W and 0 <= ny < H) or g[ny, nx] == 1 or (nx, ny) in path: break
        path.append((nx, ny)); x, y = nx, ny
        if (x, y) == goal: break
    pts = [((px + 0.5) * cw, (py + 0.5) * ch) for px, py in path]
    if len(pts) > 1:
        d.line(pts, fill=(255, 240, 180), width=max(2, int(cw * 0.18)), joint="curve")
    for (cx, cy), col, rr in [(start, (120, 200, 255), 0.32), (goal, (255, 120, 120), 0.36)]:
        d.ellipse([(cx + .5) * cw - cw * rr, (cy + .5) * ch - ch * rr,
                   (cx + .5) * cw + cw * rr, (cy + .5) * ch + ch * rr], fill=col)
    # agent head at end of current path
    hx, hy = path[-1]
    d.ellipse([(hx + .5) * cw - cw * .22, (hy + .5) * ch - ch * .22,
               (hx + .5) * cw + cw * .22, (hy + .5) * ch + ch * .22], fill=(255, 255, 255))
    bloom = img.filter(ImageFilter.GaussianBlur(6))
    return np.clip(np.asarray(img, np.float32) + 0.35 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)


def _setup(seed):
    rng = np.random.default_rng(seed)
    g = gen_maze(rng)
    H, W = g.shape
    start = (1, 1); goal = (W - 2, H - 2)
    g[goal[1], goal[0]] = 0
    Q = np.zeros((H * W, 4), np.float32)
    return rng, g, Q, start, goal


def render_loop(args):
    rng, g, Q, start, goal = _setup(args.seed)
    frames = max(2, int(args.seconds * args.fps))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        eps = 0.3
        for i in range(frames):
            train_episodes(g, Q, start, goal, rng, n=6, eps=eps)
            eps = max(0.03, eps * 0.985)
            w.append_data(_render(g, Q, start, goal))
    finally:
        w.close()
    print(f"[OK] maze RL: Q-learning value bloom + greedy path -> {out}")


def render_still(args):
    rng, g, Q, start, goal = _setup(args.seed)
    train_episodes(g, Q, start, goal, rng, n=400, eps=0.2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(g, Q, start, goal)).save(args.output)
    print(f"[OK] maze RL still -> {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    (render_loop if args.mode == "loop" else render_still)(args)


if __name__ == "__main__":
    main()
