"""Diffusion-Limited Aggregation (DLA) mp4 (#020). 樹枝状成長 (稲妻/珊瑚/霜/川 と同じ機構)。

機構: 中心に種。粒子が遠くからランダムウォークして来て、クラスタに触れたらその場で固着。
これを繰り返すと枝分かれの樹枝(フラクタル)が育つ。色は到着順のグラデで美しく & 見やすく。

modes:
  walk  — 1 粒のランダムウォークを見せて「触れたら止まる」を説明 (見方ガイド用、ゆっくり)
  grow  — クラスタが枝を伸ばして育つ (到着順グラデ)
  art   — 育ち切った樹枝を発光配色でゆっくり (Beauty Beat)

使い方:
  python dla_simulator.py --mode grow --output x.mp4 --duration 18
"""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

BG = (4, 6, 13)
NEI = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _grad(t):
    """到着順 t(0..1) → cyan→azure→violet の落ち着いたグラデ (gentle, F9-4)。"""
    r = int(70 + 130 * t)
    g = int(210 - 120 * t)
    b = 255
    return (r, g, b)


def simulate(n_particles, N, seed=7, record_every=1):
    rng = np.random.default_rng(seed)
    occ = np.zeros((N, N), dtype=bool)
    order = np.full((N, N), -1, dtype=np.int32)
    c = N // 2
    occ[c, c] = True; order[c, c] = 0
    mr = 1.0          # 実際の最大固着半径 (クラスタ extent)
    maxR = N / 2 - 4
    count = 1
    snaps = []
    def release():
        rad = min(mr + 12, maxR)            # 放出半径 = クラスタ extent + 12 (glom 防止)
        ang = rng.uniform(0, 2 * np.pi)
        x = int(round(c + rad * np.cos(ang)))
        y = int(round(c + rad * np.sin(ang)))
        return min(max(x, 1), N - 2), min(max(y, 1), N - 2)

    for p in range(1, n_particles + 1):
        x, y = release()
        entered = False   # 放出円 glom 防止: クラスタ extent 近く(mr+2)まで侵入してから固着可
        kill = min(mr + 24, maxR + 18)
        for step in range(N * 12):
            d = np.hypot(x - c, y - c)
            if not entered and d <= mr + 2:
                entered = True
            stuck = False
            if entered:
                for dx, dy in NEI:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < N and 0 <= ny < N and occ[nx, ny]:
                        stuck = True; break
            if stuck:
                occ[x, y] = True; order[x, y] = count; count += 1
                rr = np.hypot(x - c, y - c)
                if rr > mr: mr = rr
                break
            dx, dy = NEI[rng.integers(0, 4)]
            x += dx; y += dy
            if x < 1 or x >= N - 1 or y < 1 or y >= N - 1 or np.hypot(x - c, y - c) > kill:
                x, y = release()
        if p % record_every == 0:
            snaps.append(order.copy())
    snaps.append(order.copy())
    return snaps, count


def _img(order, w, h, total, glow=False):
    N = order.shape[0]
    rgb = np.zeros((N, N, 3), dtype=np.uint8); rgb[:] = BG
    ys, xs = np.where(order >= 0)
    for x, y in zip(ys, xs):
        t = order[x, y] / max(1, total)
        rgb[x, y] = _grad(t)
    im = Image.fromarray(rgb, "RGB")
    if glow:
        from PIL import ImageFilter
        base = im.filter(ImageFilter.GaussianBlur(2))
        im = Image.blend(base, im, 0.6)
    side = min(w, h)
    im = im.resize((side, side), Image.NEAREST if not glow else Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), BG); canvas.paste(im, ((w - side) // 2, (h - side) // 2))
    return np.asarray(canvas)


def render(mode, output, duration, fps, w, h):
    steps = max(2, int(duration * fps))
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)

    if mode in ("grow", "art"):
        N = 261 if mode == "grow" else 281
        target = 3800 if mode == "grow" else 4000
        every = max(1, target // steps)
        snaps, total = simulate(target, N, record_every=every)
        # snaps を steps に合わせて間引き/詰め
        if mode == "art":
            # 育ち切りを発光でゆっくり (最後の数 snap を hold)
            final = snaps[-1]
            for i in range(steps):
                wr.append_data(_img(final, w, h, total, glow=True))
        else:
            idx = np.linspace(0, len(snaps) - 1, steps).astype(int)
            for i in idx:
                wr.append_data(_img(snaps[i], w, h, total))

    elif mode == "walk":
        # 1 粒の歩行を見せる: 種は中央の小さな塊、walker が来て触れて止まる
        N = 121; c = N // 2
        rng = np.random.default_rng(2)
        occ = np.zeros((N, N), dtype=bool); occ[c, c] = True
        order = np.full((N, N), -1, dtype=np.int32); order[c, c] = 0
        cnt = 1
        # 数粒だけ先に固着させて小さな種に
        for _ in range(40):
            snaps, _ = simulate(1, N, seed=rng.integers(1, 999))
        # walker をゆっくり描く
        for f in range(steps):
            x = int(c + 45 * np.cos(f * 0.05)); y = int(c + 45 * np.sin(f * 0.13))
            rgb = np.zeros((N, N, 3), dtype=np.uint8); rgb[:] = BG
            rgb[c, c] = (180, 230, 255)
            if 0 <= x < N and 0 <= y < N:
                rgb[x, y] = (255, 200, 120)
            im = Image.fromarray(rgb).resize((min(w, h), min(w, h)), Image.NEAREST)
            canvas = Image.new("RGB", (w, h), BG); canvas.paste(im, ((w - min(w, h)) // 2, 0))
            wr.append_data(np.asarray(canvas))

    wr.close()
    print(f"[OK] {mode} -> {out} ({w}x{h})")


def render_still(output, w, h, seed=7):
    """育ち切った樹枝(十分成長した DLA)を発光配色で 1 枚の PNG に。
    walk は pure-Python で重いので still 用に grid/粒子数を抑える(枝形は十分出る)。"""
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    N = 121; target = 320
    snaps, total = simulate(target, N, seed=seed, record_every=10 ** 9)
    final = snaps[-1]
    arr = _img(final, w, h, total, glow=True)
    Image.fromarray(arr).save(output)
    print(f"[OK] still -> {out} ({w}x{h}, {total} particles)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grow", choices=["walk", "grow", "art", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.mode == "still":
        render_still(a.output, a.width, a.height, a.seed)
    else:
        render(a.mode, a.output, a.duration, a.fps, a.width, a.height)


if __name__ == "__main__":
    main()
