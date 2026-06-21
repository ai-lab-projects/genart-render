"""Abelian sandpile / self-organized criticality mp4 (#019).

ルール: 各セルが砂粒を持つ。4 以上になったら「崩れて」4 粒を失い、上下左右に 1 粒ずつ渡す。
これを安定するまで繰り返す (abelian: 崩す順序によらず最終形は同じ)。

modes:
  grow      — 中央に砂を足し続ける → 自己相似なマンダラ模様が育つ
  avalanche — 臨界状態 (ほぼ満杯) に 1 粒ずつ落とす → 崩れ (なだれ) が大小さまざまに広がる (SOC)
  art       — 大きなマンダラを発光配色でゆっくり (Beauty Beat)

使い方:
  python sandpile_simulator.py --mode grow --output x.mp4 --duration 16
"""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

# 高さ 0,1,2,3 の配色 (dark navy → cyan → 白)
PALETTE = np.array([
    [4, 6, 13],       # 0
    [30, 70, 130],    # 1
    [70, 150, 220],   # 2
    [200, 235, 255],  # 3
], dtype=np.uint8)

PALETTE_WARM = np.array([
    [6, 6, 14],
    [40, 60, 120],
    [120, 110, 220],
    [255, 220, 150],
], dtype=np.uint8)


def stabilize(g):
    """4 以上のセルを一斉に崩す (vectorized, abelian)。安定するまで。"""
    while True:
        over = g >= 4
        if not over.any():
            return
        t = over.astype(np.int32)
        g -= 4 * t
        g[1:, :] += t[:-1, :]
        g[:-1, :] += t[1:, :]
        g[:, 1:] += t[:, :-1]
        g[:, :-1] += t[:, 1:]


def topple_once(g):
    """1 ステップだけ崩す (なだれの伝播を可視化用)。崩れたマスク返す。"""
    over = g >= 4
    if not over.any():
        return None
    t = over.astype(np.int32)
    g -= 4 * t
    g[1:, :] += t[:-1, :]; g[:-1, :] += t[1:, :]
    g[:, 1:] += t[:, :-1]; g[:, :-1] += t[:, 1:]
    return over


def _img(g, w, h, pal=PALETTE, flash=None):
    """正方グリッドを引き伸ばさず、ダーク 16:9 キャンバスに中央配置 (mandala が円形を保つ)。"""
    from PIL import Image
    rgb = pal[np.clip(g, 0, 3)]
    if flash is not None:
        rgb = rgb.copy()
        rgb[flash] = [255, 180, 90]
    side = min(w, h)
    im = Image.fromarray(rgb, "RGB").resize((side, side), Image.NEAREST)
    canvas = Image.new("RGB", (w, h), tuple(int(x) for x in pal[0]))
    canvas.paste(im, ((w - side) // 2, (h - side) // 2))
    return np.asarray(canvas)


def render_still(args):
    """最も美しい最終状態を PNG 1 枚で保存。
    art と同じ大きなマンダラ (warm 配色) を 1 発で安定化させた完成形。"""
    from PIL import Image
    N = 333
    g = np.zeros((N, N), dtype=np.int32)
    c = N // 2
    g[c, c] += 90000          # art の総量を一度に投入
    stabilize(g)
    frame = _img(g, args.width, args.height, pal=PALETTE_WARM)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame, "RGB").save(str(out))
    print(f"[OK] still -> {out} ({args.width}x{args.height})")


def render(mode, output, duration, fps, w, h):
    steps = max(2, int(duration * fps))
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)

    if mode in ("grow", "art"):
        N = 333
        g = np.zeros((N, N), dtype=np.int32)
        c = N // 2
        total = 90000 if mode == "art" else 60000
        per = total // steps
        pal = PALETTE_WARM if mode == "art" else PALETTE
        for i in range(steps):
            g[c, c] += per
            stabilize(g)
            wr.append_data(_img(g, w, h, pal=pal))

    elif mode == "avalanche":
        # 臨界状態: 一様に 3 (崩れる一歩手前)。1 粒落とすたびに「なだれ」が大小さまざまに広がる。
        N = 149
        rng = np.random.default_rng(3)
        # 臨界に近い質感のある場: ランダム 2-3 → 安定化 (一様な真っ白を避ける)
        g = rng.integers(2, 4, size=(N, N)).astype(np.int32)
        stabilize(g)
        for i in range(steps):
            x, y = int(rng.integers(0, N)), int(rng.integers(0, N))
            g[x, y] += 1
            # この 1 粒で崩れた全セルを集計 (= なだれの範囲) してフラッシュ
            toppled = np.zeros((N, N), dtype=bool)
            for _ in range(2000):
                flash = topple_once(g)
                if flash is None:
                    break
                toppled |= flash
            wr.append_data(_img(g, w, h, flash=toppled if toppled.any() else None))

    wr.close()
    print(f"[OK] {mode} -> {out} ({w}x{h})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grow", choices=["grow", "avalanche", "art", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    if a.mode == "still":
        render_still(a)
        return
    render(a.mode, a.output, a.duration, a.fps, a.width, a.height)


if __name__ == "__main__":
    main()
