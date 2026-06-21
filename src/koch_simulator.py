"""Koch curve / snowflake mp4 (#018). 海岸線パラドックスの数学モデル。

各反復で 1 本の線分を 4 本 (中央に 60度の山) に置換 → 長さ ×4/3 が無限に増えるが
図形は有界。modes:
  build     — 直線から Koch 曲線へ、反復 0..N を 1 段ずつ
  snowflake — 正三角形 → Koch 雪片 (反復 0..N)
  zoom      — 縁に zoom-in して自己相似 (どの拡大率でも同じギザギザ)
  art       — Beauty Beat: 雪片を発光ストロークでゆっくり回転/呼吸

使い方:
  python koch_simulator.py --mode snowflake --output x.mp4 --duration 16
"""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

BG = (4, 6, 13)
CYAN = (120, 210, 255)


def koch_iter(points: np.ndarray) -> np.ndarray:
    """各線分を 4 本に置換した点列を返す。points: (n,2) 開いた折れ線。"""
    out = []
    for i in range(len(points) - 1):
        p1 = points[i]; p2 = points[i + 1]
        d = p2 - p1
        a = p1 + d / 3
        b = p1 + 2 * d / 3
        # 山の頂点: a→b を 60度回転
        ang = -np.pi / 3  # 上向き (画面 y 下向きなので符号調整)
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        peak = a + rot @ (b - a)
        out.extend([p1, a, peak, b])
    out.append(points[-1])
    return np.array(out)


def koch_levels(base: np.ndarray, n: int):
    levels = [base]
    for _ in range(n):
        levels.append(koch_iter(levels[-1]))
    return levels


def _base(mode):
    if mode == "snowflake" or mode == "art":
        # 正三角形 (時計回り閉路)
        cx, cy, r = 0.5, 0.56, 0.36
        pts = []
        for k in range(3):
            a = -np.pi / 2 + k * 2 * np.pi / 3
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
        pts.append(pts[0])
        return np.array(pts)
    # build / zoom: 横一直線
    return np.array([[0.08, 0.5], [0.92, 0.5]])


def _draw(points, w, h, col=CYAN, lw=2, glow=False):
    SS = 2; W, H = w * SS, h * SS
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    px = [(float(x) * W, float(y) * H) for x, y in points]
    if glow:
        for width, c in [(lw*6*SS,(20,60,110)),(lw*3*SS,(50,120,200)),(lw*SS,(180,230,255))]:
            d.line(px, fill=c, width=max(1,width), joint="curve")
    else:
        d.line(px, fill=col, width=lw * SS, joint="curve")
    return img.resize((w, h), Image.LANCZOS)


def render(mode, output, duration, fps, w, h):
    steps = max(2, int(duration * fps))
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    base = _base(mode)
    N = 5

    if mode in ("build", "snowflake"):
        levels = koch_levels(base, N)
        # 各 level を hold しつつ、次 level へ morph（簡易: level を一定フレーム表示）
        hold = steps // (N + 1)
        for lv in range(N + 1):
            pts = levels[lv]
            for _ in range(hold):
                wr.append_data(np.asarray(_draw(pts, w, h, glow=(mode == "snowflake"))))
        # 余り
        for _ in range(steps - hold * (N + 1)):
            wr.append_data(np.asarray(_draw(levels[N], w, h, glow=(mode == "snowflake"))))

    elif mode == "zoom":
        levels = koch_levels(base, 6)
        pts = levels[-1]
        for i in range(steps):
            f = i / steps
            scale = 1.0 + 7.0 * f               # 徐々に拡大
            cx = 0.5; cy = 0.5
            zp = (pts - [cx, cy]) * scale + [0.30, 0.5]
            wr.append_data(np.asarray(_draw(zp, w, h, lw=2)))

    elif mode == "art":
        levels = koch_levels(base, 4)
        pts = levels[-1]
        c = pts.mean(0)
        for i in range(steps):
            f = i / steps
            ang = 0.25 * np.sin(f * 2 * np.pi)          # ゆっくり回転
            breath = 1.0 + 0.04 * np.sin(f * 2 * np.pi)  # 呼吸
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            zp = (rot @ ((pts - c) * breath).T).T + c
            wr.append_data(np.asarray(_draw(zp, w, h, glow=True)))

    wr.close()
    print(f"[OK] {mode} -> {out} ({steps}f, {w}x{h})")


def render_still(output, w, h):
    """完成した Koch 雪片(反復 N)を発光ストロークで 1 枚の PNG に。"""
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    base = _base("snowflake")
    levels = koch_levels(base, 5)
    img = _draw(levels[-1], w, h, glow=True)
    img.save(output)
    print(f"[OK] still -> {out} ({w}x{h})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="snowflake", choices=["build", "snowflake", "zoom", "art", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    if a.mode == "still":
        render_still(a.output, a.width, a.height)
    else:
        render(a.mode, a.output, a.duration, a.fps, a.width, a.height)


if __name__ == "__main__":
    main()
