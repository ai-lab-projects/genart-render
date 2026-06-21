"""Modular times-table circle mp4 (#021). 円周上の n を i→(i*k mod N) で結ぶと曲線が浮かぶ。

k=2 でカージオイド(心臓形)、k=3 でネフロイド… 乗数 k を変えると別の曲線。
乗数を連続変化させると曲線がモーフィングする (F10: variant + 動く outro)。

modes:
  cardioid — k=2、線が増えてカージオイドが現れる (progressive)
  nephroid — k=3 の曲線 (variant)
  petals   — k=5 など (variant 2)
  morph    — k を 2→k_max まで連続変化、曲線がモーフ (動的 outro / Beauty Beat)

使い方:
  python times_table_simulator.py --mode morph --output x.mp4 --duration 18
"""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

BG = (4, 6, 13)


def _points(N, w, h, margin=0.86):
    cx, cy = w / 2, h / 2
    R = min(w, h) / 2 * margin
    ang = -np.pi / 2 + np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.stack([cx + R * np.cos(ang), cy + R * np.sin(ang)], 1)


def _grad(t):
    # cyan -> azure -> violet
    return (int(80 + 130 * t), int(210 - 120 * t), 255)


def _draw(N, k, w, h, frac=1.0, SS=2):
    W, H = w * SS, h * SS
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    pts = _points(N, W, H)
    m = int(N * frac)
    for i in range(m):
        j = int((i * k) % N)
        col = _grad(i / max(1, N))
        d.line([tuple(pts[i]), tuple(pts[j])], fill=col, width=SS)
    return img.resize((w, h), Image.LANCZOS)


def render_still(args):
    """最も美しい状態を PNG 1 枚で保存。倍率 51 の花/曼荼羅模様 (完成形を full で描画)。"""
    N = 300
    k = args.mult                       # 51 = 多弁の花。2 = 心臓形 (cardioid)。
    img = _draw(N, k, args.width, args.height, frac=1.0)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    print(f"[OK] still (k={k}) -> {out} ({args.width}x{args.height})")


def render(mode, output, duration, fps, w, h):
    steps = max(2, int(duration * fps))
    N = 300
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)

    if mode in ("cardioid", "nephroid", "petals"):
        k = {"cardioid": 2, "nephroid": 3, "petals": 5}[mode]
        # 線を progressive に増やす → 曲線が現れる。後半は完成形を hold。
        grow = int(steps * 0.7)
        for i in range(steps):
            frac = min(1.0, (i + 1) / grow)
            wr.append_data(np.asarray(_draw(N, k, w, h, frac=frac)))

    elif mode == "morph":
        # k を連続変化 (2 → 50) → 曲線がモーフ。動的 (F10-4)。
        kmin, kmax = 2.0, 50.0
        for i in range(steps):
            f = i / (steps - 1)
            k = kmin + (kmax - kmin) * f
            wr.append_data(np.asarray(_draw(N, k, w, h, frac=1.0)))

    wr.close()
    print(f"[OK] {mode} -> {out} ({w}x{h})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cardioid", choices=["cardioid", "nephroid", "petals", "morph", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--mult", type=float, default=51.0, help="still モードの倍率 k (51=花, 2=心臓形)")
    a = ap.parse_args()
    if a.mode == "still":
        render_still(a)
        return
    render(a.mode, a.output, a.duration, a.fps, a.width, a.height)


if __name__ == "__main__":
    main()
