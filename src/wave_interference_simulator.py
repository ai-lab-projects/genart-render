"""Wave interference mp4 (#017). 点波源の重ね合わせ → 干渉縞。

物理: 各源は円形波 sin(k*r - ω*t) を放つ。複数源の振幅を足す (superposition)。
crest が重なれば強め合い (constructive)、crest と trough が出会えば打ち消し (destructive)。
二源 = 二重スリット (Huygens) と同じ干渉パターン。

modes:
  one     — 単一源 (同心円の基準)
  two     — 二源干渉 (双曲線状の縞 = 主役、double-slit と同型)
  many    — 多源 (モアレ的な複雑パターン)
  art     — Beauty Beat 用: 二源をゆっくり動かし、発光配色で「呼吸する干渉」

使い方:
  python wave_interference_simulator.py --mode two --output x.mp4 --duration 18
"""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

W, H = 1280, 720


def _sources(mode, t_frac):
    """各 mode の波源座標 (データ座標 0..1, アスペクト考慮で x は 0..16/9)。"""
    ax = W / H  # ~1.778
    cx = ax / 2
    if mode == "one":
        return [(cx, 0.5)]
    if mode in ("two", "art"):
        sep = 0.30
        if mode == "art":
            sep = 0.24 + 0.10 * np.sin(t_frac * 2 * np.pi)   # ゆっくり呼吸
        return [(cx - sep, 0.5), (cx + sep, 0.5)]
    if mode == "many":
        return [(cx + 0.55 * np.cos(a), 0.5 + 0.55 * np.sin(a) * 0.55)
                for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)]
    return [(cx, 0.5)]


def _field(srcs, t, k, omega):
    ax = W / H
    xs = np.linspace(0, ax, W, dtype=np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    f = np.zeros((H, W), dtype=np.float32)
    for (sx, sy) in srcs:
        r = np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2) + 1e-3
        f += np.sin(k * r - omega * t) / np.sqrt(r + 0.06)   # 1/√r 減衰
    return f / len(srcs)


def _colorize(f):
    """振幅場 → ダーク基調の発光配色。crest=cyan→white、trough=deep blue、node≈黒。"""
    a = np.clip(f / 0.9, -1, 1)
    pos = np.clip(a, 0, 1)        # crest
    neg = np.clip(-a, 0, 1)       # trough
    r = 60 * neg + 200 * pos ** 1.5
    g = 90 * neg + 230 * pos ** 1.2
    b = 150 * neg + 255 * pos
    img = np.stack([r, g, b], -1)
    # base な暗い navy を下敷きに
    img += np.array([4, 6, 13], dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def render_still(args):
    """最も美しい状態を PNG 1 枚で保存。二源干渉の双曲線縞 (主役パターン)。"""
    from PIL import Image
    k = 70.0
    omega = 6.0
    # 二源を固定配置で。crest がよく出る位相 t を選ぶ。
    srcs = _sources("two", 0.0)
    f = _field(srcs, 1.0, k, omega)
    img = _colorize(f)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGB").save(str(out))
    print(f"[OK] still -> {out} ({W}x{H})")


def render(mode, output, duration, fps):
    steps = max(2, int(duration * fps))
    k = 70.0          # 波数 (縞の細かさ)
    omega = 6.0       # 角周波数 (伝播速度)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for i in range(steps):
        t = i / fps
        tf = i / steps
        srcs = _sources(mode, tf)
        f = _field(srcs, t, k, omega)
        wr.append_data(_colorize(f))
    wr.close()
    print(f"[OK] {mode} -> {out} ({steps}f, {W}x{H})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="two", choices=["one", "two", "many", "art", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=16.0)
    ap.add_argument("--fps", type=int, default=24)
    a = ap.parse_args()
    if a.mode == "still":
        render_still(a)
        return
    render(a.mode, a.output, a.duration, a.fps)


if __name__ == "__main__":
    main()
