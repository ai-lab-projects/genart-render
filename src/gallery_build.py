"""Build an ambient gallery slideshow: generative-art stills with slow Ken Burns zoom + crossfade.
Motion (zoom + crossfade) keeps it 'produced' (not a static dump). Music muxed separately via ffmpeg.

Usage: python gallery_build.py <stills_dir> <out_mp4> [--per 9 --xfade 1.6 --loops 8]
"""
from __future__ import annotations
import sys, glob, argparse
import numpy as np
from PIL import Image
from moviepy import VideoClip, concatenate_videoclips, vfx

W, H = 1280, 720
BG = (4, 6, 13)                                   # ambient dark背景(#04060d)
MOTIONS = ["zoom_in", "zoom_out", "pan_r", "pan_l", "pan_u", "pan_d"]


def _fit_canvas(base: Image.Image) -> Image.Image:
    """被写体をbboxで切り出し→16:9暗キャンバスにcontainフィット(中央)。
    小さい被写体(dla等)は拡大され, 正方形(apollonian等)は上下を切らずに収まる。背景は暗いので継ぎ目なし。"""
    arr = np.asarray(base)
    lum = arr.max(axis=2)
    ys, xs = np.where(lum > 26)                    # 背景(#04060d≈lum13)より明るい=被写体
    if len(xs) > 80:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        pad = int(0.05 * max(x1 - x0, y1 - y0))    # 少し余白
        base = base.crop((max(x0 - pad, 0), max(y0 - pad, 0),
                          min(x1 + pad, base.width), min(y1 + pad, base.height)))
    iw, ih = base.size
    scale = min(W * 0.86 / iw, H * 0.86 / ih)      # 86%でcontain(緩い動きでも被写体が切れないよう余白確保)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    img = base.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(img, ((W - nw) // 2, (H - nh) // 2))
    return canvas


def kb_clip(path: str, per: float, motion: str = "zoom_in"):
    """Ken Burns: containフィット後に動き(ズームin/out・上下左右パン)を付ける。被写体は切れない。"""
    canvas = _fit_canvas(Image.open(path).convert("RGB"))
    HEAD = 1.10                                    # パン/ズーム用のヘッドルーム(緩め)
    bw, bh = int(W * HEAD), int(H * HEAD)
    big = canvas.resize((bw, bh), Image.LANCZOS)

    def make(t):
        u = t / per
        z, cx, cy = 1.03, bw / 2.0, bh / 2.0
        if motion == "zoom_in":
            z = 1.0 + 0.05 * u                     # 緩いズーム(最大5%=86%被写体の余白内→切れない)
        elif motion == "zoom_out":
            z = 1.05 - 0.05 * u
        else:                                      # pan系: 中ズーム固定で窓をゆっくり平行移動(可動域の6割)
            z = 1.04
            cw, ch = W / z, H / z
            mx, my = (bw - cw) / 2.0 * 0.6, (bh - ch) / 2.0 * 0.6
            if motion == "pan_r":   cx = bw / 2 + (u - 0.5) * 2 * mx
            elif motion == "pan_l": cx = bw / 2 - (u - 0.5) * 2 * mx
            elif motion == "pan_d": cy = bh / 2 + (u - 0.5) * 2 * my
            elif motion == "pan_u": cy = bh / 2 - (u - 0.5) * 2 * my
        cw, ch = W / z, H / z
        c = min(max(cx - cw / 2.0, 0.0), bw - cw)  # 窓がbig内に収まるようclamp
        f = min(max(cy - ch / 2.0, 0.0), bh - ch)
        a, e = cw / W, ch / H
        return np.asarray(big.transform((W, H), Image.AFFINE, (a, 0, c, 0, e, f), resample=Image.BILINEAR))

    return VideoClip(make, duration=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stills_dir"); ap.add_argument("out")
    ap.add_argument("--per", type=float, default=9.0)
    ap.add_argument("--xfade", type=float, default=1.6)
    ap.add_argument("--loops", type=int, default=1)
    a = ap.parse_args()
    paths = sorted(glob.glob(f"{a.stills_dir}/*.png"))
    if not paths:
        raise SystemExit("no stills")
    seq = paths * a.loops
    clips = []
    for i, p in enumerate(seq):
        # ファイル名(s00_zoom_in.png)から画像ごとの最適な動きを読む。無ければローテにフォールバック。
        import os as _os
        stem = _os.path.splitext(_os.path.basename(p))[0]
        mo = next((m for m in MOTIONS if stem.endswith(m)), MOTIONS[i % len(MOTIONS)])
        c = kb_clip(p, a.per, motion=mo)
        if i > 0:
            c = c.with_effects([vfx.CrossFadeIn(a.xfade)])
        clips.append(c)
    montage = concatenate_videoclips(clips, method="compose", padding=-a.xfade, bg_color=(4, 6, 13))
    montage = montage.resized((W, H))
    montage.write_videofile(a.out, fps=24, codec="libx264", audio=False,
                            ffmpeg_params=["-pix_fmt", "yuv420p"], preset="veryfast", bitrate="2500k")
    print(f"[OK] gallery {len(seq)} cuts -> {a.out} ({montage.duration:.1f}s)")


if __name__ == "__main__":
    main()
