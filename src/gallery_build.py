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


def kb_clip(path: str, per: float, z0: float = 1.0, z1: float = 1.08):
    """Smooth Ken Burns: subpixel zoom via PIL AFFINE+BILINEAR (no integer-resize jitter)."""
    base = Image.open(path).convert("RGB")
    iw, ih = base.size
    cover = max(W / iw, H / ih)
    # pre-upscale once with headroom so the zoom never exceeds source res
    bw, bh = int(iw * cover * (z1 + 0.02)), int(ih * cover * (z1 + 0.02))
    big = base.resize((bw, bh), Image.LANCZOS)

    def make(t):
        z = z0 + (z1 - z0) * (t / per)          # smooth continuous zoom factor
        cw, ch = W / z, H / z                    # source window (subpixel)
        cx, cy = bw / 2.0, bh / 2.0
        a, e = cw / W, ch / H
        c, f = cx - cw / 2.0, cy - ch / 2.0
        frame = big.transform((W, H), Image.AFFINE, (a, 0, c, 0, e, f), resample=Image.BILINEAR)
        return np.asarray(frame)

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
        c = kb_clip(p, a.per)
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
