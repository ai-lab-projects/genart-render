"""Apollonian gasket mp4 generator (#013). Descartes circle theorem, recursive packing.

modes:
  fill — circles revealed largest-first (watch the gaps fill forever)
  zoom — full gasket, slow zoom into one gap (infinite detail)

Descartes (Vieta form, no sqrt): the 4th circle tangent to 3 mutually tangent ones,
other than a known 4th, has curvature k4' = 2(k1+k2+k3) - k4 and
k4'*z4' = 2(k1 z1 + k2 z2 + k3 z3) - k4 z4  (complex centers).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import colorsys

BG = (4, 6, 13)


class C:
    __slots__ = ("z", "k")
    def __init__(self, z, k):
        self.z = complex(z); self.k = float(k)
    @property
    def r(self):
        return abs(1.0 / self.k) if self.k != 0 else 0.0


def _other(a, b, c, d):
    k = 2 * (a.k + b.k + c.k) - d.k
    if abs(k) < 1e-12:
        return None
    kz = 2 * (a.k * a.z + b.k * b.z + c.k * c.z) - d.k * d.z
    return C(kz / k, k)


def gasket(min_r=0.004, max_circles=900):
    # start: outer (-1) + two (2) + top/bottom (3) → the (-1,2,2,3) gasket
    c1 = C(0 + 0j, -1.0)
    c2 = C(-0.5 + 0j, 2.0)
    c3 = C(0.5 + 0j, 2.0)
    c4 = C(0 + (2.0 / 3.0) * 1j, 3.0)
    c4b = C(0 - (2.0 / 3.0) * 1j, 3.0)
    circles = [c1, c2, c3, c4, c4b]

    def rec(a, b, c, d, depth):
        if len(circles) >= max_circles or depth <= 0:
            return
        n = _other(a, b, c, d)
        if n is None or n.r < min_r or n.k <= 0:
            return
        circles.append(n)
        rec(a, b, n, c, depth - 1)
        rec(a, c, n, b, depth - 1)
        rec(b, c, n, a, depth - 1)

    for d in (c4, c4b):
        rec(c1, c2, c3, d, 14)
    return circles


def _palette(n):
    out = []
    for i in range(n):
        h = 0.58 + 0.42 * (i / max(1, n - 1))  # cyan→magenta-ish sweep
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.55, 0.98)
        out.append((int(255 * r), int(255 * g), int(255 * b)))
    return out


def render(mode, output, duration, fps, w, h, min_r):
    circles = gasket(min_r=min_r)
    circles_sorted = sorted(circles, key=lambda c: c.r, reverse=True)
    pal = _palette(len(circles_sorted))
    frames = max(1, int(duration * fps))
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    lim = 1.05
    SS = 2  # supersample for smooth (anti-aliased) circle outlines
    W, H = w * SS, h * SS

    for f in range(frames):
        frac = (f + 1) / frames
        if mode == "zoom":
            n = len(circles_sorted)
            z = 1.0 + 2.5 * frac
            cx, cy, hw = 0.0, 0.45, lim / z
        else:  # fill
            n = max(4, int(frac * len(circles_sorted)))
            cx, cy, hw = 0.0, 0.0, lim
        ppu = H / (2 * hw)  # pixels per unit (square aspect, fit to height)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for i, c in enumerate(circles_sorted[:n]):
            xp = W / 2 + (c.z.real - cx) * ppu
            yp = H / 2 - (c.z.imag - cy) * ppu
            rp = c.r * ppu
            if rp < 0.7 or xp + rp < 0 or xp - rp > W or yp + rp < 0 or yp - rp > H:
                continue
            col = (220, 230, 247) if c.k < 0 else pal[i]
            lw = max(SS, int(rp * 0.05) + SS) if c.k < 0 else max(1, min(3 * SS, int(rp * 0.05) + SS))
            d.ellipse([xp - rp, yp - rp, xp + rp, yp + rp], outline=col, width=lw)
        wr.append_data(np.asarray(img.resize((w, h), Image.LANCZOS)))
    wr.close()
    print(f"[OK] {mode} -> {out} ({frames}f @ {fps}fps, {w}x{h}, {len(circles)} circles)")


def render_still(output, w, h, min_r):
    """完成したガスケット(円が十分詰まった最終状態)を 1 枚の PNG に。"""
    circles = gasket(min_r=min_r)
    circles_sorted = sorted(circles, key=lambda c: c.r, reverse=True)
    pal = _palette(len(circles_sorted))
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    lim = 1.05
    SS = 2  # supersample for smooth (anti-aliased) circle outlines
    W, H = w * SS, h * SS
    cx, cy, hw = 0.0, 0.0, lim
    ppu = H / (2 * hw)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for i, c in enumerate(circles_sorted):
        xp = W / 2 + (c.z.real - cx) * ppu
        yp = H / 2 - (c.z.imag - cy) * ppu
        rp = c.r * ppu
        if rp < 0.7 or xp + rp < 0 or xp - rp > W or yp + rp < 0 or yp - rp > H:
            continue
        col = (220, 230, 247) if c.k < 0 else pal[i]
        lw = max(SS, int(rp * 0.05) + SS) if c.k < 0 else max(1, min(3 * SS, int(rp * 0.05) + SS))
        d.ellipse([xp - rp, yp - rp, xp + rp, yp + rp], outline=col, width=lw)
    img.resize((w, h), Image.LANCZOS).save(output)
    print(f"[OK] still -> {out} ({w}x{h}, {len(circles)} circles)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fill", choices=["fill", "zoom", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--min-r", type=float, default=0.004)
    a = ap.parse_args()
    if a.mode == "still":
        render_still(a.output, a.width, a.height, a.min_r)
    else:
        render(a.mode, a.output, a.duration, a.fps, a.width, a.height, a.min_r)


if __name__ == "__main__":
    main()
