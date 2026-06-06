"""Mandelbulb — 3D fractal, distance-estimator ray marching (per pixel). Heavy (CI), self-implemented.

The Mandelbulb iterates z -> z^p + c in 3D (spherical power), with the standard analytic distance
estimate DE = 0.5*ln(r)*r/dr. We ray-march every pixel through the field, shade by surface normal +
ambient occlusion (march steps), and orbit the camera. Pure math (PD); numpy-vectorised over pixels.

  python mandelbulb.py --output out.mp4 --frames 120 --start 0 --count 0 --w 1280 --h 720 [--still]
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
from PIL import Image
import imageio.v2 as imageio

POWER = 8.0
ITERS = 8


def de(pos):
    """Mandelbulb distance estimate, vectorised over pos (P,3)."""
    z = pos.copy()
    dr = np.ones(len(pos), np.float32)
    r = np.zeros(len(pos), np.float32)
    for _ in range(ITERS):
        r = np.sqrt((z*z).sum(1))
        m = r < 2.0
        rs = np.maximum(r, 1e-9)
        theta = np.arccos(np.clip(z[:, 2]/rs, -1, 1))
        phi = np.arctan2(z[:, 1], z[:, 0])
        dr = np.where(m, (rs**(POWER-1))*POWER*dr + 1.0, dr)
        zr = rs**POWER
        th, ph = theta*POWER, phi*POWER
        st = np.sin(th)
        nz = np.stack([st*np.cos(ph), st*np.sin(ph), np.cos(th)], 1) * zr[:, None]
        z = np.where(m[:, None], nz + pos, z)
    return 0.5*np.log(np.maximum(r, 1e-9))*r/np.maximum(dr, 1e-9)


def render_frame(cam_ang, W, H, fov=0.5, max_steps=90, max_t=8.0, eps=0.0012):
    cam_d = 2.7
    inc = 0.5
    C = cam_d*np.array([math.sin(cam_ang)*math.cos(inc), math.sin(inc), math.cos(cam_ang)*math.cos(inc)], np.float32)
    fwd = -C/np.linalg.norm(C); up0 = np.array([0, 1, 0], np.float32)
    right = np.cross(fwd, up0); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    aspect = W/H
    xs = np.linspace(-1, 1, W)*math.tan(fov)*aspect
    ys = np.linspace(1, -1, H)*math.tan(fov)
    gx, gy = np.meshgrid(xs, ys)
    d = (fwd[None, None, :] + gx[..., None]*right + gy[..., None]*up).reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    P = W*H
    t = np.zeros(P, np.float32)
    hit = np.zeros(P, bool); steps_used = np.zeros(P, np.float32)
    alive = np.ones(P, bool)
    for s in range(max_steps):
        pos = C[None, :] + t[:, None]*d
        dist = de(pos)
        newhit = alive & (dist < eps)
        hit |= newhit
        steps_used = np.where(newhit, s, steps_used)
        alive = alive & ~newhit & (t < max_t)
        t = np.where(alive, t + np.maximum(dist, eps*0.5), t)
        if not alive.any():
            break
    # shade hits: normal via DE gradient
    img = np.zeros((P, 3), np.float32)
    if hit.any():
        ph = C[None, :] + t[hit, None]*d[hit]
        e = 0.0015
        def D(off): return de(ph+off)
        nx = D(np.array([e, 0, 0], np.float32)) - D(np.array([-e, 0, 0], np.float32))
        ny = D(np.array([0, e, 0], np.float32)) - D(np.array([0, -e, 0], np.float32))
        nz = D(np.array([0, 0, e], np.float32)) - D(np.array([0, 0, -e], np.float32))
        n = np.stack([nx, ny, nz], 1); n /= (np.linalg.norm(n, axis=1, keepdims=True)+1e-9)
        L = np.array([0.6, 0.7, 0.5], np.float32); L /= np.linalg.norm(L)
        diff = np.clip((n*L).sum(1), 0, 1)
        ao = 1.0 - steps_used[hit]/max_steps                 # fewer steps -> more exposed -> brighter
        # color by normal direction + radius for iridescence
        base = 0.5 + 0.5*n
        glow = (0.25 + 0.75*diff) * (0.4 + 0.6*ao)
        col = np.stack([base[:, 0]*255, base[:, 1]*210+30, base[:, 2]*255], 1) * glow[:, None]
        col += (ao**2)[:, None]*np.array([40, 20, 60], np.float32)  # ambient tint
        img[hit] = np.clip(col, 0, 255)
    # background: dark gradient
    bg = np.linspace(8, 22, H)[:, None]*np.ones((1, W))
    bgimg = np.stack([bg*0.6, bg*0.5, bg], -1).reshape(-1, 3)
    img[~hit] = bgimg[~hit]
    return np.clip(img.reshape(H, W, 3), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="/tmp/mb.mp4")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--still", action="store_true")
    a = ap.parse_args()
    if a.still:
        Image.fromarray(render_frame(0.7, a.w, a.h)).save(a.output)
        print(f"[OK] still -> {a.output}"); return
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    cnt = a.count or a.frames
    wri = imageio.get_writer(str(out), fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
    for k in range(cnt):
        i = a.start + k
        wri.append_data(render_frame(2*math.pi*i/a.frames, a.w, a.h))
        print(f"  frame {i+1}/{a.frames}", flush=True)
    wri.close()
    print(f"[OK] mandelbulb frames [{a.start},{a.start+cnt}) -> {out}")


if __name__ == "__main__":
    main()
