"""GAN visualized on 2D data — watch a generator turn noise into a target SHAPE while a discriminator
chases it. Self-implemented MLPs + manual backprop + Adam (numpy). VM-friendly, no ML framework.

The 'wow': generated points (cyan) start as a random blob and gradually organize into the target
distribution (a ring / spiral), as the discriminator (background heatmap = its real/fake belief)
co-adapts. The classic adversarial dance, made visible.

Modes:
  python gan_simulator.py --mode loop --output out.mp4 --target ring
  python gan_simulator.py --mode still --output out.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import imageio.v2 as imageio

OUT_W, OUT_H = 1280, 720
FPS = 24


def target_samples(kind, n, rng):
    if kind == "blobs":
        K = 6; cen = np.stack([0.32 * np.cos(np.linspace(0, 2 * np.pi, K, endpoint=False)),
                               0.32 * np.sin(np.linspace(0, 2 * np.pi, K, endpoint=False))], 1)
        idx = rng.integers(0, K, n)
        return (cen[idx] + rng.normal(0, 0.035, (n, 2))).astype(np.float32)
    if kind == "spiral":
        t = rng.uniform(0, 1, n) ** 0.5 * 3.2
        r = 0.05 + t * 0.12
        x = r * np.cos(t * 3.0); y = r * np.sin(t * 3.0)
    elif kind == "two_rings":
        t = rng.uniform(0, 2 * np.pi, n); r = np.where(rng.random(n) < 0.5, 0.18, 0.42)
        x = r * np.cos(t); y = r * np.sin(t)
    else:  # ring
        t = rng.uniform(0, 2 * np.pi, n); r = 0.38 + rng.normal(0, 0.015, n)
        x = r * np.cos(t); y = r * np.sin(t)
    return np.stack([x, y], 1).astype(np.float32)


class MLP:
    """1 hidden layer, tanh. out_act: 'linear' (generator) or 'sigmoid' (discriminator)."""
    def __init__(self, din, dh, dout, out_act, rng):
        s1 = np.sqrt(1.0 / din); s2 = np.sqrt(1.0 / dh)
        self.W1 = (rng.normal(0, s1, (dh, din))).astype(np.float32); self.b1 = np.zeros(dh, np.float32)
        self.W2 = (rng.normal(0, s2, (dout, dh))).astype(np.float32); self.b2 = np.zeros(dout, np.float32)
        self.out_act = out_act
        self.m = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in ("W1", "b1", "W2", "b2")}
        self.t = 0

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1.T + self.b1; self.h = np.tanh(self.z1)
        o = self.h @ self.W2.T + self.b2
        self.out = 1.0 / (1.0 + np.exp(-o)) if self.out_act == "sigmoid" else o
        return self.out

    def backward(self, dout):
        """dout = dL/d(out). Returns dL/dX (for backprop into generator). Accumulates param grads."""
        N = self.X.shape[0]
        dW2 = dout.T @ self.h / N; db2 = dout.mean(0)
        dh = dout @ self.W2
        dz1 = dh * (1 - self.h ** 2)
        dW1 = dz1.T @ self.X / N; db1 = dz1.mean(0)
        self.g = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return dz1 @ self.W1

    def step(self, lr):
        self.t += 1; b1, b2, eps = 0.9, 0.999, 1e-8
        for k in ("W1", "b1", "W2", "b2"):
            g = self.g[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))


def train(kind, iters, rng):
    G = MLP(2, 96, 2, "linear", rng)          # more capacity
    D = MLP(2, 64, 1, "sigmoid", rng)
    snaps = []
    NB = 256
    for it in range(iters):
        sig = 0.12 * max(0.0, 1 - it / (iters * 0.7))      # instance noise (decaying) — fights mode collapse
        real = target_samples(kind, NB, rng)
        z = rng.normal(0, 1, (NB, 2)).astype(np.float32)
        fake = G.forward(z)
        nz = lambda a: a + rng.normal(0, sig, a.shape).astype(np.float32)
        # --- D step (2x, with instance noise) ---
        for _ in range(2):
            dr = D.forward(nz(real)); D.backward(-(1 - dr)); D.step(1.5e-3)
            df = D.forward(nz(fake)); D.backward(df); D.step(1.5e-3)
        # --- G step (non-saturating) ---
        df2 = D.forward(fake)
        dfake_pts = D.backward(-(1 - df2))
        G.forward(z); G.backward(dfake_pts); G.step(2.5e-3)
        if it % max(1, iters // 220) == 0:
            zz = rng.normal(0, 1, (700, 2)).astype(np.float32)
            snaps.append((G.forward(zz).copy(), it))
    return G, D, snaps


def _heat(D):
    gx = np.linspace(-0.6, 0.6, OUT_W // 8, dtype=np.float32)
    gy = np.linspace(-0.4, 0.4, OUT_H // 8, dtype=np.float32)
    XX, YY = np.meshgrid(gx, gy)
    P = np.stack([XX.ravel(), YY.ravel()], 1)
    d = D.forward(P).reshape(YY.shape)
    return d


def _render(pts, real, D, it):
    img = np.zeros((OUT_H, OUT_W, 3), np.float32) + np.array([8, 10, 20], np.float32)
    # discriminator belief heatmap (red=thinks real, blue=thinks fake)
    d = _heat(D)
    heat = np.zeros((d.shape[0], d.shape[1], 3), np.float32)
    heat[..., 0] = 60 * d; heat[..., 2] = 60 * (1 - d)
    img += np.asarray(Image.fromarray(heat.astype(np.uint8)).resize((OUT_W, OUT_H), Image.BILINEAR), np.float32)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8));
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    def to_px(p): return (p[0] * 700 + OUT_W / 2, -p[1] * 700 + OUT_H / 2)
    for p in real[:400]:
        x, y = to_px(p); dr.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(90, 100, 120))
    for p in pts:
        x, y = to_px(p); dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(90, 230, 255))
    dr.text((24, 22), f"GAN training   step {it}", fill=(235, 240, 255))
    bloom = im.filter(ImageFilter.GaussianBlur(4))
    return np.clip(np.asarray(im, np.float32) + 0.35 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)


def render_loop(args):
    rng = np.random.default_rng(args.seed)
    G, D, snaps = train(args.target, args.iters, rng)
    real = target_samples(args.target, 400, rng)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for pts, it in snaps:
            for _ in range(3):                       # hold each snapshot a few frames
                w.append_data(_render(pts, real, D, it))
    finally:
        w.close()
    print(f"[OK] GAN viz: {len(snaps)} snapshots over {args.iters} steps -> {out}")


def render_still(args):
    rng = np.random.default_rng(args.seed)
    G, D, snaps = train(args.target, args.iters, rng)
    real = target_samples(args.target, 400, rng)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(snaps[-1][0], real, D, snaps[-1][1])).save(args.output)
    print(f"[OK] GAN still -> {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", default="ring", choices=["ring", "spiral", "two_rings"])
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    (render_loop if args.mode == "loop" else render_still)(args)


if __name__ == "__main__":
    main()
