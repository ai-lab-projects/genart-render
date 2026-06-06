"""Diffusion model visualized on 2D data — the PRINCIPLE behind Stable Diffusion / DALL-E / Sora.

A tiny denoiser MLP is trained (stable MSE regression — unlike a GAN's adversarial game) to remove
noise. Sampling: start from pure noise and iteratively DENOISE; a random cloud organizes into the
target shape, step by step. This is exactly how modern image/video-gen AI works, in 2D miniature.

Self-implemented (numpy + manual backprop, reuses gan_simulator.MLP). VM-friendly.

Modes:
  python diffusion_simulator.py --mode loop --output out.mp4 --target spiral
  python diffusion_simulator.py --mode still --output out.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import imageio.v2 as imageio

from gan_simulator import MLP, target_samples

OUT_W, OUT_H = 1280, 720
FPS = 24
T = 60                                   # diffusion steps


def schedule():
    beta = np.linspace(1e-4, 0.02, T).astype(np.float32)
    alpha = 1 - beta
    abar = np.cumprod(alpha).astype(np.float32)
    return beta, alpha, abar


def train(kind, iters, rng):
    beta, alpha, abar = schedule()
    net = MLP(3, 96, 2, "linear", rng)   # input: (x, y, t/T) -> predicted noise (2)
    NB = 256
    for it in range(iters):
        x0 = target_samples(kind, NB, rng)
        t = rng.integers(0, T, NB)
        noise = rng.normal(0, 1, (NB, 2)).astype(np.float32)
        at = abar[t][:, None]
        xt = np.sqrt(at) * x0 + np.sqrt(1 - at) * noise
        inp = np.concatenate([xt, (t / T)[:, None].astype(np.float32)], 1)
        pred = net.forward(inp)
        net.backward(2 * (pred - noise) / NB)            # MSE grad (stable regression)
        net.step(2e-3)
    return net


def sample(net, rng, n=700, record_every=1):
    """DDIM (deterministic) reverse sampling -> crisp trajectories that land ON the target manifold."""
    beta, alpha, abar = schedule()
    x = rng.normal(0, 1, (n, 2)).astype(np.float32)
    snaps = [(x.copy(), T)]
    for t in reversed(range(T)):
        inp = np.concatenate([x, np.full((n, 1), t / T, np.float32)], 1)
        eps = net.forward(inp)
        ab = abar[t]
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)              # predicted clean point
        ab_prev = abar[t - 1] if t > 0 else np.float32(1.0)
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * eps      # DDIM step (no added noise)
        if t % record_every == 0 or t == 0:
            snaps.append((x.copy(), t))
    return snaps


def _render(pts, real, title, prog, show_target=True):
    im = Image.new("RGB", (OUT_W, OUT_H), (8, 10, 20))
    d = ImageDraw.Draw(im)
    def to_px(p): return (p[0] * 640 + OUT_W / 2, -p[1] * 640 + OUT_H / 2)
    if show_target:
        for p in real[:380]:
            x, y = to_px(p); d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(64, 72, 92))
    for p in pts:
        x, y = to_px(p)
        if -60 < x < OUT_W + 60 and -60 < y < OUT_H + 60:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(150, 230, 255))
    # progress bar
    d.rectangle([24, OUT_H - 36, 24 + 300, OUT_H - 24], outline=(80, 90, 110))
    d.rectangle([24, OUT_H - 36, 24 + int(prog * 300), OUT_H - 24], fill=(120, 220, 255))
    # legend (so it's not confusing)
    d.ellipse([OUT_W - 250, 30, OUT_W - 240, 40], fill=(150, 230, 255)); d.text((OUT_W - 232, 28), "generated points", fill=(200, 220, 240))
    if show_target:
        d.ellipse([OUT_W - 250, 52, OUT_W - 240, 62], fill=(64, 72, 92)); d.text((OUT_W - 232, 50), "target (real data)", fill=(150, 160, 180))
    d.text((24, 22), title, fill=(235, 240, 255))
    bloom = im.filter(ImageFilter.GaussianBlur(4))
    return np.clip(np.asarray(im, np.float32) + 0.32 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)


def render_loop(args):
    rng = np.random.default_rng(args.seed)
    net = train(args.target, args.iters, rng)
    real = target_samples(args.target, 380, rng)
    beta, alpha, abar = schedule()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        # PHASE 1 (forward): show real data, then progressively add noise -> pure noise (the 'training' idea)
        x0 = target_samples(args.target, 700, rng)
        for _ in range(16):
            w.append_data(_render(x0, real, "STEP 1: this is the real data we want to generate", 0.0, show_target=False))
        for t in range(0, T, 1):
            at = abar[t]
            xt = np.sqrt(at) * x0 + np.sqrt(1 - at) * rng.normal(0, 1, x0.shape).astype(np.float32)
            w.append_data(_render(xt, real, "STEP 2: keep adding noise  ->  pure random noise", t / T, show_target=False))
        for _ in range(10):
            w.append_data(_render(rng.normal(0, 1, (700, 2)).astype(np.float32), real, "now it's just noise", 1.0, show_target=False))
        # PHASE 2 (reverse): from noise, the AI DENOISES step by step -> regenerates the data distribution
        snaps = sample(net, rng)
        for pts, t in snaps:
            for _ in range(2):
                w.append_data(_render(pts, real, "STEP 3: the AI removes noise step by step  ->  generates data", 1 - t / T))
        for _ in range(48):
            w.append_data(_render(snaps[-1][0], real, "generated! (matches the target distribution)", 1.0))
    finally:
        w.close()
    print(f"[OK] diffusion forward+reverse explained -> {out}")


def render_still(args):
    rng = np.random.default_rng(args.seed)
    net = train(args.target, args.iters, rng)
    real = target_samples(args.target, 380, rng)
    snaps = sample(net, rng)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_render(snaps[-1][0], real, "generated (matches target)", 1.0)).save(args.output)
    print(f"[OK] diffusion still -> {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--target", default="spiral", choices=["ring","spiral","two_rings","blobs"])
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    (render_loop if args.mode == "loop" else render_still)(args)


if __name__ == "__main__":
    main()
