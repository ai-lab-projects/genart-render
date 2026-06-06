"""Evolved walking creatures (mass-spring soft body + evolution strategy) — self-implemented physics.

The 'video' is ongoing DRAMA: a population of creatures tries to move right; each generation's best
is shown walking, and over generations they go from flailing -> scooting -> bounding. Watch them learn.

Physics: point masses + springs; some springs are 'muscles' whose rest length oscillates (genome =
per-muscle amp/freq/phase). Gravity + ground contact + friction. Fitness = forward distance. Evolution
= (mu,lambda) ES: keep best, mutate. All numpy, ~VM-friendly.

Modes:
  python creature_rl_simulator.py --mode loop --output out.mp4
  python creature_rl_simulator.py --mode still --output out.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import imageio.v2 as imageio

OUT_W, OUT_H = 1280, 720
FPS = 24
DT = 0.004
SUBSTEPS = 6
GROUND_Y = 0.0
K = 140.0          # spring stiffness
DAMP = 6.0         # velocity damping
GRAV = 9.0
MU = 0.9           # ground friction


def make_body():
    """A 2-legged creature, FEET START ON THE GROUND (no weird fall). Body + two legs (thigh+foot)."""
    pts = [
        [0.40, 0.92],   # 0 head/top
        [0.40, 0.60],   # 1 hip
        [0.20, 0.30],   # 2 left knee
        [0.18, 0.00],   # 3 left foot (on ground)
        [0.60, 0.30],   # 4 right knee
        [0.62, 0.00],   # 5 right foot (on ground)
    ]
    P = np.array(pts, np.float32)
    pairs = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5),       # spine + two legs
             (0, 2), (0, 4), (1, 3), (1, 5), (3, 5)]        # cross-braces keep it from collapsing
    springs = [[i, j, float(np.linalg.norm(P[i] - P[j]))] for i, j in pairs]
    return P, springs


def make_amoeba(n_ring=11):
    """A blob: center + ring of masses, cross-linked. Many muscles -> oozes/wobbles like an amoeba."""
    cx, cy, R = 0.45, 0.45, 0.40
    pts = [[cx, cy]] + [[cx + R * np.cos(2 * np.pi * k / n_ring), cy + R * np.sin(2 * np.pi * k / n_ring)]
                        for k in range(n_ring)]
    P = np.array(pts, np.float32)
    P[:, 1] -= P[:, 1].min()                                # rest on ground
    pairs = []
    for k in range(n_ring):
        a, b = 1 + k, 1 + (k + 1) % n_ring
        pairs.append((a, b)); pairs.append((0, a))          # ring + spokes
        pairs.append((a, 1 + (k + 2) % n_ring))             # skip-one (shear) for shape
    springs = [[i, j, float(np.linalg.norm(P[i] - P[j]))] for i, j in pairs]
    return P, springs


def get_body(kind):
    return make_amoeba() if kind == "amoeba" else make_body()


def simulate(P0, springs, genome, steps, record=None):
    """genome: (n_muscle, 3) amp,freq,phase per spring (all springs are muscles here)."""
    P = P0.copy(); V = np.zeros_like(P)
    rest = np.array([s[2] for s in springs], np.float32)
    idx = np.array([[s[0], s[1]] for s in springs])
    amp, freq, phase = genome[:, 0], genome[:, 1], genome[:, 2]
    t = 0.0; frames = []
    for st in range(steps):
        for _ in range(SUBSTEPS):
            cur = rest * (1.0 + 0.35 * amp * np.sin(2 * np.pi * freq * t + phase))
            a = P[idx[:, 0]]; b = P[idx[:, 1]]
            dvec = b - a; L = np.linalg.norm(dvec, axis=1) + 1e-6
            f = (K * (L - cur) / L)[:, None] * dvec          # spring force on a toward b
            F = np.zeros_like(P)
            np.add.at(F, idx[:, 0], f); np.add.at(F, idx[:, 1], -f)
            F[:, 1] -= GRAV                                   # gravity
            F -= DAMP * V
            V += F * DT
            P += V * DT
            below = P[:, 1] < GROUND_Y
            if below.any():
                P[below, 1] = GROUND_Y
                V[below, 1] = np.maximum(V[below, 1], 0.0) * 0.2
                # ANISOTROPIC (ratchet) friction: grip when sliding backward, slip forward -> net locomotion
                vx = V[below, 0]
                vx[vx < 0] *= (1.0 - 0.96)                    # backward slide -> heavy grip (anchor)
                vx[vx >= 0] *= (1.0 - 0.14)                   # forward slide -> low friction (glide)
                V[below, 0] = vx
            t += DT
        if record is not None and st % record == 0:
            frames.append(P.copy())
    return P, frames


def fitness(P0, springs, genome):
    P, _ = simulate(P0, springs, genome, steps=int(2.4 / DT / SUBSTEPS))
    if not np.isfinite(P).all():
        return -1e9
    return float(P[:, 0].mean() - P0[:, 0].mean())           # forward distance


def evolve(P0, springs, rng, gens=22, pop=24):
    nm = len(springs)
    g = np.zeros((nm, 3), np.float32)
    g[:, 0] = rng.uniform(0.3, 1.0, nm); g[:, 1] = rng.uniform(0.5, 2.5, nm); g[:, 2] = rng.uniform(0, 2 * np.pi, nm)
    best = g.copy(); best_f = fitness(P0, springs, g); history = [(best.copy(), best_f)]
    sigma = 0.4
    for gen in range(gens):
        cands = []
        for _ in range(pop):
            c = best.copy()
            c[:, 0] = np.clip(c[:, 0] + rng.normal(0, sigma, nm), 0, 1.2)
            c[:, 1] = np.clip(c[:, 1] + rng.normal(0, sigma, nm), 0.2, 3.5)
            c[:, 2] = (c[:, 2] + rng.normal(0, sigma, nm)) % (2 * np.pi)
            cands.append((c, fitness(P0, springs, c)))
        cands.sort(key=lambda x: -x[1])
        if cands[0][1] > best_f:
            best, best_f = cands[0]
        history.append((best.copy(), best_f))
        sigma *= 0.93
    return history


def _draw(P, springs, cam_x, label):
    img = Image.new("RGB", (OUT_W, OUT_H), (10, 14, 26))
    d = ImageDraw.Draw(img)
    sc = 230.0; gy = OUT_H - 160
    def to_px(p): return ((p[0] - cam_x) * sc + OUT_W * 0.3, gy - p[1] * sc)
    # ground + stripes (parallax of progress)
    d.rectangle([0, gy, OUT_W, OUT_H], fill=(18, 24, 40))
    for gx in range(-2, 40):
        x = (gx - cam_x) * sc + OUT_W * 0.3
        if 0 <= x <= OUT_W:
            d.line([(x, gy), (x, gy + 8)], fill=(40, 54, 80), width=2)
    for s in springs:
        a = to_px(P[s[0]]); b = to_px(P[s[1]])
        d.line([a, b], fill=(70, 150, 200), width=5)
    for p in P:
        x, y = to_px(p)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(120, 220, 255))
    d.text((24, 22), label, fill=(230, 240, 255))
    bloom = img.filter(ImageFilter.GaussianBlur(5))
    return np.clip(np.asarray(img, np.float32) + 0.3 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)


def render_loop(args):
    rng = np.random.default_rng(args.seed)
    P0, springs = get_body(args.body)
    hist = evolve(P0, springs, rng, gens=args.gens)
    # show a few milestone generations walking (early, mid, best)
    gens_to_show = sorted(set([0, len(hist)//4, len(hist)//2, 3*len(hist)//4, len(hist)-1]))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for gi in gens_to_show:
            genome, fit = hist[gi]
            _, frames = simulate(P0, springs, genome, steps=int(3.0 / DT / SUBSTEPS), record=1)
            for P in frames:
                cam = P[:, 0].mean()
                w.append_data(_draw(P, springs, cam, f"Generation {gi}/{len(hist)-1}   distance: {fit:.2f}"))
    finally:
        w.close()
    print(f"[OK] evolved walker: {len(gens_to_show)} milestone generations -> {out}")


def render_best(args):
    """ONE continuous long run of the final best individual (single initial condition, camera follows)."""
    rng = np.random.default_rng(args.seed)
    P0, springs = get_body(args.body)
    hist = evolve(P0, springs, rng, gens=args.gens)
    genome, fit = hist[-1]
    _, frames = simulate(P0, springs, genome, steps=int(args.seconds / DT / SUBSTEPS), record=1)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for P in frames:
            w.append_data(_draw(P, springs, P[:, 0].mean(), f"trained {args.gens} gens   distance: {P[:, 0].mean() - P0[:, 0].mean():.2f}"))
    finally:
        w.close()
    print(f"[OK] best individual continuous run ({args.seconds:.0f}s) -> {out}")


def render_still(args):
    rng = np.random.default_rng(args.seed)
    P0, springs = get_body(args.body)
    hist = evolve(P0, springs, rng, gens=10)
    genome, fit = hist[-1]
    P, _ = simulate(P0, springs, genome, steps=int(1.5 / DT / SUBSTEPS))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_draw(P, springs, P[:, 0].mean(), f"best distance {fit:.2f}")).save(args.output)
    print(f"[OK] creature still -> {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still", "best"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--body", default="legs", choices=["legs", "amoeba"])
    ap.add_argument("--gens", type=int, default=22)
    ap.add_argument("--seconds", type=float, default=22.0)
    args = ap.parse_args()
    {"loop": render_loop, "still": render_still, "best": render_best}[args.mode](args)


if __name__ == "__main__":
    main()
