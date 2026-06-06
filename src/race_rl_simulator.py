"""AI learns to drive (2D race) — cars with ray sensors evolve a policy to lap a track.

Ongoing DRAMA / mastery: early generations crash into walls almost immediately; later generations
hug the racing line and complete smooth laps. Watch the AI get good (the Mario-Maker 'skilled play'
appeal, but it's our own track + our own AI = zero copyright). Self-implemented, numpy, VM-friendly.

Sensors (5 rays) -> tiny linear policy (genome=weights) -> steer + throttle. Off-track = crash.
Fitness = track progress (laps). Evolution = (mu,lambda) ES.

Modes:
  python race_rl_simulator.py --mode loop --output out.mp4
  python race_rl_simulator.py --mode still --output out.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import imageio.v2 as imageio

OUT_W, OUT_H = 1280, 720
FPS = 24
GW, GH = 384, 216
HALF_W = 0.038                 # hard but learnable: gen-0 wipes out (~0.03 laps), late gens master it (~5 laps)
SENSOR_ANGLES = np.radians([-72, -36, 0, 36, 72]).astype(np.float32)
N_SENS = len(SENSOR_ANGLES)


def make_track():
    # sharper harmonics + a hairpin = genuinely hard; a naive "go straight" crashes at the turns,
    # so gen-0 cars mostly wipe out and only later gens carry speed through the bends.
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False, dtype=np.float32)
    cx = 0.5 + 0.33 * np.cos(t) + 0.08 * np.cos(3 * t) + 0.020 * np.cos(5 * t + 0.6)
    cy = 0.5 + 0.23 * np.sin(t) + 0.06 * np.sin(2 * t + 1.0) + 0.020 * np.sin(4 * t)
    center = np.stack([cx, cy], 1)
    # rasterize on-track grid + progress (loop fraction of nearest centerline point)
    gx = np.linspace(0, 1, GW, dtype=np.float32)[None, :]
    gy = np.linspace(0, 1, GH, dtype=np.float32)[:, None]
    on = np.zeros((GH, GW), bool); prog = np.zeros((GH, GW), np.float32)
    GXX = np.broadcast_to(gx, (GH, GW)); GYY = np.broadcast_to(gy, (GH, GW))
    mind = np.full((GH, GW), 1e9, np.float32)
    for i in range(len(center)):
        d = (GXX - center[i, 0]) ** 2 + (GYY - center[i, 1]) ** 2
        upd = d < mind
        mind[upd] = d[upd]; prog[upd] = i / len(center)
    on = mind < HALF_W ** 2
    return center, on, prog


def _ontrack(on, x, y):
    if not (0 <= x < 1 and 0 <= y < 1):
        return False
    return bool(on[min(GH - 1, int(y * GH)), min(GW - 1, int(x * GW))])


def _sensors(on, x, y, h):
    out = np.empty(N_SENS, np.float32)
    for i, a in enumerate(SENSOR_ANGLES):
        ca, sa = np.cos(h + a), np.sin(h + a); r = 0.0
        while r < 0.28 and _ontrack(on, x + ca * r, y + sa * r):
            r += 0.012
        out[i] = r / 0.28
    return out


def run_car(center, on, prog, genome, max_steps=900):
    # start at centerline[0], heading toward next point
    x, y = float(center[0, 0]), float(center[0, 1])
    h = float(np.arctan2(center[1, 1] - y, center[1, 0] - x))
    speed = 0.004; W = genome.reshape(2, N_SENS + 2)
    last = prog[min(GH - 1, int(y * GH)), min(GW - 1, int(x * GW))]; total = 0.0
    trail = [(x, y, h)]
    for st in range(max_steps):
        s = _sensors(on, x, y, h)
        inp = np.concatenate([s, [speed * 60, 1.0]]).astype(np.float32)
        steer, throttle = np.tanh(W @ inp)
        h += float(steer) * 0.21              # more steering authority -> sharp turns are passable once learned
        speed = float(np.clip(speed + (throttle * 0.5 + 0.1) * 0.0009, 0.0015, 0.011))
        x += np.cos(h) * speed; y += np.sin(h) * speed
        if not _ontrack(on, x, y):
            break
        p = prog[min(GH - 1, int(y * GH)), min(GW - 1, int(x * GW))]
        dp = p - last
        if dp < -0.5: dp += 1.0          # lap wrap
        if dp > 0.5: dp -= 1.0
        total += dp; last = p
        trail.append((x, y, h))                       # store heading for car orientation
    return total, trail


def evolve(center, on, prog, rng, gens=34, pop=26, topk=16):
    """Returns per-gen (best_genome, best_fit, population_top_genomes) so render can show the SWARM."""
    nW = 2 * (N_SENS + 2)
    best = rng.normal(0, 0.8, nW).astype(np.float32)
    bf, _ = run_car(center, on, prog, best)
    hist = [(best.copy(), bf, [best + rng.normal(0, 0.6, nW).astype(np.float32) for _ in range(topk)])]
    sigma = 0.6
    for g in range(gens):
        cands = [(best.copy(), bf)]
        for _ in range(pop):
            c = best + rng.normal(0, sigma, nW).astype(np.float32)
            f, _ = run_car(center, on, prog, c); cands.append((c, f))
        cands.sort(key=lambda x: -x[1])
        if cands[0][1] > bf: best, bf = cands[0]
        hist.append((best.copy(), bf, [c for c, _ in cands[:topk]]))
        sigma *= 0.93
    return hist


def _bg(on, center):
    """Asphalt road + curbs (red/white edges) + dashed yellow centerline = top-down racing look."""
    img = Image.new("RGB", (OUT_W, OUT_H), (22, 30, 24))     # grass-ish
    road = np.zeros((GH, GW, 3), np.uint8)
    road[on] = (54, 56, 62)                                  # asphalt grey
    mask = Image.fromarray((on * 255).astype(np.uint8)).resize((OUT_W, OUT_H))
    img = Image.composite(Image.fromarray(road).resize((OUT_W, OUT_H), Image.LANCZOS), img, mask)
    d = ImageDraw.Draw(img)
    # edges (curbs): offset centerline by ±half_width along normals
    tang = np.gradient(center, axis=0); nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
    for side, col in ((1, (210, 80, 80)), (-1, (220, 220, 230))):
        e = center + side * HALF_W * nrm
        for i in range(0, len(e), 2):
            if (i // 2) % 2 == 0:                            # dashed curb
                a = e[i]; b = e[(i + 2) % len(e)]
                d.line([(a[0] * OUT_W, a[1] * OUT_H), (b[0] * OUT_W, b[1] * OUT_H)], fill=col, width=5)
    # dashed yellow centerline
    for i in range(0, len(center), 4):
        if (i // 4) % 2 == 0:
            a = center[i]; b = center[(i + 2) % len(center)]
            d.line([(a[0] * OUT_W, a[1] * OUT_H), (b[0] * OUT_W, b[1] * OUT_H)], fill=(225, 200, 90), width=3)
    return img


def _car(d, x, y, h, col=(90, 200, 255), alive=True):
    cs, sn = np.cos(h), np.sin(h)
    def T(pts, ox=0.0, oy=0.0):
        return [((x + (px * cs - py * sn) + ox) * OUT_W, (y + (px * sn + py * cs) + oy) * OUT_H) for px, py in pts]
    if not alive:
        col = (120, 80, 80)
    body = [(0.024, 0), (0.012, 0.013), (-0.022, 0.013), (-0.022, -0.013), (0.012, -0.013)]
    d.polygon(T(body, 0.005, 0.006), fill=(6, 10, 16))                  # drop shadow (fake 3D lift)
    # wheels (dark, just outside the chassis) read as a vehicle from top-down
    def wheel(wx, wy):
        return T([(wx + 0.006, wy + 0.004), (wx - 0.006, wy + 0.004), (wx - 0.006, wy - 0.004), (wx + 0.006, wy - 0.004)])
    for wx, wy in [(0.013, 0.015), (0.013, -0.015), (-0.015, 0.015), (-0.015, -0.015)]:
        d.polygon(wheel(wx, wy), fill=(16, 18, 24))
    d.polygon(T(body), fill=col)
    d.polygon(T([(0.012, 0.013), (-0.022, 0.013), (-0.022, 0.005), (0.012, 0.005)]),    # shaded lower flank = curvature
              fill=tuple(int(c * 0.68) for c in col))
    roof = [(0.012, 0.009), (-0.014, 0.009), (-0.014, -0.009), (0.012, -0.009)]
    lighter = tuple(min(255, int(c * 1.32) + 28) for c in col)
    d.polygon(T(roof), fill=lighter)                                    # roof highlight = volume
    d.polygon(T([(0.011, 0.007), (0.001, 0.007), (0.001, -0.007), (0.011, -0.007)]), fill=(18, 30, 46))  # windshield
    for hy in (0.008, -0.008):                                          # headlights
        hp = T([(0.023, hy)])[0]
        d.ellipse([hp[0] - 3, hp[1] - 3, hp[0] + 3, hp[1] + 3], fill=(255, 240, 180) if alive else (90, 60, 60))


def _explosion(d, x, y, age):
    """Fireball that expands + fades over ~age frames since the crash. age in render-frame units."""
    fade = max(0.0, 1.0 - age / 22.0)
    if fade <= 0:
        return
    px, py = x * OUT_W, y * OUT_H
    rad = 0.005 + age * 0.0011
    for i in range(11):                                                 # flung debris/sparks (deterministic)
        a = i * (2 * np.pi / 11) + age * 0.12
        dist = rad * (0.55 + 0.45 * ((i * 37) % 11) / 11.0)
        ex, ey = px + np.cos(a) * dist * OUT_W, py + np.sin(a) * dist * OUT_H
        s = max(1.5, 7.0 - age * 0.28)
        d.ellipse([ex - s, ey - s, ex + s, ey + s], fill=(255, int(170 * fade) + 40, 30))
    core = (0.011 * OUT_W) * fade                                       # bright hot core
    if core > 1:
        d.ellipse([px - core, py - core, px + core, py + core], fill=(255, int(225 * fade) + 25, 140))
        d.ellipse([px - core * 0.5, py - core * 0.5, px + core * 0.5, py + core * 0.5], fill=(255, 255, 220))


def _frame_pop(bg, states, gi, ngi, fit):
    img = bg.copy(); d = ImageDraw.Draw(img)
    for trail_pts, x, y, h, alive, isbest, death_age in states:
        if len(trail_pts) > 1:
            d.line(trail_pts, fill=(48, 52, 60) if alive else (44, 30, 30), width=4, joint="curve")
    for trail_pts, x, y, h, alive, isbest, death_age in states:
        if (not alive) and death_age is not None and death_age < 22:
            _explosion(d, x, y, death_age)                  # fresh crash -> fireball, no car drawn
            continue
        col = (255, 225, 90) if (isbest and alive) else ((90, 200, 255) if alive else (90, 60, 60))
        _car(d, x, y, h, col, alive)                        # survivors + burnt-out wrecks
    d.text((24, 22), f"Generation {gi}/{ngi}   best laps: {fit:.2f}", fill=(235, 240, 255))
    bloom = img.filter(ImageFilter.GaussianBlur(3))
    return np.clip(np.asarray(img, np.float32) + 0.16 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8)


def render_loop(args):
    rng = np.random.default_rng(args.seed)
    center, on, prog = make_track()
    hist = evolve(center, on, prog, rng)
    bg = _bg(on, center)
    # show EARLY gens densely (where the swarm crashes/struggles) + milestones
    gens = sorted(set([0, 1, 2, 3, 5, len(hist) // 3, 2 * len(hist) // 3, len(hist) - 1]))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8, macro_block_size=8)
    try:
        for gi in gens:
            best, fit, popg = hist[gi]
            trails = [run_car(center, on, prog, g, max_steps=1500)[1] for g in popg]   # whole SWARM
            order = sorted(range(len(trails)), key=lambda i: -len(trails[i]))
            best_i = order[0]
            maxlen = max(len(t) for t in trails)
            for k in range(2, maxlen + 24, 3):              # +24 so final crashes' fireballs finish playing
                states = []
                for i, t in enumerate(trails):
                    if len(t) < 2:
                        continue
                    crash = len(t) - 1
                    kk = min(k, crash); x, y, h = t[kk]; alive = k < crash
                    death_age = None if alive else (k - crash)
                    pts = [(px * OUT_W, py * OUT_H) for px, py, _ in t[:kk + 1]]
                    states.append((pts, x, y, h, alive, i == best_i, death_age))
                w.append_data(_frame_pop(bg, states, gi, len(hist) - 1, fit))
    finally:
        w.close()
    print(f"[OK] AI race swarm: {len(gens)} generations -> {out}")


def render_still(args):
    rng = np.random.default_rng(args.seed)
    center, on, prog = make_track()
    hist = evolve(center, on, prog, rng, gens=12)
    best, fit, popg = hist[2]                                   # an early gen -> show the struggling swarm
    trails = [run_car(center, on, prog, g, max_steps=900)[1] for g in popg]
    states = []
    for i, t in enumerate(trails):
        if len(t) < 2:
            continue
        x, y, h = t[-1]; alive = len(t) > 850
        pts = [(px * OUT_W, py * OUT_H) for px, py, _ in t]
        states.append((pts, x, y, h, alive, i == 0, None if alive else 6))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_frame_pop(_bg(on, center), states, 2, len(hist) - 1, fit)).save(args.output)
    print(f"[OK] race still -> {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="loop", choices=["loop", "still"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    (render_loop if args.mode == "loop" else render_still)(args)


if __name__ == "__main__":
    main()
