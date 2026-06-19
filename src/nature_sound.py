"""Procedural nature ambience (rain / stream / waves) — numpy, synthesized, copyright-free.

Pairs with lo-fi (the proven "lo-fi + rain" combo) or ambient pads. No field recordings,
no samples -> monetization-safe + Spotify-distributable, like our music/visuals.

Method: shaped filtered noise (the bed) + sparse transient events (droplets / bubbles),
gentle low-pass for warmth, slow amplitude drift so it breathes.

CLI:
  python nature_sound.py --type rain   --output outputs/rain.wav  --seconds 60
  python nature_sound.py --type stream --output outputs/stream.wav --seconds 60
  python nature_sound.py --type waves  --output outputs/waves.wav  --seconds 60
"""
from __future__ import annotations
import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100


def box_lowpass(x, win):
    if win <= 1:
        return x
    return np.convolve(x, np.ones(win, np.float32) / win, mode="same").astype(np.float32)


def _highpass(x, win):
    return x - box_lowpass(x, win)


def rain(n, rng):
    # SIMPLE = better here (user prefers near white/pink noise over heavy texture, 2026-06-05).
    # mostly gentle pink-ish noise + only a hint of droplets + very subtle slow drift.
    t = np.arange(n, dtype=np.float32) / SR
    bed = box_lowpass(rng.standard_normal(n).astype(np.float32), 4) * 0.22   # pink-ish, not over-dark
    bed = _highpass(bed, 80)
    bed *= 0.92 + 0.08 * np.sin(2 * np.pi * 0.07 * t)                        # barely-there drift (no strong gusts)
    drops = np.zeros(n, np.float32)
    for i in rng.integers(0, n - 700, int(n / SR * 28)):                    # few, quiet
        ln = int(rng.integers(150, 380)); tt = np.arange(ln, dtype=np.float32) / SR
        nz = rng.standard_normal(ln).astype(np.float32)
        splash = (box_lowpass(nz, int(rng.integers(3, 6))) - box_lowpass(nz, 26)) * np.exp(-tt * rng.uniform(30, 55))
        drops[i:i + ln] += splash * rng.uniform(0.03, 0.08)
    return bed + drops


def stream(n, rng):
    # gurgling water: low-passed noise that breathes + occasional higher bubbles
    bed = box_lowpass(rng.standard_normal(n).astype(np.float32), 8) * 0.5
    t = np.arange(n, dtype=np.float32) / SR
    drift = 0.7 + 0.3 * np.sin(2 * np.pi * 0.15 * t) + 0.15 * np.sin(2 * np.pi * 0.07 * t + 1.0)
    bed *= drift
    bubbles = np.zeros(n, np.float32)
    n_b = int(n / SR * 30)
    for i in rng.integers(0, n - 2000, n_b):
        ln = rng.integers(600, 1800)
        tt = np.arange(ln, dtype=np.float32) / SR
        f0 = rng.uniform(400, 1100)
        chirp = np.sin(2 * np.pi * (f0 + 600 * tt / (ln / SR)) * tt) * np.exp(-tt * 14)
        bubbles[i:i + ln] += chirp * rng.uniform(0.04, 0.12)
    return bed * 0.5 + bubbles


def waves(n, rng):
    # slow ocean swell: noise enveloped by a ~9s wash in/out
    t = np.arange(n, dtype=np.float32) / SR
    bed = box_lowpass(rng.standard_normal(n).astype(np.float32), 10) * 0.6
    period = 9.0
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * t / period - np.pi / 2)
    swell = swell ** 1.8                                # sharper crest, long trough
    foam = _highpass(rng.standard_normal(n).astype(np.float32), 20) * 0.12 * (swell ** 3)
    return bed * swell + foam


def fireplace(n, rng):
    # soft continuous roar + irregular WOODY crackle of varied size (not uniform clicks)
    roar = box_lowpass(rng.standard_normal(n).astype(np.float32), 40) * 0.16   # softer, deeper
    crackle = np.zeros(n, np.float32)
    pos = 0.0
    while pos < n - 800:
        ln = int(rng.integers(120, 420))
        tt = np.arange(ln, dtype=np.float32) / SR
        nz = rng.standard_normal(ln).astype(np.float32)
        # band-limited woody pop (not a sharp digital click), varied brightness + decay
        pop = (box_lowpass(nz, int(rng.integers(2, 5))) - box_lowpass(nz, 22)) * np.exp(-tt * rng.uniform(45, 110))
        big = 2.6 if rng.random() < 0.05 else 1.0            # rare louder snap
        crackle[int(pos):int(pos) + ln] += pop * rng.uniform(0.05, 0.22) * big
        pos += rng.exponential(0.045) * SR + 0.012 * SR      # irregular gaps (Poisson), avg ~17/s
    return roar + crackle


def thunder(n, rng):
    # rumbles at IRREGULAR gaps (Poisson), each a rolling low rumble + occasional sharp crack
    out = np.zeros(n, np.float32)
    pos = rng.uniform(2, 9) * SR
    while pos < n - 5 * SR:
        ln = int(rng.uniform(3.0, 6.0) * SR)
        tt = np.arange(ln, dtype=np.float32) / SR
        nz = box_lowpass(rng.standard_normal(ln).astype(np.float32), int(rng.integers(45, 85)))  # deep
        env = (1 - np.exp(-tt * rng.uniform(2.0, 5.0))) * np.exp(-tt * rng.uniform(0.6, 1.1))
        roll = 0.65 + 0.35 * np.sin(2 * np.pi * rng.uniform(0.8, 2.4) * tt + rng.random() * 6.0)  # rolling undulation
        rumble = nz * env * roll
        if rng.random() < 0.5:                               # sometimes a sharper initial crack
            cl = int(0.18 * SR); ct = np.arange(cl, dtype=np.float32) / SR
            crack = (rng.standard_normal(cl).astype(np.float32))
            crack = (crack - box_lowpass(crack, 6)) * np.exp(-ct * 26) * 0.45
            rumble[:cl] += crack
        out[int(pos):int(pos) + ln] += rumble * rng.uniform(0.55, 1.0)
        pos += rng.exponential(13.0) * SR + 5.0 * SR         # irregular: avg ~18s, min 5s
    return out


def cafe(n, rng):
    # coffee-shop BABBLE: speech-band noise modulated at a wobbling syllable rate (chatter, not hiss),
    # kept quiet, + low room tone + sparse cup clinks. NOTE: convincing crowd voice is hard to synth.
    t = np.arange(n, dtype=np.float32) / SR
    base = rng.standard_normal(n).astype(np.float32)
    band = box_lowpass(base, 6) - box_lowpass(base, 26)      # ~speech mid-band
    rate = box_lowpass(rng.standard_normal(n).astype(np.float32), 300) * 7.0   # wobbling syllable rate
    am = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t + rate), 0, 1) ** 1.6  # syllabic envelope
    babble = band * am * 0.14                                # quieter than before (was a too-loud 'shhh')
    room = box_lowpass(rng.standard_normal(n).astype(np.float32), 45) * 0.04
    clinks = np.zeros(n, np.float32)
    for i in rng.integers(0, n - 800, max(1, int(n / SR * 0.5))):   # ~1 clink / 2s
        ln = int(rng.integers(200, 500)); tt = np.arange(ln, dtype=np.float32) / SR
        nz = rng.standard_normal(ln).astype(np.float32)
        clink = (nz - box_lowpass(nz, 4)) * np.exp(-tt * rng.uniform(50, 100))
        clinks[i:i + ln] += clink * rng.uniform(0.04, 0.10)
    return babble + room + clinks


GEN = {"rain": rain, "stream": stream, "waves": waves,
       "fireplace": fireplace, "thunder": thunder, "cafe": cafe}


def render(kind, seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    mono = GEN[kind](n, rng)
    mono = box_lowpass(mono, 2)                          # warmth
    # gentle stereo: decorrelate slightly
    L = mono + 0.3 * np.roll(box_lowpass(rng.standard_normal(n).astype(np.float32), 4) * 0.05, 5)
    R = mono + 0.3 * np.roll(box_lowpass(rng.standard_normal(n).astype(np.float32), 4) * 0.05, -5)
    fade = int(2.0 * SR)
    for ch in (L, R):
        ch[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        ch[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
    g = 10 ** (-6 / 20) / peak                           # leave headroom (it's a layer, not the lead)
    return L * g, R * g


def write_wav(path, L, R):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    inter = np.empty((len(L) * 2,), np.float32)
    inter[0::2] = L; inter[1::2] = R
    pcm = (np.clip(inter, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="rain", choices=list(GEN))
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    L, R = render(a.type, a.seconds, a.seed)
    write_wav(a.output, L, R)
    print(f"[OK] nature '{a.type}' {a.seconds:.0f}s -> {a.output}")


if __name__ == "__main__":
    main()
