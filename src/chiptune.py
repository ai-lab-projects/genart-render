"""Chiptune (NES/Famicom-style) synth — ORIGINAL game-style music, 100% copyright-clean.

We can't take famous game soundtracks (they're owned + Content-ID'd -> would strike the channel).
Instead we synthesize the SOUND of that era from scratch (pulse + triangle + noise channels, the
literal 2A03 voice set) and play our OWN composition. Captures the 'game music' vibe, zero IP risk.

Channels: 2 pulse (lead + arpeggio harmony, duty-cycled square), 1 triangle (bass), 1 noise (drums).
All numpy, no samples, no framework.

CLI:
  python chiptune.py --output out.wav --tune adventure --seconds 60 [--seed 3]
"""
from __future__ import annotations
import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100


def midi_freq(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def pulse(freq, dur, duty=0.5, vol=0.25, vib=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = freq * (1.0 + vib * np.sin(2 * np.pi * 5.5 * t))     # subtle vibrato
    ph = np.cumsum(f) / SR
    w = np.where((ph % 1.0) < duty, 1.0, -1.0).astype(np.float32)
    return w * env(n) * vol


def triangle(freq, dur, vol=0.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = (freq * t) % 1.0
    w = (2.0 * np.abs(2.0 * ph - 1.0) - 1.0).astype(np.float32)  # /\/\ triangle
    w = np.round(w * 7) / 7                                       # 4-bit-ish quantize = NES bass grit
    return w * env(n, a=0.002, r=0.02) * vol


def noise(dur, vol=0.3, lp=1, decay=True, seed=0):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, n).astype(np.float32)
    if lp > 1:
        w = np.convolve(w, np.ones(lp, np.float32) / lp, mode="same")
    e = np.exp(-np.linspace(0, 6, n)).astype(np.float32) if decay else env(n)
    return w * e * vol


def env(n, a=0.004, d=0.04, s=0.7, r=0.05):
    """Simple ADSR over n samples."""
    e = np.ones(n, np.float32) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na, nd, nr = min(na, n), min(nd, n), min(nr, n)
    if na: e[:na] = np.linspace(0, 1, na)
    if nd: e[na:na + nd] = np.linspace(1, s, nd)
    if nr: e[-nr:] = np.linspace(e[-nr], 0, nr)
    return e


def add(buf, sig, t):
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[:j - i]


# --- original tunes (our compositions; scale degrees as MIDI) ---
TUNES = {
    # I-V-vi-IV style, bright C major adventure/overworld feel
    "adventure": {
        "bpm": 144,
        "chords": [  # (root_midi, [chord tones for arpeggio])
            (48, [60, 64, 67]), (43, [55, 59, 62]), (45, [57, 60, 64]), (41, [53, 57, 60]),
            (48, [60, 64, 67]), (43, [55, 59, 62]), (41, [53, 57, 60]), (43, [55, 59, 62]),
        ],
        "lead": [  # (midi, beats) per bar, concatenated
            (76, 1), (79, 1), (76, 1), (72, 1),     (74, 1), (79, 1), (74, 1), (71, 1),
            (72, 1), (76, 1), (69, 1), (72, 1),     (69, 1), (77, 1), (76, 0.5), (74, 0.5), (72, 1),
            (76, 1), (79, 1), (84, 1), (79, 1),     (71, 1), (74, 1), (79, 1), (74, 1),
            (69, 1), (72, 1), (77, 1), (81, 1),     (79, 0.5), (74, 0.5), (71, 1), (67, 2),
        ],
    },
    # darker minor "boss/battle" feel, A minor, faster
    "boss": {
        "bpm": 168,
        "chords": [
            (45, [57, 60, 64]), (44, [56, 59, 64]), (41, [53, 57, 60]), (43, [55, 59, 62]),
            (45, [57, 60, 64]), (40, [52, 55, 59]), (41, [53, 57, 60]), (43, [55, 59, 62]),
        ],
        "lead": [
            (69, 0.5), (72, 0.5), (76, 1), (75, 1), (72, 1),    (71, 0.5), (74, 0.5), (76, 1), (72, 2),
            (69, 0.5), (68, 0.5), (65, 1), (69, 1), (72, 1),    (74, 1), (71, 1), (67, 2),
            (69, 0.5), (72, 0.5), (76, 1), (79, 1), (76, 1),    (75, 0.5), (76, 0.5), (75, 1), (71, 2),
            (72, 1), (69, 1), (65, 1), (68, 1),                 (69, 4),
        ],
    },
}


def render(tune, seconds, seed):
    spec = TUNES[tune]
    bpm = spec["bpm"]; beat = 60.0 / bpm; bar = 4 * beat
    n_bars = len(spec["chords"])
    loop_len = n_bars * bar
    loops = max(1, int(np.ceil(seconds / loop_len))) if seconds else 1
    total = loop_len * loops + 0.5
    buf = np.zeros(int(total * SR) + SR, np.float32)
    sixteenth = beat / 4

    for lp in range(loops):
        base = lp * loop_len
        # bass (triangle) + arpeggio (pulse 25%) + drums, per bar
        for bi, (root, tones) in enumerate(spec["chords"]):
            t0 = base + bi * bar
            # bass: root on beats 1 & 3, an octave below the arpeggio root
            for bo in (0, 2):
                add(buf, triangle(midi_freq(root), beat * 1.4, vol=0.5), t0 + bo * beat)
            # arpeggio harmony: fast 16th cycling through chord tones (classic chiptune)
            for s in range(16):
                n = tones[s % len(tones)]
                add(buf, pulse(midi_freq(n + 12), sixteenth * 0.95, duty=0.25, vol=0.11), t0 + s * sixteenth)
            # drums: kick on 1&3, snare(noise) on 2&4, hat every 8th
            add(buf, noise(0.12, vol=0.5, lp=40, seed=seed + bi), t0)                  # kick (lowpassed)
            add(buf, noise(0.12, vol=0.5, lp=40, seed=seed + bi + 99), t0 + 2 * beat)
            add(buf, noise(0.10, vol=0.35, lp=3, seed=seed + bi + 7), t0 + beat)        # snare
            add(buf, noise(0.10, vol=0.35, lp=3, seed=seed + bi + 8), t0 + 3 * beat)
            for h in range(8):
                add(buf, noise(0.03, vol=0.10, lp=1, seed=seed + h), t0 + h * 0.5 * beat)
        # lead melody (pulse 50%, with vibrato) across the 8 bars
        tcur = base
        for (n, b) in spec["lead"]:
            add(buf, pulse(midi_freq(n), beat * b * 0.96, duty=0.5, vol=0.26, vib=0.006), tcur)
            tcur += beat * b

    # trim + fades + normalize -3 dBFS, duplicate to stereo
    cut = int((loop_len * loops) * SR)
    mono = buf[:cut]
    f = int(0.04 * SR)
    mono[:f] *= np.linspace(0, 1, f, dtype=np.float32)
    mono[-int(0.3 * SR):] *= np.linspace(1, 0, int(0.3 * SR), dtype=np.float32)
    peak = max(np.abs(mono).max(), 1e-6)
    mono *= 10 ** (-3 / 20) / peak
    return mono, mono.copy()


def write_wav(path, L, R):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    inter = np.empty((len(L) * 2,), np.float32)
    inter[0::2] = L; inter[1::2] = R
    pcm = (np.clip(inter, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--tune", default="adventure", choices=list(TUNES))
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    L, R = render(a.tune, a.seconds, a.seed)
    write_wav(a.output, L, R)
    print(f"[OK] chiptune '{a.tune}' (original, copyright-clean) -> {a.output}  ({len(L)/SR:.1f}s)")


if __name__ == "__main__":
    main()
