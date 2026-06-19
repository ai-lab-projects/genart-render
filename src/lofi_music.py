"""Procedural lo-fi hip-hop track generator (numpy, no samples, no copyright).

Why lo-fi: research shows lo-fi / chill beats are the dominant proven 'focus/study/sleep' audio
demand (user 2026-06-05). 100% synthesized from scratch -> monetization-safe + Spotify-distributable.

Synthesizes: jazzy Rhodes-ish 7th chords + sine bass + boom-bap drums (kick/snare/hat) +
vinyl crackle + tape hiss, gently low-passed (the 'lo-fi' warmth) and swung. ~75 BPM, no lyrics.

CLI:
  python lofi_music.py --output outputs/lofi.wav --seconds 60 --seed 1 --key c
"""
from __future__ import annotations
import argparse
import wave
from pathlib import Path

import numpy as np

SR = 44100


def midi_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, float) - 69.0) / 12.0)


def box_lowpass(x, win):
    """Fast vectorized boxcar low-pass (cumsum). win larger = duller (lower cutoff)."""
    if win <= 1:
        return x
    k = np.ones(win, np.float32) / win
    return np.convolve(x, k, mode="same").astype(np.float32)


def env_ad(n, attack, decay):
    """Attack-decay envelope of length n samples."""
    e = np.ones(n, np.float32)
    a = max(1, int(attack * SR))
    d = max(1, int(decay * SR))
    e[:a] = np.linspace(0, 1, a, dtype=np.float32)
    tail = np.exp(-np.linspace(0, 5, max(1, n - a), dtype=np.float32))
    e[a:] = tail[: n - a]
    return e


def rhodes(freq, dur, rng):
    """Soft electric-piano-ish tone: fundamental + a few decaying partials, slight detune."""
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    det = 1.0 + rng.uniform(-0.0015, 0.0015)
    parts = [(1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (6, 0.05)]
    sig = np.zeros(n, np.float32)
    for h, amp in parts:
        sig += amp * np.sin(2 * np.pi * freq * det * h * t)
    sig *= env_ad(n, 0.012, dur * 0.9)
    # bell-ish FM shimmer, very light
    sig += 0.06 * np.sin(2 * np.pi * freq * 4.01 * t) * np.exp(-t * 6)
    return sig


def sine_bass(freq, dur):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float32) / SR
    sig = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    return sig * env_ad(n, 0.008, dur * 0.8)


def kick():
    n = int(0.32 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = 110 * np.exp(-t * 28) + 45            # pitch drop
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return sig * np.exp(-t * 9)


def snare(rng):
    n = int(0.22 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    noise = rng.standard_normal(n).astype(np.float32)
    body = box_lowpass(noise, 6) - box_lowpass(noise, 40)   # band-ish
    tone = 0.3 * np.sin(2 * np.pi * 180 * t)
    return (body * 1.2 + tone) * np.exp(-t * 18)


def hat(rng, open_=False):
    n = int((0.16 if open_ else 0.05) * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    hp = noise - box_lowpass(noise, 8)        # high-pass
    t = np.arange(n, dtype=np.float32) / SR
    return hp * np.exp(-t * (10 if open_ else 45))


def add(buf, sig, at, gain=1.0):
    i = int(at * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i] * gain


# ii-V-I-vi style 7th chords (semitone offsets from key root), root + chord tones up an octave
PROG = [
    (2, [2, 5, 9, 12]),    # Dm7 (in C)
    (7, [7, 11, 14, 17]),  # G7
    (0, [0, 4, 7, 11]),    # Cmaj7
    (9, [9, 12, 16, 19]),  # Am7
]


def render(seconds, seed, key_root, style="classic"):
    rng = np.random.default_rng(seed)
    bpm = 68.0 if style == "cafe" else 75.0     # cafe = slower, gentler
    beat = 60.0 / bpm
    bar = 4 * beat
    eighth = beat / 2
    swing = 0.06 * beat                       # delay offbeat 8ths -> swung lo-fi feel
    n_total = int(seconds * SR) + SR
    L = np.zeros(n_total, np.float32)
    R = np.zeros(n_total, np.float32)

    n_bars = int(seconds / bar) + 1
    for b in range(n_bars):
        t0 = b * bar
        root, tones = PROG[b % len(PROG)]
        base = key_root + 48                  # chord register (~C4)
        # chord stab on beat 1 and the 'and' of 3 (held-ish)
        for onset, dur in [(0.0, bar * 0.55), (beat * 2 + eighth, bar * 0.4)]:
            for k, semi in enumerate(tones):
                f = float(midi_hz(base + semi))
                tone = rhodes(f, dur, rng) * (0.16 - 0.012 * k)
                pan = 0.5 + (k - 1.5) * 0.08
                add(L, tone, t0 + onset, gain=(1 - pan))
                add(R, tone, t0 + onset, gain=pan)
        # bass on beats 1 and 3 (root, low)
        for _bi, onset in [(0, 0.0), (1, beat * 2)]:
            f = float(midi_hz(key_root + 36 + root))
            add(L, sine_bass(f, beat * 1.6), t0 + onset, 0.5)
            add(R, sine_bass(f, beat * 1.6), t0 + onset, 0.5)
        # melody: gentle single notes over the chord (higher register) — prominent for cafe, sparse for classic
        mel_onsets = [(0.0, beat * 0.9), (beat * 1.5 + swing, beat * 0.7),
                      (beat * 2.5, beat * 0.9), (beat * 3.5 + swing, beat * 0.6)]
        p_play = 0.8 if style == "cafe" else 0.35
        for j, (onset, dur) in enumerate(mel_onsets):
            if rng.random() < (p_play if j in (0, 2) else p_play * 0.65):
                semi = int(tones[rng.integers(0, len(tones))]) + 12      # an octave above the chord
                if rng.random() < 0.3:
                    semi += int(rng.choice([-1, 2]))                     # passing color tone
                note = rhodes(float(midi_hz(base + semi)), dur, rng) * (0.15 if style == "cafe" else 0.11)
                add(L, note, t0 + onset, 0.5); add(R, note, t0 + onset, 0.5)
        # drums: cafe = brushy/soft (no backbeat snare), classic = boom-bap
        for s in range(8):
            sw = swing if s % 2 == 1 else 0.0
            at = t0 + s * eighth + sw
            if style == "cafe":
                if s == 0:
                    add(L, kick(), at, 0.5); add(R, kick(), at, 0.5)
                if s % 2 == 1:
                    h = hat(rng, open_=False); vel = rng.uniform(0.05, 0.10)
                    add(L, h, at, vel); add(R, h, at, vel * 0.9)
            else:
                if s in (0, 6):
                    add(L, kick(), at, 0.9); add(R, kick(), at, 0.9)
                if s == 4:
                    snr = snare(rng); add(L, snr, at, 0.5); add(R, snr, at, 0.5)
                if s % 2 == 1 or s in (2,):
                    h = hat(rng, open_=(s == 7)); vel = rng.uniform(0.12, 0.22)
                    add(L, h, at, vel); add(R, h, at, vel * 0.9)

    # vinyl crackle: sparse pops + tape hiss
    n = len(L)
    pops = rng.random(n) < 0.0009
    crackle = (pops * rng.uniform(-1, 1, n)).astype(np.float32)
    crackle = box_lowpass(crackle, 3) * 0.10
    hiss = box_lowpass(rng.standard_normal(n).astype(np.float32), 5) * 0.006
    L += crackle + hiss
    R += box_lowpass(crackle, 2) + hiss

    # lo-fi warmth: roll off highs (boxcar low-pass) + soft saturation
    L = box_lowpass(L, 4); R = box_lowpass(R, 4)
    L = np.tanh(L * 1.2); R = np.tanh(R * 1.2)

    # trim to length, fades, normalize to -3 dBFS peak
    cut = int(seconds * SR)
    L, R = L[:cut], R[:cut]
    fade = int(2.5 * SR)
    for ch in (L, R):
        ch[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        ch[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
    g = 10 ** (-3 / 20) / peak
    return (L * g), (R * g)


def write_wav(path, L, R):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    inter = np.empty((len(L) * 2,), np.float32)
    inter[0::2] = L; inter[1::2] = R
    pcm = np.clip(inter, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--key", default="c")
    ap.add_argument("--style", default="classic", choices=["classic", "cafe"])
    a = ap.parse_args()
    roots = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9}
    L, R = render(a.seconds, a.seed, roots.get(a.key.lower(), 0), a.style)
    write_wav(a.output, L, R)
    print(f"[OK] lo-fi {a.seconds:.0f}s @ {SR}Hz stereo -> {a.output}")


if __name__ == "__main__":
    main()
