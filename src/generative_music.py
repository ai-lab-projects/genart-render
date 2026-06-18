"""Generate clean procedural ambient music as loopable 16-bit stereo WAV.

This is intentionally self-contained: synthesis uses numpy, and WAV writing uses
the standard library. The output is suitable as quiet background music for long
visual loops where Content-ID cleanliness matters.
"""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np


SAMPLE_RATE = 44_100
PEAK_TARGET = 10 ** (-3.0 / 20.0)


NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def note_freq(note: str, octave: int) -> float:
    midi = (octave + 1) * 12 + NOTE_TO_SEMITONE[note]
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def chord_frequencies(root: str, quality: str, octave: int = 3) -> list[float]:
    root_f = note_freq(root, octave)
    third = 3 if quality == "minor" else 4
    semitones = [0, third, 7, 12, 19]
    return [root_f * (2.0 ** (s / 12.0)) for s in semitones]


def progression(key_mode: str) -> list[tuple[str, str]]:
    if key_mode == "major":
        return [("C", "major"), ("G", "major"), ("A", "minor"), ("F", "major")]
    return [("A", "minor"), ("F", "major"), ("C", "major"), ("G", "major")]


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def one_pole_lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sample_rate)
    out = np.empty_like(signal)
    z = np.zeros(signal.shape[1], dtype=np.float64)
    for i in range(signal.shape[0]):
        z += alpha * (signal[i] - z)
        out[i] = z
    return out


def feedback_delay(signal: np.ndarray, delay_samples: int, feedback: float, wet: float) -> np.ndarray:
    out = signal.copy()
    delayed = np.zeros(signal.shape[1], dtype=np.float64)
    for i in range(signal.shape[0]):
        dry = out[i].copy()
        out[i] += delayed * wet
        delayed = dry + delayed * feedback
        if i >= delay_samples:
            delayed = out[i - delay_samples] + delayed * feedback
    return out


def tiny_reverb(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    wet = np.zeros_like(signal)
    taps = [
        (0.113, 0.16, 0.13),
        (0.197, 0.13, 0.10),
        (0.331, 0.10, 0.08),
        (0.463, 0.08, 0.06),
    ]
    for delay_s, gain_l, gain_r in taps:
        d = int(delay_s * sample_rate)
        wet[d:, 0] += signal[:-d, 0] * gain_l
        wet[d:, 1] += signal[:-d, 1] * gain_r
    return np.clip(signal + wet, -2.0, 2.0)


def render_chord(
    duration: float,
    root: str,
    quality: str,
    rng: np.random.Generator,
    sample_rate: int,
) -> np.ndarray:
    n = max(1, int(round(duration * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    freqs = chord_frequencies(root, quality)
    chord = np.zeros((n, 2), dtype=np.float64)

    for voice, base_freq in enumerate(freqs):
        base_amp = [0.34, 0.22, 0.24, 0.16, 0.10][voice]
        for layer in range(4):
            cents = rng.uniform(-7.0, 7.0) + (layer - 1.5) * 2.0
            freq = base_freq * (2.0 ** (cents / 1200.0))
            phase = rng.uniform(0.0, 2.0 * math.pi)
            pan = rng.uniform(-0.35, 0.35)
            sine = np.sin(2.0 * math.pi * freq * t + phase)
            tri = (2.0 / math.pi) * np.arcsin(np.sin(2.0 * math.pi * freq * 0.5 * t + phase * 0.7))
            tone = 0.78 * sine + 0.22 * tri
            amp = base_amp * rng.uniform(0.16, 0.25)
            chord[:, 0] += tone * amp * math.sqrt(0.5 * (1.0 - pan))
            chord[:, 1] += tone * amp * math.sqrt(0.5 * (1.0 + pan))

    lfo_rate = rng.uniform(0.055, 0.13)
    lfo_phase = rng.uniform(0.0, 2.0 * math.pi)
    swell = 0.70 + 0.30 * (0.5 + 0.5 * np.sin(2.0 * math.pi * lfo_rate * t + lfo_phase))
    chord *= swell[:, None]

    edge = min(n // 3, int(3.0 * sample_rate))
    if edge > 1:
        env = np.ones(n, dtype=np.float64)
        env[:edge] = smoothstep(np.linspace(0.0, 1.0, edge))
        env[-edge:] = smoothstep(np.linspace(1.0, 0.0, edge))
        chord *= env[:, None]
    return chord


def synthesize(seconds: float, key_mode: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    prog = progression(key_mode)
    steps_per_cycle = len(prog)
    cycles = max(1, int(round(seconds / (12.0 * steps_per_cycle))))
    chord_duration = seconds / (cycles * steps_per_cycle)
    crossfade_s = min(4.0, max(0.25, seconds * 0.08), chord_duration * 0.45)
    xfade_n = int(round(crossfade_s * SAMPLE_RATE))

    chunks = []
    total_steps = cycles * steps_per_cycle
    for i in range(total_steps):
        root, quality = prog[i % steps_per_cycle]
        chunks.append(render_chord(chord_duration, root, quality, rng, SAMPLE_RATE))
    audio = np.vstack(chunks)
    target_n = int(round(seconds * SAMPLE_RATE))
    if audio.shape[0] < target_n:
        audio = np.pad(audio, ((0, target_n - audio.shape[0]), (0, 0)))
    audio = audio[:target_n]

    if xfade_n > 1 and target_n > xfade_n * 2:
        fade = smoothstep(np.linspace(0.0, 1.0, xfade_n))[:, None]
        audio[-xfade_n:] = audio[-xfade_n:] * (1.0 - fade) + audio[:xfade_n] * fade

    slow_t = np.arange(target_n, dtype=np.float64) / SAMPLE_RATE
    shimmer = 0.94 + 0.06 * np.sin(2.0 * math.pi * 0.037 * slow_t + rng.uniform(0.0, 2.0 * math.pi))
    audio *= shimmer[:, None]
    audio = one_pole_lowpass(audio, cutoff_hz=3_800.0, sample_rate=SAMPLE_RATE)
    audio = tiny_reverb(audio, SAMPLE_RATE)

    # RMS target approximates a calm -16 LUFS bed; peak is capped at -3 dBFS.
    rms = float(np.sqrt(np.mean(audio * audio)) + 1e-12)
    audio *= 10 ** (-16.5 / 20.0) / rms
    peak = float(np.max(np.abs(audio)) + 1e-12)
    if peak > PEAK_TARGET:
        audio *= PEAK_TARGET / peak
    return audio


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate loopable calm ambient procedural WAV.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--key", choices=["minor", "major"], default="minor")
    args = parser.parse_args()

    if args.seconds <= 1.0:
        parser.error("--seconds must be > 1.0")

    audio = synthesize(args.seconds, args.key, args.seed)
    write_wav(args.output, audio)
    peak_db = 20.0 * math.log10(float(np.max(np.abs(audio))) + 1e-12)
    rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(audio * audio))) + 1e-12)
    print(f"[OK] wrote {args.output} ({args.seconds:.2f}s, 44100Hz stereo 16-bit, peak {peak_db:.1f} dBFS, rms {rms_db:.1f} dBFS)")


if __name__ == "__main__":
    main()
