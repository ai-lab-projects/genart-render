"""Audio post-FX (numpy only). Main one: convolution REVERB to add the natural tail / room ambience
that dry soundfont renders lack (user: 'no resonance, the sound cuts off abruptly').

We synthesize a room impulse response (exponentially-decaying, slightly-damped stereo noise) and
convolve the dry signal with it -> notes ring out and sit in a space instead of stopping dead.
"""
from __future__ import annotations
import numpy as np


def _smooth(x, w):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w, np.float32) / w, mode="same").astype(np.float32)


def _fftconv(a, ir):
    N = len(a) + len(ir) - 1
    F = np.fft.rfft(a, N) * np.fft.rfft(ir, N)
    return np.fft.irfft(F, N).astype(np.float32)


def reverb(L, R, sr=44100, decay=1.9, wet=0.26, predelay=0.018, damp=4, seed=0, peak=0.89):
    """Convolution reverb with a synthetic room IR. Returns (L, R) lengthened by the reverb tail."""
    rng = np.random.default_rng(seed)
    n = int(decay * sr)
    t = np.arange(n) / sr
    env = np.exp(-t * (6.0 / decay)).astype(np.float32)               # exponential decay
    irL = _smooth(rng.standard_normal(n).astype(np.float32), damp) * env
    irR = _smooth(rng.standard_normal(n).astype(np.float32), damp) * env
    pd = int(predelay * sr); z = np.zeros(pd, np.float32)
    irL = np.concatenate([z, irL]); irR = np.concatenate([z, irR])
    irL /= (np.sqrt((irL ** 2).sum()) + 1e-9)                         # unit-energy IR
    irR /= (np.sqrt((irR ** 2).sum()) + 1e-9)
    wetL = _fftconv(L, irL); wetR = _fftconv(R, irR)
    out_len = len(wetL)
    dryL = np.concatenate([L, np.zeros(out_len - len(L), np.float32)])
    dryR = np.concatenate([R, np.zeros(out_len - len(R), np.float32)])
    wetL *= (dryL.std() + 1e-9) / (wetL.std() + 1e-9)                 # match wet level to dry
    wetR *= (dryR.std() + 1e-9) / (wetR.std() + 1e-9)
    outL = (1 - wet) * dryL + wet * wetL
    outR = (1 - wet) * dryR + wet * wetR
    g = peak / max(np.abs(outL).max(), np.abs(outR).max(), 1e-9)
    return (outL * g).astype(np.float32), (outR * g).astype(np.float32)
