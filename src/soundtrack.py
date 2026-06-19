"""Ambient soundtrack = lo-fi + nature, by ROTATION over a small curated recipe list.

Strategy (user 2026-06-05): don't generate infinitely. Curate a handful of (lofi x nature)
recipes and rotate them per video (like the voice rotation for explainers). A few assets ->
many fresh-feeling combos, all vetted, all copyright-free (synthesized from scratch).

Each recipe = a lo-fi style/key/seed + a nature bed + a mix weight. `make(index, seconds, out)`
generates the lo-fi and nature at the requested length, mixes, masters, and writes a WAV — so we
store recipes, not huge audio files, and can render any length on demand.

CLI:
  python soundtrack.py --index 0 --seconds 840 --output track.wav
  python soundtrack.py --list
"""
from __future__ import annotations
import argparse
import wave
from pathlib import Path

import numpy as np

import lofi_music
import nature_sound

SR = 44100
_ROOTS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9}

# curated rotation, DATA-WEIGHTED (2026-06-05 research): rain dominates > thunder+fireplace > cafe;
# waves/stream weak for lo-fi. Each recipe = lofi (style/key/seed) + one or more nature 'beds'.
# Seed by this research; later PRUNE/REWEIGHT by our own YouTube retention data (KPI loop).
RECIPES = [
    {"name": "rainy-lofi",        "style": "classic", "key": "a", "seed": 4,
     "beds": [{"type": "rain", "seed": 3, "w": 0.30}]},                                     # #1 pairing (rain kept light, user)
    # cafe-piano-rain = cafe-style PIANO MUSIC (style=cafe) + rain — NOT crowd noise (this is what 'cafe piano' means)
    {"name": "cafe-piano-rain",   "style": "cafe",    "key": "d", "seed": 6,
     "beds": [{"type": "rain", "seed": 5, "w": 0.26}]},
    {"name": "storm-fireside",    "style": "classic", "key": "e", "seed": 14,
     "beds": [{"type": "rain", "seed": 9, "w": 0.34}, {"type": "thunder", "seed": 2, "w": 1.2},
              {"type": "fireplace", "seed": 7, "w": 1.0}]},     # thunder/fire boosted so they're audible under rain+lofi
    # cafe-crowd = cafe CROWD ambience (nature=cafe, hard to synth) — distinct from cafe-piano music
    {"name": "cafe-crowd",        "style": "classic", "key": "c", "seed": 11,
     "beds": [{"type": "cafe", "seed": 4, "w": 0.55}]},                                     # strong niche
    {"name": "fireside-piano",    "style": "cafe",    "key": "f", "seed": 8,
     "beds": [{"type": "fireplace", "seed": 3, "w": 0.55}, {"type": "rain", "seed": 6, "w": 0.30}]},
    {"name": "rainy-lofi-2",      "style": "classic", "key": "g", "seed": 21,
     "beds": [{"type": "rain", "seed": 11, "w": 0.30}]},
]


def make(index, seconds, output):
    r = RECIPES[index % len(RECIPES)]
    lL, lR = lofi_music.render(seconds, r["seed"], _ROOTS[r["key"]], r["style"])
    L = lL.copy(); R = lR.copy()
    for bed in r["beds"]:
        nL, nR = nature_sound.render(bed["type"], seconds, bed["seed"])
        w = np.float32(bed["w"])
        L = L + nL * w; R = R + nR * w
    # master: soft-limit + normalize to -3 dBFS (both layers were pre-faded/normalized)
    L = np.tanh(L * 1.05); R = np.tanh(R * 1.05)
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
    g = 10 ** (-3 / 20) / peak
    L, R = L * g, R * g
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    inter = np.empty((len(L) * 2,), np.float32)
    inter[0::2] = L; inter[1::2] = R
    pcm = (np.clip(inter, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(output), "w") as wv:
        wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR)
        wv.writeframes(pcm.tobytes())
    beds = "+".join(b["type"] for b in r["beds"])
    print(f"[OK] soundtrack #{index} '{r['name']}' ({r['style']} {r['key']} + {beds}) {seconds:.0f}s -> {output}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=840.0)
    ap.add_argument("--output")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, r in enumerate(RECIPES):
            beds = ", ".join(f"{b['type']}({b['w']})" for b in r["beds"])
            print(f"{i}: {r['name']} = lofi({r['style']},{r['key']}) + [{beds}]")
        return
    if not a.output:
        ap.error("--output required unless --list")
    make(a.index, a.seconds, a.output)


if __name__ == "__main__":
    main()
