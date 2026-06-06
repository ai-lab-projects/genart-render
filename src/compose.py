"""Score-first original composer: decide a PLAN (key / form / progression) -> emit a symbolic SCORE
(note events) -> render through the sampled engine + piano-roll. Separating composition from synthesis
(vs hardcoding audio) lets us iterate the music as data and apply theory deliberately.

All output is our OWN composition (theory applied generatively, not copied) = copyright-clean, publishable.

Styles: cafe / lofi / jazz / minimal / atonal

  python compose.py --style cafe --wav out.wav --roll out.mp4 --bars 16
"""
from __future__ import annotations
import argparse

import numpy as np

import virtuoso as V
import midi_render as MR

# chord interval recipes (semitones from root)
CH = {
    "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "dom7": [0, 4, 7, 10],
    "maj9": [0, 4, 7, 11, 14], "min9": [0, 3, 7, 10, 14], "dom9": [0, 4, 7, 10, 14],
    "m7b5": [0, 3, 6, 10], "dim7": [0, 3, 6, 9], "maj": [0, 4, 7], "min": [0, 3, 7],
}
MAJ = [0, 2, 4, 5, 7, 9, 11]
DOR = [0, 2, 3, 5, 7, 9, 10]

# (style) -> dict(key, bpm, swing, form, prog[(root, chordtype)], programs{ch:gm}, drums)
STYLES = {
    "cafe":  dict(key=0,  bpm=76,  swing=0.34, prog=[(0,"maj9"),(9,"min7"),(2,"min9"),(7,"dom9")],
                  programs={0:0, 1:32, 2:11}, drums=True, lead_oct=12, desc="jazzy lo-fi cafe piano: I-vi-ii-V, soft swing, vibraphone tint"),
    "lofi":  dict(key=5,  bpm=70,  swing=0.30, prog=[(5,"maj7"),(4,"min7"),(2,"min7"),(0,"maj7")],
                  programs={0:0, 1:32, 2:4},  drums=True, lead_oct=12, desc="laid-back lo-fi: ii-V-ish loop in F, e-piano pad, soft kick"),
    "jazz":  dict(key=0,  bpm=128, swing=0.36, prog=[(2,"min7"),(7,"dom9"),(0,"maj7"),(9,"dom7"),
                                                     (2,"min7"),(7,"dom9"),(0,"maj7"),(0,"maj7")],
                  programs={0:0, 1:32, 2:66}, drums=True, lead_oct=12, walk=True, desc="straight-ahead jazz: ii-V-I + turnaround, walking bass, ride, sax lead"),
    "minimal": dict(key=9, bpm=104, swing=0.0, prog=[(9,"min"),(4,"min"),(5,"maj"),(7,"maj")],
                    programs={0:0, 1:48, 2:0}, drums=False, lead_oct=12, additive=True,
                    desc="Glass/Reich-style minimalism: steady 8th arpeggio cells, additive layering, strings pad"),
    "atonal": dict(key=0, bpm=60, swing=0.0, prog=[(0,"maj"),(0,"maj"),(0,"maj"),(0,"maj")],
                   programs={0:0, 1:48, 2:48}, drums=False, lead_oct=12, atonal=True,
                   desc="contemporary/atonal: free chromatic clusters, sparse pointillism, no functional harmony"),
    "waltz": dict(key=0, bpm=138, swing=0.0, meter=3, prog=[(0,"maj"),(9,"min"),(5,"maj"),(7,"dom7")],
                  programs={0:9, 1:0, 2:0}, drums=False, lead_oct=12, waltz=True,
                  desc="warm 'honobono' waltz: 3/4 oom-pah-pah, glockenspiel melody over real piano. original, publishable"),
    "minimal_e": dict(key=9, bpm=104, swing=0.0, prog=[(9,"min"),(4,"min"),(5,"maj"),(7,"maj")],
                      programs={0:0, 1:4, 2:4}, drums=False, lead_oct=12, additive=True,
                      desc="minimalism with an electric-piano (electone-ish) lower voice + real-piano arpeggio lead"),
    "chaos": dict(key=0, bpm=126, swing=0.0, prog=[(0,"maj")]*4,
                  programs={0:0, 1:48, 2:0}, drums=False, lead_oct=12, chaos=True,
                  desc="dense chaotic cluster-storm: rapid random chromatic across the whole keyboard, overlapping voices, wild (experimental, not BGM)"),
}


def plan(style):
    s = STYLES[style]
    names = {0:"C",1:"Db",2:"D",3:"Eb",4:"E",5:"F",6:"Gb",7:"G",8:"Ab",9:"A",10:"Bb",11:"B"}
    prog = " ".join(f"{names[r%12]}{t}" for r, t in s["prog"])
    return (f"PLAN [{style}] key={names[s['key']]} bpm={s['bpm']} swing={s['swing']}\n"
            f"  form: 4-bar progression looped with melodic variation\n"
            f"  progression: {prog}\n  idea: {s['desc']}")


def compose(style, seed, bars):
    s = STYLES[style]
    rng = np.random.default_rng(seed)
    bpm = s["bpm"]; b = 60.0 / bpm; bar = s.get("meter", 4) * b
    key = s["key"] + 60
    prog = s["prog"]
    sw = s.get("swing", 0.0)
    ev = []
    def swung(onset_beats):
        # delay the off-beat 8ths for swing feel
        frac = onset_beats % 1.0
        if abs(frac - 0.5) < 1e-3:
            return onset_beats + sw * 0.5
        return onset_beats

    for bi in range(bars):
        t0 = bi * bar
        root, ctype = prog[bi % len(prog)]
        chord = [key + root + iv for iv in CH[ctype]]

        if s.get("waltz"):
            # 3/4 oom-pah-pah: bass on beat 1, chord on beats 2 & 3, gentle stepwise melody
            scale = [key + root + i for i in (DOR if "min" in ctype else MAJ)]
            ev.append((t0, chord[0] - 24, b * 0.9, 72, 1))                      # bass (real piano low)
            for beat in (1, 2):                                                 # pah-pah chords
                for k, n in enumerate(chord[:3]):
                    ev.append((t0 + beat * b, n - 12, b * 0.8, 46 - 3 * k, 2))
            prev = chord[2] + s["lead_oct"]
            for onset, dur in [(0.0, 1.0), (1.0, 0.5), (1.5, 0.5), (2.0, 1.0)]:  # warm melody (glockenspiel)
                if rng.random() < (0.9 if onset in (0.0, 2.0) else 0.6):
                    pool = [c + s["lead_oct"] for c in chord] + [x + s["lead_oct"] for x in scale]
                    n = int(min(pool, key=lambda x: abs(x - prev) + rng.uniform(0, 4)))
                    ev.append((t0 + onset * b, n, b * dur, 62, 0)); prev = n
            continue
        if s.get("chaos"):
            # storm: many fast random chromatic notes across the full keyboard, overlapping voices
            for _ in range(int(rng.integers(16, 24))):
                onset = rng.uniform(0, 4) * b
                n = 36 + int(rng.integers(0, 52))
                ev.append((t0 + onset, n, b * rng.uniform(0.12, 0.6), int(rng.integers(58, 104)), int(rng.choice([0, 2]))))
            if rng.random() < 0.8:                              # rolling chromatic run (a "scramble")
                base = 40 + int(rng.integers(0, 40)); step = int(rng.choice([1, -1, 2, -2]))
                for k in range(int(rng.integers(7, 13))):
                    ev.append((t0 + rng.uniform(0, 4) * b, base + step * k, b * 0.22, 76, 1))
            continue
        if s.get("atonal"):
            # sparse pointillism: random chromatic notes + occasional cluster
            for _ in range(rng.integers(3, 6)):
                onset = rng.uniform(0, 4) * b
                n = 48 + int(rng.integers(0, 36))
                ev.append((t0 + onset, n, b * rng.uniform(0.4, 1.6), int(rng.integers(40, 80)), 0))
            if rng.random() < 0.5:                              # tone cluster
                base = 52 + int(rng.integers(0, 20))
                for k in range(rng.integers(2, 4)):
                    ev.append((t0 + rng.uniform(0, 3) * b, base + k, b * 1.5, 55, 2))
            ev.append((t0, 36 + int(rng.integers(0, 12)), bar, 50, 1))   # low drone
            continue

        # --- chordal comp (track 2) ---
        comp_hits = [(0.0, bar * 0.9)] if s.get("additive") else [(0.0, bar*0.5), (2.5*b, bar*0.4)]
        for onset, dur in comp_hits:
            voicing = chord if not s.get("additive") else chord[:3]
            for k, n in enumerate(voicing):
                ev.append((t0 + onset, n - 12 + (0 if s.get("additive") else 0), dur, 46 - 3*k, 2))

        # --- bass (track 1) ---
        if s.get("walk"):                                       # walking quarter notes
            scale = [key + root + i for i in MAJ]
            for beat in range(4):
                bn = (chord[0] - 24) if beat == 0 else (rng.choice(scale) - 24)
                ev.append((t0 + beat * b, int(bn), b * 0.95, 74, 1))
        else:
            for onset in (0.0, 2 * b):
                ev.append((t0 + onset, chord[0] - 24, b * 1.7, 72, 1))

        # --- melody (track 0): chord tones + passing tones, style rhythm ---
        if s.get("additive"):
            # minimalism: steady 8th arpeggio of the triad, cell length grows over bars
            cell = (chord * 3)[: 4 + (bi % 4)]
            for j in range(8):
                n = cell[j % len(cell)] + 12
                ev.append((t0 + j * 0.5 * b, n, b * 0.5, 60, 0))
        else:
            scale = [key + root + i for i in (DOR if "min" in ctype else MAJ)]
            rhythm = [(0.0,0.9),(1.5,0.6),(2.5,0.9),(3.5,0.6)] if style != "jazz" else \
                     [(0.0,0.5),(0.5,0.5),(1.5,0.5),(2.0,0.9),(3.0,0.5),(3.5,0.5)]
            prev = chord[2] + s["lead_oct"]
            for onset, dur in rhythm:
                if rng.random() < (0.85 if onset in (0.0,2.5) else 0.6):
                    pool = [c + s["lead_oct"] for c in chord] + [x + s["lead_oct"] for x in scale]
                    n = int(min(pool, key=lambda x: abs(x - prev) + rng.uniform(0,5)))  # smooth-ish leading
                    if rng.random() < 0.2: n += int(rng.choice([-1, 1]))                # chromatic passing
                    ev.append((t0 + swung(onset) * b, n, b * dur, 60, 0)); prev = n

        # --- drums (channel 9) --- kick kept low in the mix (user: lower the low-freq drum)
        if s.get("drums"):
            kv, sv = (40, 56) if style == "lofi" else (52, 58) if style == "cafe" else (66, 70)
            ev.append((t0, 36, b*0.3, kv, 9))                   # kick beat 1 (quieter)
            ev.append((t0 + 2*b, 36, b*0.3, kv-8, 9))           # kick beat 3
            ev.append((t0 + 1*b, 38, b*0.3, sv, 9)); ev.append((t0 + 3*b, 38, b*0.3, sv, 9))  # snare 2&4
            for h in range(8):                                   # hats (swung)
                ev.append((t0 + swung(h*0.5) * b, 42, b*0.2, 50, 9))

    ev.sort(key=lambda e: e[0])
    span = max(e[0] + e[2] for e in ev)
    programs = dict(s["programs"])
    return ev, programs, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, choices=list(STYLES))
    ap.add_argument("--wav", required=True)
    ap.add_argument("--roll", default=None)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    print(plan(a.style))
    ev, programs, span = compose(a.style, a.seed, a.bars)
    print(f"[compose] {len(ev)} notes, {span:.1f}s")
    L, R = MR.render_gm_audio(ev, programs, span, a.seed)
    V.write_wav(a.wav, L, R)
    print(f"[OK] audio -> {a.wav}")
    if a.roll:
        tk = list(V.TRACKS)
        roll = [(t, n, d, vl, tk[c % len(tk)]) for (t, n, d, vl, c) in ev]
        V.render_roll(roll, span, a.roll, f"Original - {a.style} (composed score -> rendered)")


if __name__ == "__main__":
    main()
