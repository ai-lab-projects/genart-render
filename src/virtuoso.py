"""Virtuoso / orchestra engine — PUBLIC-DOMAIN showpieces + a machine-only 'impossible' etude,
rendered to (1) real audio via the GM SoundFont and (2) a Synthesia-style PIANO-ROLL video so the
note density / superhuman speed is visible. All compositions are PD (Beethoven, Beethoven's Ode to
Joy) or our own; the GM bank is freely licensed -> copyright-clean, safe to publish.

Note-event format (shared by audio + visuals): (t_sec, midi, dur_sec, vel, track).
Tracks map to GM programs + roll colors.

Pieces:
  moonlight3  — Beethoven, Moonlight Sonata mvt.3 'Presto agitato' (the rocketing arpeggios) — piano
  ode         — Beethoven, 'Ode to Joy' — small orchestra (strings / horn / bass / timpani)
  impossible  — our machine-only etude: both hands sweeping the whole keyboard at ~24 notes/sec

CLI:
  python virtuoso.py --piece moonlight3 --wav out.wav            # audio only
  python virtuoso.py --piece impossible --wav a.wav --roll v.mp4 # audio + piano-roll video
"""
from __future__ import annotations
import argparse
import os
import wave
from pathlib import Path

import numpy as np
import tinysoundfont

SR = 44100
# soundfont dir: env override (for the standalone CI render repo) else the ai_business assets_sf
_SF_DIR = Path(os.environ.get("GENART_SF_DIR", str(Path(__file__).resolve().parents[3] / "assets_sf")))
# prefer the better-sounding banks: Salamander piano (if present) -> GeneralUser GS -> FluidR3 GM
SF2 = next((str(_SF_DIR / n) for n in ("sala.sf2", "gu.sf2", "gm.sf2") if (_SF_DIR / n).exists()),
           str(_SF_DIR / "gm.sf2"))

# track -> (GM program, roll color)
TRACKS = {
    "piano":   (0,  (120, 200, 255)),
    "piano_l": (0,  (255, 170, 120)),
    "strings": (48, (140, 230, 170)),
    "horn":    (60, (255, 210, 110)),
    "bass":    (43, (200, 150, 255)),
    "timp":    (47, (240, 120, 120)),
}
TRACK_IDX = {k: i for i, k in enumerate(TRACKS)}


# ---------------------------------------------------------------- pieces
def moonlight3(seed=3):
    """Presto agitato: rising broken-chord 'rockets' in 16ths + two sforzando chord stabs."""
    rng = np.random.default_rng(seed)
    bpm = 168.0; b = 60.0 / bpm; six = b / 4
    ev = []; t = 0.0
    # harmony per 2-bar cell (root triads, C# minor world): C#m, then dominant-ish, etc.
    cells = [
        [49, 52, 56],   # C#m  (C# E G#)
        [49, 52, 56],
        [44, 48, 51],   # G#-ish (G# C D#)  approx dominant tension
        [49, 52, 56],
        [42, 45, 49],   # F#m
        [44, 47, 51],
        [49, 52, 56],
        [49, 52, 56],
    ]
    for cell in cells:
        # rocket: arpeggiate the triad upward across ~3 octaves in continuous 16ths (16 notes)
        notes = []
        base = cell
        for octv in range(3):
            for n in base:
                notes.append(n + 12 * octv + 12)     # start an octave up
        notes = notes[:14]
        for k, n in enumerate(notes):
            ev.append((t + k * six, n, six * 1.6, 70, "piano"))
        t += 14 * six
        # two sforzando chord stabs (the famous "ta-ta")
        top = cell[0] + 36
        chord = [cell[0] + 24, cell[1] + 24, cell[2] + 24, top]
        for s in range(2):
            for n in chord:
                ev.append((t + s * b * 0.5, n, b * 0.35, 104, "piano"))
        t += b
    return ev, t


def ode(seed=3):
    """Ode to Joy — strings melody, horn harmony a third below, contrabass roots, timpani on downbeats."""
    bpm = 108.0; b = 60.0 / bpm
    # melody (C major), (scale midi, beats)
    mel = [
        (64,1),(64,1),(65,1),(67,1), (67,1),(65,1),(64,1),(62,1),
        (60,1),(60,1),(62,1),(64,1), (64,1.5),(62,0.5),(62,2),
        (64,1),(64,1),(65,1),(67,1), (67,1),(65,1),(64,1),(62,1),
        (60,1),(60,1),(62,1),(64,1), (62,1.5),(60,0.5),(60,2),
    ]
    ev = []; t = 0.0
    for (n, d) in mel:
        ev.append((t, n, b * d * 0.98, 80, "strings"))
        ev.append((t, n - 4, b * d * 0.98, 60, "horn"))            # harmony a third under
        # bass root (nearest C/G under the melody) on each beat
        root = 36 if n in (64, 65, 67, 62, 60) else 43
        ev.append((t, root, b * d * 0.98, 78, "bass"))
        if abs(t / b - round(t / b)) < 1e-3 and int(round(t / b)) % 2 == 0:
            ev.append((t, 36, b * 0.4, 90, "timp"))
        t += b * d
    return ev, t


def impossible(seed=5):
    """Machine-only etude: two independent hands sweeping the full 88-key range in fast 16ths,
    crossing and cascading, with periodic full-range chord blasts. ~24 notes/sec — unplayable by hands."""
    rng = np.random.default_rng(seed)
    bpm = 150.0; b = 60.0 / bpm; six = b / 4
    scale = [0, 2, 4, 5, 7, 9, 11]                  # C major (clean cascades)
    def snap(m):
        pc = m % 12
        best = min(scale, key=lambda s: abs(s - pc))
        return m - pc + best
    ev = []; t = 0.0; bars = 20
    for bar in range(bars):
        # right hand: ascending then descending sweep, octave drift each bar
        lo = 60 + (bar % 4) * 2
        for k in range(16):
            phase = k / 16.0
            tri = 1 - abs(2 * phase - 1)             # 0->1->0 sweep
            n = snap(int(lo + tri * 36 + 12 * np.sin(bar * 0.7)))
            ev.append((t + k * six, n, six * 1.3, 72, "piano"))
        # left hand: counter-sweep (descending then ascending), lower register, offset by an 8th
        hi = 52 - (bar % 3) * 2
        for k in range(16):
            phase = k / 16.0
            tri = abs(2 * phase - 1)
            n = snap(int(hi - tri * 30 - 12 * np.cos(bar * 0.5)))
            ev.append((t + k * six + six * 0.5, max(24, n), six * 1.3, 64, "piano_l"))
        # every 4th bar: a thunderous full-range chord blast (10 notes at once)
        if bar % 4 == 3:
            for n in [36, 43, 48, 55, 60, 64, 67, 72, 76, 84]:
                ev.append((t + 3 * b, n, b * 0.9, 110, "piano"))
        t += 4 * six * 4   # 16 sixteenths = 4 beats = 1 bar
    return ev, t


def bumblebee(seed=3):
    """Rimsky-Korsakov - Flight of the Bumblebee: continuous chromatic 16ths (the 'buzz'). PD.
    Famous as a virtuoso SPEED showpiece -> ideal for a superhuman-tempo machine rendition."""
    bpm = 150.0; b = 60.0 / bpm; six = b / 4
    seq = []
    for rep in range(2):                       # phrase A: long chromatic descents
        for k in range(16):
            seq.append(88 - k)
    center = 76                                 # phrase B: oscillating buzz (chromatic wiggles)
    for k in range(32):
        seq.append(center + (k % 6 - 3))
    for k in range(16):                         # phrase C: chromatic ascent
        seq.append(64 + k)
    for rep in range(2):                        # phrase D: turn figures
        for k in range(16):
            seq.append(76 - (k % 8))
    for k in range(16):                         # tail: chromatic descent + land
        seq.append(80 - k)
    ev = []; t = 0.0
    for n in seq:
        ev.append((t, int(np.clip(n, 40, 96)), six * 1.2, 78, "piano"))
        t += six
    return ev, t


def heroic(seed=3):
    """Chopin - Polonaise 'Heroique' Op.53, MAIN THEME (my transcription attempt -> ear-check).
    Grand march tune in Ab major with the polonaise rhythm + booming octave bass."""
    bpm = 138.0; b = 60.0 / bpm
    AB = 68                                      # Ab4
    # main theme melody (top line), (midi, beats) — recognizable contour of the grand A theme
    mel = [
        (75, 0.5), (75, 0.25), (77, 0.25), (80, 1), (79, 0.5), (77, 0.5),
        (75, 0.5), (73, 0.25), (75, 0.25), (77, 1), (75, 1),
        (72, 0.5), (75, 0.25), (77, 0.25), (80, 1), (84, 0.5), (82, 0.5),
        (80, 0.5), (79, 0.25), (77, 0.25), (75, 1.5), (73, 0.5),
        (75, 0.5), (75, 0.25), (77, 0.25), (80, 1), (79, 0.5), (77, 0.5),
        (75, 0.5), (77, 0.25), (79, 0.25), (80, 2),
    ]
    ev = []; t = 0.0
    bar = 3 * b                                   # 3/4
    # bass: octave roots on the polonaise rhythm (1, &-of-1, 2, 3)
    n_bars = int(sum(d for _, d in mel) * b / bar) + 1
    for bb in range(n_bars):
        t0 = bb * bar
        root = AB - 12 if bb % 2 == 0 else AB - 12 + 3   # Ab / Cb-ish alternation for drive
        for onset in (0.0, 0.5 * b, b, 2 * b):
            ev.append((t0 + onset, root, b * 0.45, 92, "piano_l"))
            ev.append((t0 + onset, root + 12, b * 0.45, 80, "piano_l"))
    for (n, d) in mel:
        ev.append((t, n, b * d * 0.95, 96, "piano"))     # melody in octaves = grand
        ev.append((t, n - 12, b * d * 0.95, 78, "piano"))
        t += b * d
    span = max(t, n_bars * bar)
    return ev, span


PIECES = {"moonlight3": moonlight3, "ode": ode, "impossible": impossible,
          "bumblebee": bumblebee, "heroic": heroic}


# ---------------------------------------------------------------- audio
def render_audio(events, span, seed):
    rng = np.random.default_rng(seed)
    synth = tinysoundfont.Synth(samplerate=SR, gain=0.55)
    sfid = synth.sfload(SF2)
    used = sorted({tr for *_, tr in events}, key=lambda k: TRACK_IDX[k])
    chan = {}
    for i, tr in enumerate(used):
        prog, _ = TRACKS[tr]
        synth.program_select(i, sfid, 0, prog)
        chan[tr] = i
    seq = []
    for (t, n, dur, vel, tr) in events:
        jt = rng.uniform(-0.006, 0.006)
        v = int(np.clip(vel + rng.integers(-5, 6), 24, 120))
        seq.append((max(0, t + jt), 1, chan[tr], int(n), v))
        seq.append((max(0, t + jt) + dur, 0, chan[tr], int(n), 0))
    seq.sort(key=lambda e: e[0])
    chunks = []; cursor = 0.0
    for (t, kind, ch, n, v) in seq:
        gap = int((t - cursor) * SR)
        if gap > 0:
            chunks.append(np.frombuffer(synth.generate(gap), dtype=np.float32)); cursor = t
        synth.noteon(ch, n, v) if kind == 1 else synth.noteoff(ch, n)
    chunks.append(np.frombuffer(synth.generate(int(3.0 * SR)), dtype=np.float32))
    arr = np.concatenate(chunks).reshape(-1, 2)
    L, R = arr[:, 0].copy(), arr[:, 1].copy()
    fo = int(2.0 * SR)
    for ch in (L, R):
        ch[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
    g = 10 ** (-3 / 20) / peak
    L, R = L * g, R * g
    try:
        from audio_fx import reverb
        L, R = reverb(L, R, SR, wet=0.24)
    except Exception:
        pass
    return L, R


def write_wav(path, L, R):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    inter = np.empty((len(L) * 2,), np.float32)
    inter[0::2] = L; inter[1::2] = R
    pcm = (np.clip(inter, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- piano-roll video
def render_roll(events, span, out, title, fps=30):
    from PIL import Image, ImageDraw, ImageFilter
    import imageio.v2 as imageio
    W, H = 1280, 720
    KEY_H = 90; LOOK = 2.2                       # seconds of lookahead visible above the keyboard
    LOMIDI, HIMIDI = 21, 108                     # full 88 keys
    span_k = HIMIDI - LOMIDI
    def kx(m): return (m - LOMIDI) / span_k * W
    kw = W / span_k
    roll_top = 0; roll_bot = H - KEY_H
    # precompute notes/sec for the "speed" readout
    nfrm = int((span + 2.0) * fps)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=8)
    # bucket events for quick per-frame lookup is overkill; just iterate (counts are modest)
    ev = events
    try:
        for fi in range(nfrm):
            t = fi / fps
            im = Image.new("RGB", (W, H), (10, 12, 20))
            d = ImageDraw.Draw(im)
            # falling notes within [t, t+LOOK]
            active = set()
            for (et, n, dur, vel, tr) in ev:
                if et + dur < t or et > t + LOOK:
                    continue
                col = TRACKS[tr][1]
                y_on = roll_bot - (et - t) / LOOK * (roll_bot - roll_top)         # onset (lower on screen)
                y_off = roll_bot - (et + dur - t) / LOOK * (roll_bot - roll_top)  # release (higher up)
                top = max(roll_top, min(y_on, y_off)); bot = min(roll_bot, max(y_on, y_off))
                if bot - top < 1:
                    continue
                x0 = kx(n) + 1; x1 = kx(n) + kw - 1
                d.rectangle([x0, top, x1, bot], fill=col, outline=(255, 255, 255))
                if y_off <= roll_bot <= y_on:
                    active.add(n)
            # keyboard
            d.rectangle([0, roll_bot, W, H], fill=(24, 26, 34))
            for m in range(LOMIDI, HIMIDI):
                x = kx(m)
                black = (m % 12) in (1, 3, 6, 8, 10)
                base = (40, 44, 56) if black else (220, 224, 232)
                if m in active:
                    base = (120, 220, 255)
                d.rectangle([x + 1, roll_bot + 2, x + kw - 1, H - 2], fill=base)
            d.line([0, roll_bot, W, roll_bot], fill=(90, 120, 200), width=3)   # strike line
            d.text((22, 18), title, fill=(235, 240, 255))
            # live note-rate readout (sells 'impossible speed')
            window = [1 for (et, *_3) in ev if t - 1.0 <= et <= t]
            d.text((22, 44), f"{len(window)} notes/sec", fill=(150, 200, 255))
            bloom = im.filter(ImageFilter.GaussianBlur(3))
            writer.append_data(np.clip(np.asarray(im, np.float32) + 0.22 * np.asarray(bloom, np.float32), 0, 255).astype(np.uint8))
    finally:
        writer.close()
    print(f"[OK] piano-roll -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", required=True, choices=list(PIECES))
    ap.add_argument("--wav", required=True)
    ap.add_argument("--roll", default=None, help="also render a piano-roll mp4")
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    ev, span = PIECES[a.piece]()
    if a.loops > 1:
        base = list(ev); ev = []
        for lp in range(a.loops):
            ev += [(t + lp * span, n, dur, vel, tr) for (t, n, dur, vel, tr) in base]
        span = span * a.loops
    L, R = render_audio(ev, span, a.seed)
    write_wav(a.wav, L, R)
    print(f"[OK] {a.piece} audio -> {a.wav} ({len(L)/SR:.1f}s, {len(ev)} notes)")
    if a.roll:
        titles = {"moonlight3": "Beethoven - Moonlight Sonata mvt.3 (Presto agitato)",
                  "ode": "Beethoven - Ode to Joy (orchestra)",
                  "impossible": "Machine Etude - music no human can play",
                  "bumblebee": "Rimsky-Korsakov - Flight of the Bumblebee (superhuman tempo)",
                  "heroic": "Chopin - Heroic Polonaise Op.53 (main theme)"}
        render_roll(ev, span, a.roll, titles[a.piece])


if __name__ == "__main__":
    main()
