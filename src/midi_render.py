"""Render a real MIDI file (faithful notes) through our sampled-piano engine + piano-roll video.

This is the 'get the actual piece data and play it' path: instead of me approximating notes from
memory (which goes wrong on complex pieces), we load a real MIDI of a PUBLIC-DOMAIN composition and
play every note exactly. mido parses the MIDI; we reuse virtuoso's soundfont audio + piano-roll.

  python midi_render.py --midi moonlight3.mid --wav out.wav --roll out.mp4 --title "..."
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from mido import MidiFile

import virtuoso as V

# ----- General MIDI multi-instrument playback (for game music: drums+bass+lead+strings, not solo piano)
CH_COLORS = [
    (120, 200, 255), (255, 170, 120), (140, 230, 170), (255, 210, 110),
    (200, 150, 255), (240, 120, 120), (120, 255, 220), (255, 140, 200),
    (160, 200, 120), (255, 99, 132), (90, 180, 255), (230, 220, 120),
    (180, 255, 140), (255, 180, 90), (200, 160, 255), (150, 150, 160),
]


def gm_events(path):
    """Parse a MIDI honoring per-channel program changes. Returns (events, programs, span).
    events: [(t, midi, dur, vel, channel)]; programs: {channel: gm_program}."""
    mid = MidiFile(path)
    t = 0.0
    on = {}
    events = []
    programs = {}
    for msg in mid:
        t += msg.time
        if msg.type == "program_change":
            programs[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            on[(msg.note, msg.channel)] = (t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            k = (msg.note, msg.channel)
            if k in on:
                t0, vel = on.pop(k)
                events.append((t0, int(msg.note), max(0.04, t - t0), int(vel), msg.channel))
    for (note, ch), (t0, vel) in on.items():
        events.append((t0, int(note), 0.3, int(vel), ch))
    events.sort(key=lambda e: e[0])
    span = max((e[0] + e[2] for e in events), default=0.0)
    return events, programs, span


def render_gm_audio(events, programs, span, seed=3, reverb_wet=0.26):
    import tinysoundfont
    synth = tinysoundfont.Synth(samplerate=V.SR, gain=0.5)
    sfid = synth.sfload(V.SF2)
    # dual soundfont: route acoustic-piano channels (GM program 0/1) to a REAL recorded grand if present
    piano_sf = V._SF_DIR / "piano_real.sf2"
    psfid = synth.sfload(str(piano_sf)) if piano_sf.exists() else None
    chans = sorted({e[4] for e in events})
    for ch in chans:
        prog = programs.get(ch, 0)
        try:
            if ch == 9:
                synth.program_select(ch, sfid, 128, 0)               # GM drum kit
            elif psfid is not None and prog in (0, 1):
                synth.program_select(ch, psfid, 0, 0)                # real acoustic grand
            else:
                synth.program_select(ch, sfid, 0, prog)
        except Exception:
            synth.program_select(ch, sfid, 0, 0)
    seq = []
    for (t, n, dur, vel, ch) in events:
        seq.append((t, 1, ch, n, vel)); seq.append((t + dur, 0, ch, n, 0))
    seq.sort(key=lambda e: e[0])
    chunks = []; cursor = 0.0
    for (t, kind, ch, n, v) in seq:
        gap = int((t - cursor) * V.SR)
        if gap > 0:
            chunks.append(np.frombuffer(synth.generate(gap), dtype=np.float32)); cursor = t
        synth.noteon(ch, n, v) if kind == 1 else synth.noteoff(ch, n)
    chunks.append(np.frombuffer(synth.generate(int(2.5 * V.SR)), dtype=np.float32))
    arr = np.concatenate(chunks).reshape(-1, 2)
    L, R = arr[:, 0].copy(), arr[:, 1].copy()
    fo = int(1.5 * V.SR)
    for c in (L, R):
        c[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
    g = 10 ** (-3 / 20) / peak
    L, R = L * g, R * g
    if reverb_wet > 0:
        from audio_fx import reverb
        L, R = reverb(L, R, V.SR, wet=reverb_wet)
    return L, R


def midi_to_events(path, track_name="piano", drums_track="timp"):
    """-> [(t_sec, midi, dur_sec, vel, track)], total_span. Iterating MidiFile yields .time in seconds."""
    mid = MidiFile(path)
    t = 0.0
    on = {}
    events = []
    for msg in mid:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[(msg.note, msg.channel)] = (t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.note, msg.channel)
            if key in on:
                t0, vel = on.pop(key)
                tr = drums_track if msg.channel == 9 else track_name
                events.append((t0, int(msg.note), max(0.05, t - t0), int(vel), tr))
    # flush any hanging notes
    for (note, ch), (t0, vel) in on.items():
        events.append((t0, int(note), 0.3, int(vel), track_name))
    events.sort(key=lambda e: e[0])
    span = max((e[0] + e[2] for e in events), default=0.0)
    return events, span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--roll", default=None)
    ap.add_argument("--title", default="MIDI")
    ap.add_argument("--track", default="piano", choices=list(V.TRACKS))
    ap.add_argument("--gm", action="store_true", help="multi-instrument GM playback (game music)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="trim to this length (0 = full)")
    a = ap.parse_args()
    if a.gm:
        ev, programs, span = gm_events(a.midi)
        if a.max_seconds and span > a.max_seconds:
            ev = [e for e in ev if e[0] < a.max_seconds]; span = a.max_seconds
        print(f"[midi-gm] {len(ev)} notes, {span:.1f}s, channels {sorted(set(e[4] for e in ev))}")
        L, R = render_gm_audio(ev, programs, span, a.seed)
        V.write_wav(a.wav, L, R)
        print(f"[OK] GM audio -> {a.wav}")
        if a.roll:
            tk = list(V.TRACKS)
            roll_ev = [(t, n, dur, vel, tk[ch % len(tk)]) for (t, n, dur, vel, ch) in ev]
            V.render_roll(roll_ev, span, a.roll, a.title)
        return
    ev, span = midi_to_events(a.midi, track_name=a.track)
    if a.max_seconds and span > a.max_seconds:
        ev = [e for e in ev if e[0] < a.max_seconds]
        span = a.max_seconds
    print(f"[midi] {len(ev)} notes, {span:.1f}s")
    L, R = V.render_audio(ev, span, a.seed)
    V.write_wav(a.wav, L, R)
    print(f"[OK] audio -> {a.wav}")
    if a.roll:
        V.render_roll(ev, span, a.roll, a.title)


if __name__ == "__main__":
    main()
