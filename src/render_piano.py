"""CI helper: render a piano MIDI (url or path) faithfully with restrained humanization -> mp4."""
import argparse, os, urllib.request
import midi_render as MR, virtuoso as V
from moviepy import VideoFileClip, AudioFileClip
ap = argparse.ArgumentParser()
ap.add_argument("--midi", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-seconds", type=float, default=0.0)
ap.add_argument("--strength", type=float, default=0.5)
ap.add_argument("--title", default="piano")
a = ap.parse_args()
midi = a.midi
if midi.startswith("http"):
    req = urllib.request.Request(midi, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open("/tmp/_dl.mid", "wb") as f:
        f.write(r.read())
    midi = "/tmp/_dl.mid"
ev, prog, span = MR.gm_events(midi)
if a.max_seconds and span > a.max_seconds:
    ev = [e for e in ev if e[0] < a.max_seconds]; span = a.max_seconds
ev, span = MR.humanize(ev, span, 0, a.strength)
print(f"[piano] {len(ev)} notes, {span:.1f}s, strength {a.strength}")
L, R = MR.render_gm_audio(ev, prog, span, 3)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
wav = a.out.replace(".mp4", ".wav"); V.write_wav(wav, L, R)
tk = list(V.TRACKS); roll = [(t, n, d, vl, tk[c % len(tk)]) for (t, n, d, vl, c) in ev]
rm = a.out.replace(".mp4", "_roll.mp4")
V.render_roll(roll, span, rm, a.title)
VideoFileClip(rm).with_audio(AudioFileClip(wav)).write_videofile(a.out, codec="libx264", audio_codec="aac", audio_bitrate="192k", logger=None)
os.remove(wav); os.remove(rm)
print("OK", a.out)
