"""CI helper: compose a style -> render audio (GM + real piano + reverb) -> piano-roll -> mux to one mp4."""
import argparse, os
import compose as C, virtuoso as V, midi_render as MR
from moviepy import VideoFileClip, AudioFileClip
ap = argparse.ArgumentParser()
ap.add_argument("--style", required=True)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--bars", type=int, default=24)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ev, prog, span = C.compose(a.style, a.seed, a.bars)
L, R = MR.render_gm_audio(ev, prog, span, a.seed)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
wav = a.out.replace(".mp4", ".wav"); V.write_wav(wav, L, R)
tk = list(V.TRACKS); roll = [(t, n, d, vl, tk[c % len(tk)]) for (t, n, d, vl, c) in ev]
rm = a.out.replace(".mp4", "_roll.mp4")
V.render_roll(roll, span, rm, f"{a.style} (seed {a.seed})")
VideoFileClip(rm).with_audio(AudioFileClip(wav)).write_videofile(a.out, codec="libx264", audio_codec="aac", audio_bitrate="192k", logger=None)
os.remove(wav); os.remove(rm)
print("OK", a.out)
