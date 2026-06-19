"""Ambient用 音楽エンジンセレクタ(共有)。ambient_build / ambient_gallery_build が import。
既存の音楽エンジンに乗る: generative(pad) / lofi(hiphop) / nature(rain/stream/waves) / soundtrack(lofi+nature)。
全て numpy・--seconds 対応・soundfont不要=CI可。compose/piano/virtuoso(soundfont)は将来追加。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
ENGINES = ["generative", "lofi", "nature", "soundtrack"]


def music_cmd(engine, out, seconds, seed=1, key="minor", variant=None):
    out, s = str(out), str(seconds)
    if engine == "generative":
        return [PY, str(HERE / "generative_music.py"), "--output", out, "--seconds", s, "--seed", str(seed), "--key", key]
    if engine == "lofi":
        return [PY, str(HERE / "lofi_music.py"), "--output", out, "--seconds", s, "--seed", str(seed)]
    if engine == "nature":
        return [PY, str(HERE / "nature_sound.py"), "--output", out, "--seconds", s, "--seed", str(seed), "--type", variant or "rain"]
    if engine == "soundtrack":
        return [PY, str(HERE / "soundtrack.py"), "--output", out, "--seconds", s, "--index", str(variant if variant is not None else 0)]
    raise SystemExit(f"[ambient_music] unknown engine: {engine} (use {ENGINES})")
