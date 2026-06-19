#!/usr/bin/env python3
"""Ambient ギャラリー動画ビルダー: **多数の別エンジン**から still を集めて gallery_build(Ken Burns+crossfade)→音楽→mux。
「いろんな数学模様の drift」= breadth を活かす。既存(各sim/gallery_build/音楽エンジン)に乗る。CI可。≤15分。
Usage: python ambient_gallery_build.py --out outputs/g.mp4 [--n 10 --seconds 840 --seed-base 100 --music-engine nature --music-variant rain]"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import imageio_ffmpeg
import ambient_music

HERE = Path(__file__).resolve().parent
PY = sys.executable
FF = imageio_ffmpeg.get_ffmpeg_exe()

# still(--mode still→PNG)を出せる多様なエンジン。1本ごとに見た目が全然違う。
STILL_ENGINES = [
    ("strange2d_simulator.py", []),
    ("phyllotaxis_simulator.py", []),
    ("newton_simulator.py", []),
    ("differential_growth_simulator.py", []),
    ("chladni_simulator.py", []),
    ("attractor3d_simulator.py", []),
    ("newton_basins_simulator.py", []),
    ("magnetic_pendulum_simulator.py", []),
    ("percolation_simulator.py", []),
    ("cyclic_ca_simulator.py", []),
    ("greenberg_hastings_simulator.py", []),
    ("three_body_simulator.py", []),
]


def run(cmd, cap=None, fatal=True):
    rc = HERE.parent.parent.parent / "lib" / "run_capped.sh"
    full = (["bash", str(rc), str(cap), "--"] + cmd) if (cap and rc.exists()) else cmd
    r = subprocess.run(full)
    if r.returncode != 0:
        if fatal:
            raise SystemExit(f"[FAIL] rc={r.returncode}: {cmd[0]}")
        print(f"  [skip] rc={r.returncode}: {Path(str(cmd[1])).name if len(cmd)>1 else cmd[0]}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10, help="still枚数(別エンジンから)")
    ap.add_argument("--seconds", type=float, default=840.0)
    ap.add_argument("--per", type=float, default=9.0)
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--engine-offset", type=int, default=0, help="STILL_ENGINES開始位置(ギャラリー間で別の混合に)")
    ap.add_argument("--music-engine", default="generative", choices=ambient_music.ENGINES)
    ap.add_argument("--music-variant", default=None, help="nature: rain/stream/waves, soundtrack: index")
    ap.add_argument("--music-seed", type=int, default=1)
    ap.add_argument("--music-key", choices=["minor", "major"], default="minor")
    a = ap.parse_args()
    assert a.seconds <= 868, "15分制限"
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f"_gal_{out.stem}"; stills = tmp / "stills"; stills.mkdir(parents=True, exist_ok=True)
    montage = tmp / "montage.mp4"; music = tmp / "music.wav"

    # 1) 別エンジンから still を集める(1枚失敗しても継続)
    print(f"[1/4] still {a.n}枚 (多エンジン)")
    got = 0
    for i in range(a.n):
        script, flags = STILL_ENGINES[(a.engine_offset + i) % len(STILL_ENGINES)]
        ok = run([PY, str(HERE / script), "--mode", "still", "--seed", str(a.seed_base + i),
                  "--output", str(stills / f"s{i:02d}.png")] + flags, cap=300, fatal=False)
        got += 1 if ok else 0
    if got < 3:
        raise SystemExit(f"[FAIL] still {got}枚しか取れず")
    print(f"  {got}枚 取得")
    # 2) gallery montage
    print("[2/4] gallery montage")
    run([PY, str(HERE / "gallery_build.py"), str(stills), str(montage), "--per", str(a.per), "--xfade", "1.6"], cap=600)
    # 3) 音楽(選択エンジン)
    print(f"[3/4] 音楽 = {a.music_engine}" + (f"/{a.music_variant}" if a.music_variant else ""))
    run(ambient_music.music_cmd(a.music_engine, music, a.seconds, a.music_seed, a.music_key, a.music_variant), cap=600)
    # 4) ループ+mux+再エンコ
    print("[4/4] ループ+mux+再エンコ")
    fo = max(a.seconds - 3.0, 0.0)
    run([FF, "-v", "error", "-y", "-stream_loop", "-1", "-i", str(montage), "-i", str(music),
         "-filter_complex", f"[1:a]afade=t=in:st=0:d=2,afade=t=out:st={fo}:d=3[a]",
         "-map", "0:v", "-map", "[a]", "-t", str(a.seconds),
         "-c:v", "libx264", "-b:v", "1800k", "-maxrate", "2400k", "-bufsize", "4M",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)], cap=900)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[OK] {out} ({a.seconds:.0f}s, {out.stat().st_size/1e6:.0f}MB)")


if __name__ == "__main__":
    main()
