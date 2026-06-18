#!/usr/bin/env python3
"""Ambient ギャラリー動画ビルダー: 多数のstillをsweep生成→gallery_build(Ken Burns+crossfade)→generative_music→mux。
既存(gallery_build/generative_music/各sim)に乗る。CI(genart-render)でも動く(run_capped無ければ素実行)。≤15分厳守。
Usage: python ambient_gallery_build.py --out outputs/g.mp4 [--n 8 --seconds 840 --music-seed 3 --seed-base 100]"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
PY = sys.executable
FF = imageio_ffmpeg.get_ffmpeg_exe()

# still を振る題材ローテ (sim, 追加flag) — 多様な generative-art 静止画
STILL_ROT = [
    ("aurora_simulator.py", []),
    ("water_simulator.py", ["--palette", "teal"]),
    ("ambient_flow_simulator.py", ["--variant", "silk lanes"]),
    ("water_simulator.py", ["--palette", "amber"]),
    ("aurora_simulator.py", []),
    ("water_simulator.py", ["--palette", "violet"]),
    ("ambient_flow_simulator.py", ["--variant", "many vortices"]),
    ("water_simulator.py", ["--palette", "emerald"]),
]


def run(cmd, cap=None):
    rc = HERE.parent.parent.parent / "lib" / "run_capped.sh"
    full = (["bash", str(rc), str(cap), "--"] + cmd) if (cap and rc.exists()) else cmd
    print("  $", " ".join(str(c) for c in cmd[:5]), "...", flush=True)
    r = subprocess.run(full)
    if r.returncode != 0:
        raise SystemExit(f"[FAIL] rc={r.returncode}: {cmd[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="still枚数")
    ap.add_argument("--seconds", type=float, default=840.0, help="最終尺(<=868)")
    ap.add_argument("--per", type=float, default=9.0, help="1枚の表示秒")
    ap.add_argument("--seed-base", type=int, default=100)
    ap.add_argument("--music-seed", type=int, default=1)
    ap.add_argument("--music-key", choices=["minor", "major"], default="minor")
    a = ap.parse_args()
    assert a.seconds <= 868, "15分制限"
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_gal_tmp"; (tmp / "stills").mkdir(parents=True, exist_ok=True)
    stills = tmp / "stills"; montage = tmp / "montage.mp4"; music = tmp / "music.wav"

    # 1) still を sweep 生成 (sim×palette×seed ローテ)
    print(f"[1/4] still {a.n}枚 生成")
    for i in range(a.n):
        script, flags = STILL_ROT[i % len(STILL_ROT)]
        run([PY, str(HERE / script), "--mode", "still", "--seed", str(a.seed_base + i),
             "--output", str(stills / f"s{i:02d}.png")] + flags, cap=300)
    # 2) gallery montage (Ken Burns + crossfade, 無音)
    print("[2/4] gallery montage")
    run([PY, str(HERE / "gallery_build.py"), str(stills), str(montage), "--per", str(a.per), "--xfade", "1.6"], cap=600)
    # 3) generative music (全尺)
    print("[3/4] 音楽生成")
    run([PY, str(HERE / "generative_music.py"), "--output", str(music),
         "--seconds", str(a.seconds), "--seed", str(a.music_seed), "--key", a.music_key], cap=600)
    # 4) montage を stream_loop で尺合わせ + music afade + 単一パス再エンコ(bitrate cap)
    print("[4/4] ループ+mux+再エンコ")
    fo = max(a.seconds - 3.0, 0.0)
    run([FF, "-v", "error", "-y", "-stream_loop", "-1", "-i", str(montage), "-i", str(music),
         "-filter_complex", f"[1:a]afade=t=in:st=0:d=2,afade=t=out:st={fo}:d=3[a]",
         "-map", "0:v", "-map", "[a]", "-t", str(a.seconds),
         "-c:v", "libx264", "-b:v", "1800k", "-maxrate", "2400k", "-bufsize", "4M",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)], cap=900)
    # 掃除
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[OK] {out} ({a.seconds:.0f}s, {out.stat().st_size/1e6:.0f}MB)")


if __name__ == "__main__":
    main()
