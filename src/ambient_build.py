"""Ambient longform ビルダー — calm sim のシームレスループ → 14分尺へ連結 → 生成音楽を fade mux。

Ambient Pixels の「ぽこぽこ量産」用(2026-06-09 user)。垂れ流し映像＋自作ambient音楽の長尺(<=15分厳守: 未認証)。
  ループ生成(sim) → ffmpeg -stream_loop で尺延長(無劣化copy) → generative_music で全尺音楽 → afade mux。

  python ambient_build.py --sim flow --style "silk lanes" --seed 11 --music-seed 3 --out outputs/ambient_pixels_001/video/flow_silk.mp4
  python ambient_build.py --sim aurora --seed 7 --music-seed 5
  python ambient_build.py --sim water --style teal --seed 4 --music-key major
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
PY = sys.executable
FF = imageio_ffmpeg.get_ffmpeg_exe()

# sim名 -> (スクリプト, styleを渡すフラグ名 or None)
SIMS = {
    "flow":     ("ambient_flow_simulator.py", "--variant"),   # 'silk lanes'|'tight eddies'|'broad streams'|'many vortices'
    "water":    ("water_simulator.py", "--palette"),          # teal 等
    "plasma":   ("plasma_simulator.py", None),                # 流れる液体色(解析式=高速・継ぎ目なし)
    # aurora/reaction/fire は user評価で除外(2026-06-21): auroraイマイチ, reaction=24sループ繰返し不向き, fire=前回不調
}


def run(cmd, cap=None):
    # run_capped は VM 専用(lib/)。CI(genart-render)等で無ければ素で実行(CI側に独自timeoutあり)
    rc = HERE.parent.parent.parent / "lib" / "run_capped.sh"
    full = (["bash", str(rc), str(cap), "--"] + cmd) if (cap and rc.exists()) else cmd
    print("  $", " ".join(str(c) for c in cmd[:6]), "...", flush=True)
    r = subprocess.run(full)
    if r.returncode != 0:
        raise SystemExit(f"[FAIL] rc={r.returncode}: {cmd[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=None, choices=list(SIMS), help="--loop-file 指定時は不要")
    ap.add_argument("--loop-file", default=None, help="既製のシームレスループmp4(CI製等)を使い sim をスキップ")
    ap.add_argument("--style", default=None, help="flow=variant / water=palette")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--loop-seconds", type=float, default=24.0, help="シームレスループ1周の秒")
    ap.add_argument("--seconds", type=float, default=864.0, help="最終尺(<=870厳守: 15分制限)")
    ap.add_argument("--music-seed", type=int, default=1)
    ap.add_argument("--music-key", choices=["minor", "major"], default="minor")
    ap.add_argument("--music", default=None, help="既存音源(wav/mp4)をBGMに使う(クラシック等)。尺は曲長に合わせる(<=870)")
    import ambient_music
    ap.add_argument("--music-engine", default="generative", choices=ambient_music.ENGINES, help="generative/lofi/nature/soundtrack")
    ap.add_argument("--music-variant", default=None, help="nature: rain/stream/waves, soundtrack: index")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # --music 指定時: 尺を曲長に合わせる(ループ無し, <=15分)
    if a.music:
        from moviepy import AudioFileClip
        _ac = AudioFileClip(a.music); a.seconds = min(_ac.duration, 868.0); _ac.close()
    assert a.seconds <= 870, "15分(870s)制限: 未認証チャンネル"
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / "_amb_tmp"
    tmp.mkdir(exist_ok=True)
    loop = tmp / f"{out.stem}_loop.mp4"
    music = tmp / f"{out.stem}_music.wav"

    # 1) シームレスループ: --loop-file あれば既製(CI製等)を使い sim スキップ、無ければ生成
    if a.loop_file:
        import shutil
        print(f"[1/3] 既製ループ使用 {Path(a.loop_file).name}")
        shutil.copy(a.loop_file, loop)
    else:
        assert a.sim, "--sim か --loop-file のどちらかが必要"
        script, styleflag = SIMS[a.sim]
        cmd = [PY, str(HERE / script), "--mode", "loop", "--seconds", str(a.loop_seconds),
               "--seed", str(a.seed), "--output", str(loop)]
        if styleflag and a.style:
            cmd += [styleflag, a.style]
        print("[1/3] ループ生成"); run(cmd, cap=900)
    # 2) BGM: --music あれば既存音源(クラシック等)、無ければ generative
    if a.music:
        print(f"[2/3] BGM=既存音源 {Path(a.music).name} ({a.seconds:.0f}s)")
        music = Path(a.music)   # 既存ファイルをそのまま使う(ffmpegがmp4からも音声抽出)
    else:
        print(f"[2/3] 音楽生成({a.music_engine}" + (f"/{a.music_variant}" if a.music_variant else "") + ")")
        run(ambient_music.music_cmd(a.music_engine, music, a.seconds, a.music_seed, a.music_key, a.music_variant), cap=600)
    # 3) stream_loop で尺延長しつつ音楽 afade mux、**単一パスで再エンコード**(crf23で~150MB、巨大中間を作らずディスク節約)
    print("[3/3] ループ延長+mux+再エンコード(crf23)")
    fo = max(a.seconds - 3.0, 0.0)
    run([FF, "-v", "error", "-y", "-stream_loop", "-1", "-i", str(loop), "-i", str(music),
         "-filter_complex", f"[1:a]afade=t=in:st=0:d=2,afade=t=out:st={fo}:d=3[a]",
         "-map", "0:v", "-map", "[a]", "-t", str(a.seconds),
         # bitrate cap(~1.8Mbps≈190MB/14分): ambient背景は detail でも cap で十分、巨大化防止(crfだと966MB)
         "-c:v", "libx264", "-b:v", "1800k", "-maxrate", "2400k", "-bufsize", "4M",
         "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", str(out)], cap=900)
    # 中間掃除(ディスク節約)。--music の本物素材は消さない(tmp配下の生成musicのみ削除)
    loop.unlink(missing_ok=True)
    if not a.music and music.parent == tmp:
        music.unlink(missing_ok=True)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(f"[OK] {out} ({a.seconds:.0f}s, {out.stat().st_size/1e6:.0f}MB)")


if __name__ == "__main__":
    main()
