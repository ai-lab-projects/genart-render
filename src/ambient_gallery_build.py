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

# still(--mode still→PNG)を出せる多様なエンジン。(script, seed対応?, 追加flags)。
# seed対応は4個のみ(他は--seed渡すとエラー=2026-06-19確認)。全12engで毎回12種の別模様。
# (script, seedable, flags, clean, motion)。clean=True=ラベル無し→クロップしない。
# motion = 画像の性質で最適化(user 2026-06-21): zoom_in=詳細が小スケールに続くフラクタル(細部を見せる),
# zoom_out=全体の構造が重要(最後に全体が見える), pan_d/pan_u=縦長で全体も詳細も見たい。
# 除外: strange2d/attractor3d(8の字しょぼい)/magnetic_pendulum(重い粗い)/newton_basins(迷路汚い)/cyclic_ca(極彩色ブロック迷路)。
# motion(2026-06-22 user): 画像に合わせ zoom_in(全体→詳細)/zoom_out(中心33%→全体=意味ある引き,S=3.0で十分寄せて開始)/pan を使い分け。
# さらに除外: greenberg_hastings(極彩色ピンク迷路=cyclic_ca同類)/percolation/three_body(sparse)。
# 確実に美しく穏やかな10エンジンに厳選。
STILL_ENGINES = [
    ("phyllotaxis_simulator.py", False, [], False, "zoom_out"),    # ひまわり: 中心花托→全体螺旋
    ("newton_simulator.py", False, [], False, "zoom_in"),          # Newtonフラクタル詳細へ
    ("chladni_simulator.py", False, [], False, "pan_l"),           # 節パターンを横パン
    ("differential_growth_simulator.py", True, [], False, "zoom_out"),  # 珊瑚: 中心→全体の塊
    # Agentic借用(2026-06-21)。ラベル無し→clean=True(クロップしない)。voronoi虹色で除外。
    ("apollonian_simulator.py", False, [], True, "zoom_in"),       # 無限の小円へダイブ
    ("koch_simulator.py", False, [], True, "zoom_in"),             # 雪片の縁詳細へ
    ("dla_simulator.py", True, [], True, "zoom_in"),               # 樹枝の枝先へダイブ
    ("sandpile_simulator.py", False, [], True, "zoom_out"),        # 曼荼羅: 中心→全体
    ("wave_interference_simulator.py", False, [], True, "pan_u"),  # 干渉縞を縦パン
    ("times_table_simulator.py", False, [], True, "zoom_in"),      # 花の中心へダイブ
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
    ap.add_argument("--n", type=int, default=12, help="still枚数(別エンジンから, 最大=エンジン数)")
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

    # 1) 別エンジンから still を集める(1枚失敗しても継続)。seedは対応simのみ。
    engines = (STILL_ENGINES[a.engine_offset:] + STILL_ENGINES[:a.engine_offset])[:a.n]
    print(f"[1/4] still {len(engines)}種エンジン")
    got = 0
    clean_idx = set()
    for i, (script, seedable, flags, clean, motion) in enumerate(engines):
        if clean:
            clean_idx.add(i)
        # ファイル名に motion を埋め込む(s00_zoom_in.png)→ gallery_build が画像ごとの動きを読む
        out_png = stills / f"s{i:02d}_{motion}.png"
        cmd = [PY, str(HERE / script), "--mode", "still", "--output", str(out_png)]
        if seedable:
            cmd += ["--seed", str(a.seed_base + i)]
        cmd += flags
        ok = run(cmd, cap=300, fatal=False)
        got += 1 if ok else 0
    if got < 3:
        raise SystemExit(f"[FAIL] still {got}枚しか取れず")
    print(f"  {got}枚 取得")
    # 1.5) 中央クロップ: 解説sim(clean=False)のstillは端にテキストラベルが乗る→各辺13%切り落とし除去。
    #      借用sim(clean=True)はラベル無し→クロップしない(切れるのを防ぐ, user 2026-06-21)。
    import re as _re
    from PIL import Image
    crop = 0.13
    for png in sorted(stills.glob("*.png")):
        mo = _re.match(r"s(\d+)_", png.stem)
        idx = int(mo.group(1)) if mo else -1
        if idx in clean_idx:
            continue   # クリーン素材はクロップせず全体を見せる
        try:
            im = Image.open(png); w, h = im.size
            dx, dy = int(w * crop), int(h * crop)
            im.crop((dx, dy, w - dx, h - dy)).save(png)
        except Exception as e:
            print(f"  [crop skip] {png.name}: {e}")
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
