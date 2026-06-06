#!/usr/bin/env bash
# Download the FREE soundfonts into ./soundfonts (not committed; fetched at render time).
set -e
mkdir -p soundfonts
echo "[sf] GeneralUser GS..."
curl -sL -o soundfonts/gu.sf2 "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/main/GeneralUser-GS.sf2"
echo "[sf] YDP Grand Piano (real recorded grand, CC-BY FreePats)..."
curl -sL -o /tmp/ydp.tar.bz2 "https://freepats.zenvoid.org/Piano/YDP-GrandPiano/YDP-GrandPiano-SF2-20160804.tar.bz2"
tar xjf /tmp/ydp.tar.bz2 -C /tmp
cp "$(find /tmp -iname 'YDP*.sf2' | head -1)" soundfonts/piano_real.sf2
ls -la soundfonts/
