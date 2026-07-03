#!/usr/bin/env bash
# Download the demo footage set (Sanada Yukimura fireworks festival, Nagano,
# 2017) from Wikimedia Commons and convert to mp4 with the exact filenames the
# committed example EDL expects.
#
# All footage by KENPEI, CC BY-SA 4.0 - see examples/fireworks/ATTRIBUTION.md
# Total download ~130MB. Requires curl + ffmpeg.
set -euo pipefail

DEST="/tmp/roughcut_demo"
BASE="https://commons.wikimedia.org/wiki/Special:FilePath"
UA="roughcut-demo-fetch/1.0 (https://github.com/shivampkumar/roughcut)"
mkdir -p "$DEST"

echo "Downloading to $DEST (~130MB from Wikimedia Commons)..."

for i in 1 2 3 4 5; do
  out="$DEST/sanada_fireworks_${i}.mp4"
  if [ -f "$out" ]; then echo "  [skip] $out"; continue; fi
  ogv="$DEST/_tmp_${i}.ogv"
  echo "  video $i/5..."
  curl -sL -A "$UA" "$BASE/Syousei_Sanada_Yukimura_fireworks2017-${i}.ogv" -o "$ogv"
  ffmpeg -y -v error -i "$ogv" -c:v libx264 -preset veryfast -crf 22 -c:a aac "$out"
  rm -f "$ogv"
done

for i in 1 2 3 4 5 6; do
  out="$DEST/sanada_fireworks_photo_${i}.jpg"
  if [ -f "$out" ]; then echo "  [skip] $out"; continue; fi
  echo "  photo $i/6..."
  curl -sL -A "$UA" "$BASE/Syousei_Sanada_Yukimura_fireworks2017-${i}.jpg" -o "$out"
done

echo "Done. Render the example reel with NO API keys:"
echo "  roughcut render --edl examples/fireworks/edl.json --assets examples/fireworks/assets.json --out reel.mp4"
