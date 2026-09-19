#!/usr/bin/env bash
# Turn the per-episode MP4s written by
#   python -m doomfly.evaluate ... --gif-dir <dir>/episodes --tics --pick all --fmt mp4
# into the tutorial clips (median-return episode per scenario, 960x720, first and
# last frame held so sub-second episodes such as `basic` are readable) and a reel.
#
#   scripts/make_clips.sh <asset-dir> <backbone>
#
# e.g.  scripts/make_clips.sh assets/videos/malecns49k_v2_final malecns49k
#
# Expects <asset-dir>/episodes/<scenario>_ep<i>.mp4 and <asset-dir>/eval_all.json
# (stdout of evaluate.py). Output:
#   <asset-dir>/mp4/<scenario>.mp4                       960x720 @35 fps
#   <asset-dir>/doomfly_<tag>_reel.mp4                    five scenarios back to back
#   tutorial/assets/videos/<backbone>_<scenario>.mp4      slots read by index.html
#   tutorial/assets/videos/doomfly_<backbone>_reel_web.mp4   only when REEL_SLOT=1
set -euo pipefail
DIR=$(cd "$1" && pwd); BACKBONE=$2
TAG=$(basename "$DIR")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SLOTS=$ROOT/tutorial/assets/videos
HOLD_IN=${HOLD_IN:-0.75}; HOLD_OUT=${HOLD_OUT:-1.5}   # seconds of held first/last frame
mkdir -p "$DIR/mp4" "$SLOTS"

LIST=$(mktemp)
for g in basic defend_the_center defend_the_line health_gathering deadly_corridor; do
  # same rule as evaluate.py --pick median: argsort(returns)[n//2]
  EP=$(python3 -c "
import json,sys,numpy as np
r=json.load(open('$DIR/eval_all.json'))['$g']['returns']; o=np.argsort(r); i=int(o[len(o)//2])
print(i, r[i], len(r))")
  read -r i ret n <<<"$EP"
  SRC=$DIR/episodes/${g}_ep$i.mp4
  [[ -f $SRC ]] || { echo "!! missing $SRC"; continue; }
  ffmpeg -hide_banner -loglevel error -y -i "$SRC" \
    -vf "tpad=start_mode=clone:start_duration=$HOLD_IN:stop_mode=clone:stop_duration=$HOLD_OUT,scale=960:720:flags=neighbor,format=yuv420p" \
    -c:v libx264 -crf ${CRF:-23} -preset medium -movflags +faststart "$DIR/mp4/$g.mp4"
  cp "$DIR/mp4/$g.mp4" "$SLOTS/${BACKBONE}_$g.mp4"
  echo "file '$DIR/mp4/$g.mp4'" >> "$LIST"
  printf '   %-20s ep%-2s return %-7s (median of %s)  %ss\n' "$g" "$i" "$ret" "$n" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$DIR/mp4/$g.mp4")"
done
REEL=$DIR/doomfly_${TAG}_reel.mp4
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$LIST" -c copy -movflags +faststart "$REEL"
rm -f "$LIST"
echo "   reel $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$REEL")s  $(du -h "$REEL" | cut -f1)"
if [[ ${REEL_SLOT:-0} == 1 ]]; then   # hero reel: re-encode smaller for the web
  ffmpeg -hide_banner -loglevel error -y -i "$REEL" -c:v libx264 -crf ${REEL_CRF:-28} -preset slow \
    -movflags +faststart "$SLOTS/doomfly_${BACKBONE}_reel_web.mp4"
  echo "   → $SLOTS/doomfly_${BACKBONE}_reel_web.mp4 $(du -h "$SLOTS/doomfly_${BACKBONE}_reel_web.mp4" | cut -f1)"
fi
echo "✓ $DIR ; slots in $SLOTS"
