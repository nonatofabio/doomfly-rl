#!/usr/bin/env bash
# Pull the per-scenario best-episode GIFs that train.py uploads to S3, convert
# them to browser-friendly MP4s, drop them into the tutorial video slots and
# build a concatenated reel.
#
#   scripts/pull_videos.sh <s3-prefix> <backbone> <step|final> [tag]
#
# e.g.  scripts/pull_videos.sh g6-12xl-use2-v2 malecns49k 60000
#       scripts/pull_videos.sh l40s-v2 flywire783 final          # gifs/final/
#
# Output:
#   assets/videos/<tag>/gif/<scenario>.gif      raw GIFs (8.75 fps, 320x240)
#   assets/videos/<tag>/mp4/<scenario>.mp4      960x720 @30 fps, yuv420p, faststart
#   assets/videos/<tag>/doomfly_<tag>_reel.mp4  all five scenarios back to back
#   tutorial/assets/videos/<backbone>_<scenario>.mp4   (slots read by index.html)
#   tutorial/assets/videos/doomfly_<backbone>_reel_web.mp4 only when REEL_SLOT=1
set -euo pipefail
PREFIX=$1; BACKBONE=$2; STEP=$3
TAG=${4:-${BACKBONE}_${PREFIX}_step${STEP}}
BUCKET=${BUCKET:-s3://doomfly-047472448415-us-west-2}
export AWS_PROFILE=${AWS_PROFILE:-fnp3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/assets/videos/$TAG
SLOTS=$ROOT/tutorial/assets/videos

if [[ $STEP == final ]]; then GIFDIR=final; else GIFDIR=$(printf 'step%07d' "$STEP"); fi
SRC=$BUCKET/$PREFIX/runs/$BACKBONE/gifs/$GIFDIR/
echo "→ $SRC"
mkdir -p "$OUT/gif" "$OUT/mp4" "$SLOTS"
aws s3 sync --only-show-errors "$SRC" "$OUT/gif/"
ls "$OUT"/gif/*.gif >/dev/null

LIST=$(mktemp)
for g in basic defend_the_center defend_the_line health_gathering deadly_corridor; do
  [[ -f $OUT/gif/$g.gif ]] || { echo "!! missing $g.gif"; continue; }
  ffmpeg -hide_banner -loglevel error -y -i "$OUT/gif/$g.gif" \
    -vf "scale=960:720:flags=neighbor,fps=30,format=yuv420p" \
    -c:v libx264 -crf ${CRF:-23} -preset medium -movflags +faststart "$OUT/mp4/$g.mp4"
  cp "$OUT/mp4/$g.mp4" "$SLOTS/${BACKBONE}_$g.mp4"
  echo "file '$OUT/mp4/$g.mp4'" >> "$LIST"
  printf '   %-20s %s frames\n' "$g" "$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$OUT/mp4/$g.mp4")"
done
REEL=$OUT/doomfly_${TAG}_reel.mp4
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$LIST" -c copy -movflags +faststart "$REEL"
rm -f "$LIST"
echo "   reel $(ffprobe -v error -show_entries format=duration -of csv=p=0 "$REEL")s  $(du -h "$REEL" | cut -f1)"
if [[ ${REEL_SLOT:-0} == 1 ]]; then
  cp "$REEL" "$SLOTS/doomfly_${BACKBONE}_reel_web.mp4"
  echo "   → $SLOTS/doomfly_${BACKBONE}_reel_web.mp4"
fi
echo "✓ $OUT ; slots in $SLOTS"
