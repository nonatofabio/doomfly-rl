#!/usr/bin/env bash
# Side-by-side clips: two checkpoints of the same backbone, same scenario, same env seed,
# played next to each other so the difference between the methods is visible.
#
#   scripts/compare_clips.sh <left-asset-dir> <right-asset-dir> <slot-prefix> "<left label>" "<right label>"
#
# e.g.  scripts/compare_clips.sh assets/videos/malecns49k_student_iter0 assets/videos/malecns49k_grpo_base_iter300 \
#         malecns49k_student_vs_grpo "distilled student (GRPO iter 0)" "GRPO base, iter 300"
#
# Each asset dir is the output of
#   python -m doomfly.evaluate --ckpt <ckpt> ... --episodes N --gif-dir <dir>/episodes --tics --pick all --fmt mp4
# i.e. <dir>/episodes/<scenario>_ep<i>.mp4 + <dir>/eval_all.json. Both sides must have been recorded
# with the same --episodes so the median rule picks comparable episodes.
#
# Per scenario: median-return episode on each side (same rule as make_clips.sh), the shorter side is
# held on its last frame until the longer one ends, first/last frame held HOLD_IN/HOLD_OUT s, then
# hstack -> 1280x480 @35 fps with the label + episode return burned in (PIL, since brew ffmpeg has no drawtext).
# Output:
#   <right-asset-dir>/compare/<scenario>.mp4
#   tutorial/assets/videos/<slot-prefix>_<scenario>.mp4
set -euo pipefail
L=$(cd "$1" && pwd); R=$(cd "$2" && pwd); PREFIX=$3; LLAB=$4; RLAB=$5
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SLOTS=$ROOT/tutorial/assets/videos
HOLD_IN=${HOLD_IN:-0.75}; HOLD_OUT=${HOLD_OUT:-1.5}
PY=${PY:-$ROOT/.venv/bin/python}
OUT=$R/compare; mkdir -p "$OUT" "$SLOTS"

pick() {  # dir scenario -> "idx return n"
  "$PY" - "$1" "$2" <<'EOF'
import json,sys,numpy as np
d,g=sys.argv[1:]; r=json.load(open(f'{d}/eval_all.json'))[g]['returns']; o=np.argsort(r); i=int(o[len(o)//2])
print(i, f"{r[i]:.1f}", len(r))
EOF
}
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }

for g in basic defend_the_center defend_the_line health_gathering deadly_corridor; do
  read -r li lret ln <<<"$(pick "$L" "$g")"
  read -r ri rret rn <<<"$(pick "$R" "$g")"
  LS=$L/episodes/${g}_ep$li.mp4; RS=$R/episodes/${g}_ep$ri.mp4
  [[ -f $LS && -f $RS ]] || { echo "!! missing $LS or $RS"; continue; }
  # label strip: 1280x40 PNG, transparent, two centred labels
  LABEL=$OUT/${g}_label.png
  "$PY" - "$LABEL" "$LLAB · return $lret" "$RLAB · return $rret" <<'EOF'
import sys
from PIL import Image, ImageDraw, ImageFont
out, a, b = sys.argv[1:]
im = Image.new("RGBA", (1280, 40), (0, 0, 0, 150)); d = ImageDraw.Draw(im)
f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
for cx, t in ((320, a), (960, b)):
    w = d.textlength(t, font=f); d.text((cx - w / 2, 9), t, font=f, fill=(255, 255, 255, 255))
im.save(out)
EOF
  # hold the shorter side on its last frame so both end together (tpad with a large stop clamps to the longer input via shortest=0 + eof_action)
  ld=$(dur "$LS"); rd=$(dur "$RS"); T=$("$PY" -c "print(max($ld,$rd))")
  ffmpeg -hide_banner -loglevel error -y -i "$LS" -i "$RS" -i "$LABEL" -filter_complex "\
[0:v]tpad=stop_mode=clone:stop_duration=$T,trim=duration=$T,tpad=start_mode=clone:start_duration=$HOLD_IN:stop_mode=clone:stop_duration=$HOLD_OUT,scale=640:480:flags=neighbor[l];\
[1:v]tpad=stop_mode=clone:stop_duration=$T,trim=duration=$T,tpad=start_mode=clone:start_duration=$HOLD_IN:stop_mode=clone:stop_duration=$HOLD_OUT,scale=640:480:flags=neighbor[r];\
[l][r]hstack=inputs=2[v];[v][2:v]overlay=0:0,format=yuv420p" \
    -c:v libx264 -crf ${CRF:-23} -preset medium -movflags +faststart "$OUT/$g.mp4"
  cp "$OUT/$g.mp4" "$SLOTS/${PREFIX}_$g.mp4"
  printf '   %-20s L ep%-2s return %-7s | R ep%-2s return %-7s  (median of %s/%s)  %ss\n' \
    "$g" "$li" "$lret" "$ri" "$rret" "$ln" "$rn" "$(dur "$OUT/$g.mp4")"
done
echo "✓ $OUT ; slots $SLOTS/${PREFIX}_<scenario>.mp4"
