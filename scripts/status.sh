#!/usr/bin/env bash
# Show pipeline status from the S3 log mirror (no SSM session needed).
#   scripts/status.sh [train|teacher|record|all]            root prefix (the us-west-2 / first box)
#   DOOMFLY_PREFIX=g6-12xl-use2 scripts/status.sh            a box that runs with DOOMFLY_PREFIX set (see scripts/boxes.sh)
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3} AWS_REGION=us-west-2
BUCKET=${DOOMFLY_BUCKET:-doomfly-$(aws sts get-caller-identity --query Account --output text)-us-west-2}
"$(dirname "$0")/boxes.sh"
P=s3://$BUCKET${DOOMFLY_PREFIX:+/${DOOMFLY_PREFIX%/}}; echo "--- $P"
T=$(mktemp -d); aws s3 sync $P/logs $T --only-show-errors
echo "--- pipeline.log"; tail -n 15 $T/pipeline.log 2>/dev/null || echo "(no logs yet)"
what=${1:-all}
for f in $T/*.log; do
  b=$(basename $f); [[ $b == pipeline.log || $b == run_all.log ]] && continue
  [[ $what == all || $b == ${what}_* ]] || continue
  echo "--- $b"; grep -v "^$" $f | tail -n 4
done
for r in flywire783 malecns49k; do
  if aws s3 ls $P/runs/$r/eval.jsonl >/dev/null 2>&1; then
    echo "--- eval $r"; aws s3 cp $P/runs/$r/eval.jsonl - | tail -n 2
  fi
done
rm -rf $T
