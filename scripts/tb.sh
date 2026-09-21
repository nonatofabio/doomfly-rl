#!/usr/bin/env bash
# Watch training in TensorBoard.
#   scripts/tb.sh                      live: SSM port-forward 6006 from the (first) running trainer -> http://localhost:6006
#   scripts/tb.sh live i-0123 us-east-2  live: a specific box (see scripts/boxes.sh); PORT=6007 to run several side by side
#   scripts/tb.sh s3                   offline: sync s3://<bucket>[/$DOOMFLY_PREFIX]/tb locally and serve it (works after the box is gone)
#   scripts/tb.sh all                  offline: sync every prefix's tb/ (root + all boxes) and serve them all side by side
# Requires: session-manager-plugin (brew install --cask session-manager-plugin) for live mode; tensorboard for s3 mode.
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3}
REGION=${AWS_REGION:-us-west-2}
BUCKET=${DOOMFLY_BUCKET:-doomfly-047472448415-us-west-2}
PORT=${PORT:-6006}
if [[ "${1:-live}" == "s3" || "${1:-live}" == "all" ]]; then
  mkdir -p /tmp/doomfly-tb
  if [[ "$1" == "all" ]]; then
    # every prefix in the bucket that has a tb/ dir: root plus one per box (its DOOMFLY_PREFIX, see scripts/boxes.sh)
    for p in "" $(aws s3 ls s3://$BUCKET/ | awk '$1=="PRE"{print $2}'); do
      [[ -n "$(aws s3 ls s3://$BUCKET/${p}tb/ 2>/dev/null)" ]] || continue
      echo "syncing ${p:-root}"; mkdir -p /tmp/doomfly-tb/${p:-root}
      aws s3 sync s3://$BUCKET/${p}tb /tmp/doomfly-tb/${p:-root} --only-show-errors
    done
  else
    P=${DOOMFLY_PREFIX:+${DOOMFLY_PREFIX%/}/}; mkdir -p /tmp/doomfly-tb/${P:-root}
    aws s3 sync s3://$BUCKET/${P}tb /tmp/doomfly-tb/${P:-root} --only-show-errors
  fi
  # prefer the repo venv's tensorboard (installed by `uv pip install -e .`), so this works without activating it
  TB=$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/tensorboard; [[ -x $TB ]] || TB=tensorboard
  exec "$TB" --logdir /tmp/doomfly-tb --port $PORT
fi
ID=${2:-}; [[ -n "$ID" ]] && REGION=${3:-$REGION}
# otherwise find the trainer in any region we might have launched in
[[ -n "$ID" ]] || for r in ${DOOMFLY_REGIONS:-$REGION us-east-1 us-east-2 eu-west-1}; do
  ID=$(aws ec2 describe-instances --region $r --filters Name=tag:Name,Values=doomfly-trainer Name=instance-state-name,Values=running \
        --query "Reservations[].Instances[].InstanceId" --output text | awk '{print $1}')
  if [[ -n "$ID" ]]; then REGION=$r; break; fi
done
[[ -n "${ID:-}" ]] || { echo "no running doomfly-trainer found"; exit 1; }
echo "forwarding $ID ($REGION):6006 -> http://localhost:$PORT   (Ctrl-C to stop)"
exec aws ssm start-session --region $REGION --target $ID --document-name AWS-StartPortForwardingSession \
  --parameters "{\"portNumber\":[\"6006\"],\"localPortNumber\":[\"$PORT\"]}"
