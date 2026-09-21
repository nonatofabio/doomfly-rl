#!/usr/bin/env bash
# Package the repo + connectomes + on-box run scripts and upload to the DoomFly bucket (run from repo root, laptop side).
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3} AWS_REGION=us-west-2
BUCKET=${DOOMFLY_BUCKET:-doomfly-$(aws sts get-caller-identity --query Account --output text)-us-west-2}
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
tar --exclude .venv --exclude .git --exclude 'data' --exclude runs --exclude '__pycache__' --exclude cdk.out \
    --exclude tutorial --exclude checkpoints --exclude assets -czf $TMP/doomfly.tar.gz .
aws s3 cp $TMP/doomfly.tar.gz s3://$BUCKET/code/doomfly.tar.gz
for rs in run_all.sh run_v2.sh run_grpo.sh run_controls.sh; do aws s3 cp scripts/$rs s3://$BUCKET/code/$rs; done
aws s3 sync data/processed s3://$BUCKET/connectomes --exclude '*' --include 'connectome_*.npz' --include 'connectome_*.json'
rm -rf $TMP
echo "shipped to s3://$BUCKET/{code,connectomes}"
