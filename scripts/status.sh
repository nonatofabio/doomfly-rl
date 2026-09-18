#!/usr/bin/env bash
# Show pipeline status from the S3 log mirror (no SSM session needed). Usage: scripts/status.sh [train|teacher|record|all]
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3} AWS_REGION=us-west-2
BUCKET=${DOOMFLY_BUCKET:-doomfly-$(aws sts get-caller-identity --query Account --output text)-us-west-2}
IID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=doomfly-trainer Name=instance-state-name,Values=pending,running --query "Reservations[].Instances[].InstanceId" --output text)
[[ -n "$IID" ]] && aws ec2 describe-instances --instance-ids $IID --query 'Reservations[0].Instances[0].[InstanceId,State.Name,Placement.AvailabilityZone,LaunchTime]' --output text || echo "(no trainer instance running)"
T=$(mktemp -d); aws s3 sync s3://$BUCKET/logs $T --only-show-errors
echo "--- pipeline.log"; tail -n 15 $T/pipeline.log 2>/dev/null || echo "(no logs yet)"
what=${1:-all}
for f in $T/*.log; do
  b=$(basename $f); [[ $b == pipeline.log || $b == run_all.log ]] && continue
  [[ $what == all || $b == ${what}_* ]] || continue
  echo "--- $b"; grep -v "^$" $f | tail -n 4
done
for r in flywire783 malecns49k; do
  if aws s3 ls s3://$BUCKET/runs/$r/eval.jsonl >/dev/null 2>&1; then
    echo "--- eval $r"; aws s3 cp s3://$BUCKET/runs/$r/eval.jsonl - | tail -n 2
  fi
done
rm -rf $T
