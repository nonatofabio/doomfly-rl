#!/usr/bin/env bash
# (Re)start the pipeline on the trainer via SSM. Usage: scripts/launch.sh [STAGES="3 4"] [extra env KEY=VAL...]
# e.g.  STAGES="3 4" TRAIN_STEPS=80000 scripts/launch.sh
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3} AWS_REGION=us-west-2
IID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=doomfly-trainer Name=instance-state-name,Values=pending,running --query "Reservations[].Instances[].InstanceId" --output text)
BUCKET=$(aws cloudformation describe-stacks --stack-name DoomFly --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text)
ENVS="export DOOMFLY_BUCKET=$BUCKET"
for v in STAGES FRAMES_PER_SCENARIO TRAIN_STEPS BATCH STOP_WHEN_DONE; do
  [[ -n "${!v:-}" ]] && ENVS="$ENVS ${v}='${!v}'"
done
CMD="$ENVS; mkdir -p /opt/doomfly/logs; aws s3 cp s3://$BUCKET/code/run_all.sh /opt/doomfly/run_all.sh; chmod +x /opt/doomfly/run_all.sh; rm -f /opt/doomfly/markers/stage0.done; nohup /opt/doomfly/run_all.sh >> /opt/doomfly/logs/run_all.log 2>&1 & echo launched"
aws ssm send-command --instance-ids "$IID" --document-name AWS-RunShellScript \
  --parameters "commands=[\"$CMD\"]" --query 'Command.CommandId' --output text
