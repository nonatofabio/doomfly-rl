#!/usr/bin/env bash
# List every DoomFly trainer box across regions (tag Name=doomfly-trainer, plus the CDK LT tag), with its S3 prefix tag.
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3}
printf '%-20s %-11s %-14s %-12s %-9s %-14s %s\n' INSTANCE REGION TYPE AZ STATE PREFIX LAUNCHED
for r in ${DOOMFLY_REGIONS:-us-west-2 us-east-1 us-east-2 eu-west-1 eu-central-1}; do
  aws ec2 describe-instances --region $r --filters "Name=tag:Name,Values=doomfly-trainer,DoomFly/TrainerLaunchTemplate" \
    Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query "Reservations[].Instances[].[InstanceId,'$r',InstanceType,Placement.AvailabilityZone,State.Name,Tags[?Key=='Prefix']|[0].Value||'-',LaunchTime]" \
    --output text | awk -v OFS='\t' '{printf "%-20s %-11s %-14s %-12s %-9s %-14s %s\n",$1,$2,$3,$4,$5,$6,$7}'
done
