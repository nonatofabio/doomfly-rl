#!/usr/bin/env bash
# Capacity probe: try to land ONE on-demand GPU instance from the DoomFly launch template, walking
# instance types x AZs until run-instances succeeds. This is what actually worked when the ASG
# (MixedInstancesPolicy) sat on InsufficientInstanceCapacity without rotating types.
# The instance is NOT ASG-owned: the pipeline ends with `shutdown -h now` (stopped, not terminated);
# terminate it yourself afterwards. Keep the ASG at desired=0 while a probed box is running.
# Usage: scripts/probe_launch.sh ["g6e.12xlarge g6e.4xlarge ..."]
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3} AWS_REGION=us-west-2
TYPES=${1:-"g6e.12xlarge g6e.24xlarge g6.12xlarge g6.24xlarge g5.12xlarge g5.24xlarge g6e.8xlarge g6e.4xlarge g6.8xlarge g5.8xlarge"}
ASG=$(aws cloudformation describe-stacks --stack-name DoomFly --query "Stacks[0].Outputs[?OutputKey=='AsgName'].OutputValue" --output text)
LT=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" \
  --query "AutoScalingGroups[0].MixedInstancesPolicy.LaunchTemplate.LaunchTemplateSpecification.LaunchTemplateId" --output text)
SUBNETS=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$ASG" --query "AutoScalingGroups[0].VPCZoneIdentifier" --output text | tr ',' ' ')
aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" --desired-capacity 0
for t in $TYPES; do for s in $SUBNETS; do
  if out=$(aws ec2 run-instances --launch-template LaunchTemplateId=$LT --instance-type $t --subnet-id $s \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=doomfly-trainer},{Key=Project,Value=doomfly}]" \
      --query "Instances[0].[InstanceId,Placement.AvailabilityZone]" --output text 2>&1); then
    echo "LAUNCHED $t -> $out"; exit 0
  else
    echo "no: $t $s ($(echo "$out" | grep -o 'Insufficient[A-Za-z]*\|Unsupported' | head -1))"
  fi
done; done
echo "no capacity for any of: $TYPES"; exit 1
