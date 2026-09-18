#!/usr/bin/env bash
# Land a DoomFly trainer in ANY region by walking instance types x AZs with raw run-instances.
# Reuses the deployed us-west-2 launch template's IAM profile / disk / user-data; resolves the DLAMI, default VPC
# subnets and an egress-only security group per region. Shutdown behaviour = terminate (no orphaned EBS).
#
#   scripts/launch_region.sh us-east-1                       # default type list (multi-GPU first)
#   TYPES="g6e.12xlarge g6e.24xlarge" scripts/launch_region.sh us-east-2
#   AZS="us-east-1a us-east-1c" scripts/launch_region.sh us-east-1
set -euo pipefail
export AWS_PROFILE=${AWS_PROFILE:-fnp3}
REGION=${1:?region}
HOME_REGION=${HOME_REGION:-us-west-2}
STACK=${STACK:-DoomFly}
BUCKET=${DOOMFLY_BUCKET:-doomfly-$(aws sts get-caller-identity --query Account --output text)-$HOME_REGION}
TYPES=${TYPES:-"g6e.12xlarge g6e.24xlarge g6e.48xlarge g6.12xlarge g6.24xlarge g5.12xlarge g5.24xlarge g6e.8xlarge g6e.4xlarge"}
AMI_NAME=${AMI_NAME:-"Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04)*"}

# --- template bits from the CDK stack (home region)
ASG=$(aws cloudformation describe-stack-resources --region $HOME_REGION --stack-name $STACK \
      --query "StackResources[?ResourceType=='AWS::AutoScaling::AutoScalingGroup'].PhysicalResourceId" --output text)
LT=$(aws autoscaling describe-auto-scaling-groups --region $HOME_REGION --auto-scaling-group-names $ASG \
      --query "AutoScalingGroups[0].MixedInstancesPolicy.LaunchTemplate.LaunchTemplateSpecification.LaunchTemplateId" --output text)
aws ec2 describe-launch-template-versions --region $HOME_REGION --launch-template-id $LT --versions '$Latest' \
  --query "LaunchTemplateVersions[0].LaunchTemplateData" > /tmp/doomfly-lt.json
PROFILE=$(python3 -c "import json;print(json.load(open('/tmp/doomfly-lt.json'))['IamInstanceProfile']['Arn'])")
BDM=$(python3 -c "import json;print(json.dumps(json.load(open('/tmp/doomfly-lt.json'))['BlockDeviceMappings']))")
# user-data: same as the template, plus AWS_DEFAULT_REGION so s3 calls target the home-region bucket directly
python3 - "$HOME_REGION" > /tmp/doomfly-userdata.sh <<'PY'
import json, base64, sys
ud = base64.b64decode(json.load(open('/tmp/doomfly-lt.json'))['UserData']).decode()
ud = ud.replace("export HOME=/root", f"export HOME=/root\nexport AWS_DEFAULT_REGION={sys.argv[1]}", 1)
print(ud, end="")
PY

# --- per-region resources
AMI=$(aws ec2 describe-images --region $REGION --owners amazon --filters "Name=name,Values=$AMI_NAME" \
      --query "sort_by(Images,&CreationDate)[-1].ImageId" --output text)
VPC=$(aws ec2 describe-vpcs --region $REGION --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text)
SG=$(aws ec2 describe-security-groups --region $REGION --filters Name=vpc-id,Values=$VPC Name=group-name,Values=doomfly-trainer \
      --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
if [[ -z "$SG" || "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --region $REGION --vpc-id $VPC --group-name doomfly-trainer \
        --description "DoomFly trainer: egress only, SSM for access" --query GroupId --output text)
fi
if [[ -z "${AZS:-}" ]]; then
  SUBNETS=$(aws ec2 describe-subnets --region $REGION --filters Name=vpc-id,Values=$VPC Name=default-for-az,Values=true \
            --query "Subnets[].[SubnetId,AvailabilityZone]" --output text)
else
  SUBNETS=$(for az in $AZS; do aws ec2 describe-subnets --region $REGION --filters Name=vpc-id,Values=$VPC Name=availability-zone,Values=$az \
            --query "Subnets[0].[SubnetId,AvailabilityZone]" --output text; done)
fi
echo "region=$REGION ami=$AMI vpc=$VPC sg=$SG profile=${PROFILE##*/}"
echo "$SUBNETS" | sed 's/^/  subnet: /'

# --- walk
while read -r type; do
  while read -r subnet az; do
    printf '%-14s %-12s ' "$type" "$az"
    out=$(aws ec2 run-instances --region $REGION --image-id $AMI --instance-type $type --subnet-id $subnet \
          --security-group-ids $SG --iam-instance-profile Arn=$PROFILE --block-device-mappings "$BDM" \
          --metadata-options HttpTokens=required --instance-initiated-shutdown-behavior terminate \
          --user-data file:///tmp/doomfly-userdata.sh \
          --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=doomfly-trainer},{Key=Project,Value=DoomFly}]" \
                               "ResourceType=volume,Tags=[{Key=Name,Value=doomfly-trainer}]" \
          --query "Instances[0].InstanceId" --output text 2>&1) && {
      echo "LANDED $out"
      echo "$out $REGION $type $az" > /tmp/doomfly-landed
      exit 0
    }
    echo "${out##*\) }" | cut -c1-90
  done <<< "$SUBNETS"
done <<< "$(echo $TYPES | tr ' ' '\n')"
echo "no capacity for [$TYPES] in $REGION"; exit 1
