"""DoomFly training infrastructure (fnp3 / us-west-2).

One stack:
  * S3 bucket           - code tarball, connectomes, rollouts, checkpoints, logs
  * IAM instance role   - SSM (no SSH), read/write to the bucket
  * Security group      - no ingress at all; egress only (SSM + S3 + pip/HF)
  * g6e.12xlarge        - on-demand via a 1-instance ASG across all AZs (capacity is AZ-dependent), 4x L40S 48GB, 48 vCPU, 384 GB RAM, 1 TB gp3 root,
                          Deep Learning OSS Nvidia Driver AMI (PyTorch 2.7, Ubuntu 22.04)

User data just fetches scripts/run_all.sh from the bucket and runs it detached; the
whole pipeline (teachers -> rollouts -> two connectome trainings -> eval) lives in
that script so it can be re-run with `scripts/launch.sh` via SSM without redeploying.
"""
import os

import aws_cdk as cdk
from aws_cdk import (
    Stack, CfnOutput, RemovalPolicy, Duration,
    aws_ec2 as ec2, aws_iam as iam, aws_s3 as s3, aws_autoscaling as autoscaling,
)
from constructs import Construct

REGION = "us-west-2"
AMI_ID = os.environ.get("DOOMFLY_AMI", "ami-0ca70308d230e8a6e")  # DLAMI OSS Nvidia PyTorch 2.7 Ubuntu 22.04
INSTANCE_TYPE = os.environ.get("DOOMFLY_INSTANCE", "g6e.12xlarge")


class DoomFlyStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kw):
        super().__init__(scope, cid, **kw)

        bucket_name = f"doomfly-{self.account}-{self.region}"
        if os.environ.get("DOOMFLY_IMPORT_BUCKET") == "1":
            # the bucket is RETAINed on stack deletion; import it on re-deploys instead of failing on "already exists"
            bucket = s3.Bucket.from_bucket_name(self, "Bucket", bucket_name)
        else:
            bucket = s3.Bucket(
                self, "Bucket",
                bucket_name=bucket_name,
                block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                encryption=s3.BucketEncryption.S3_MANAGED,
                enforce_ssl=True,
                versioned=False,
                removal_policy=RemovalPolicy.RETAIN,
            )

        vpc = ec2.Vpc.from_lookup(self, "Vpc", is_default=True)

        role = iam.Role(
            self, "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")],
            description="DoomFly GPU trainer: SSM + bucket access",
        )
        bucket.grant_read_write(role)
        # let the box stop itself when the pipeline is done (cost hygiene)
        role.add_to_policy(iam.PolicyStatement(
            actions=["ec2:StopInstances", "ec2:DescribeInstances"],
            resources=["*"],
            conditions={"StringEquals": {"aws:ResourceTag/Project": "doomfly"}},
        ))
        role.add_to_policy(iam.PolicyStatement(actions=["ec2:DescribeInstances"], resources=["*"]))

        sg = ec2.SecurityGroup(self, "Sg", vpc=vpc, allow_all_outbound=True, description="DoomFly trainer (no ingress)")

        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "set -euxo pipefail",
            "export HOME=/root",
            f"export DOOMFLY_BUCKET={bucket.bucket_name}",
            "mkdir -p /opt/doomfly/logs",
            # code is shipped with scripts/ship.sh right after deploy; wait for it (up to 30 min)
            f"for i in $(seq 1 180); do aws s3 cp s3://{bucket.bucket_name}/code/run_all.sh /opt/doomfly/run_all.sh && break; sleep 10; done",
            "chmod +x /opt/doomfly/run_all.sh",
            "nohup /opt/doomfly/run_all.sh > /opt/doomfly/logs/run_all.log 2>&1 &",
        )

        # A 1-instance Auto Scaling Group instead of a bare ec2.Instance: g6e.12xlarge capacity is
        # AZ-dependent (2a was out at the 1st deploy, 2b at the 2nd) and an ASG spanning all default
        # subnets lets EC2 place the box wherever capacity exists. The pipeline ends by setting the
        # ASG's desired capacity to 0 (terminate; everything is already in S3) - a plain `shutdown`
        # would be seen as an unhealthy instance and replaced.
        lt = ec2.LaunchTemplate(
            self, "TrainerLaunchTemplate",
            instance_type=ec2.InstanceType(INSTANCE_TYPE),
            machine_image=ec2.MachineImage.generic_linux({REGION: AMI_ID}),
            role=role,
            security_group=sg,
            user_data=user_data,
            block_devices=[ec2.BlockDevice(
                device_name="/dev/sda1",
                volume=ec2.BlockDeviceVolume.ebs(1000, volume_type=ec2.EbsDeviceVolumeType.GP3, iops=6000,
                                                 delete_on_termination=True, encrypted=True),
            )],
            require_imdsv2=True,
            instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.STOP,
        )
        asg = autoscaling.AutoScalingGroup(
            self, "Trainer",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            launch_template=lt,
            min_capacity=0, max_capacity=1, desired_capacity=1,
            health_checks=autoscaling.HealthChecks.ec2(grace_period=Duration.minutes(30)),
        )
        cdk.Tags.of(asg).add("Project", "doomfly")
        cdk.Tags.of(asg).add("Name", "doomfly-trainer")
        # the box scales itself to 0 when the pipeline is done
        role.add_to_policy(iam.PolicyStatement(
            actions=["autoscaling:SetDesiredCapacity", "autoscaling:DescribeAutoScalingGroups"], resources=["*"],
        ))

        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "AsgName", value=asg.auto_scaling_group_name)
        CfnOutput(self, "FindInstance",
                  value="aws ec2 describe-instances --filters Name=tag:Name,Values=doomfly-trainer "
                        "Name=instance-state-name,Values=pending,running --query Reservations[].Instances[].InstanceId --output text")


app = cdk.App()
DoomFlyStack(app, "DoomFly", env=cdk.Environment(account=os.environ["CDK_DEFAULT_ACCOUNT"], region=REGION),
             description="DoomFly: fruit-fly connectome trained to play Doom (GPU trainer + bucket)")
app.synth()
