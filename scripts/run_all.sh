#!/usr/bin/env bash
# DoomFly end-to-end pipeline. Runs ON the GPU instance (launched by user-data or scripts/launch.sh).
#
#   stage 0  bootstrap    venv, deps, pull code + connectomes from S3
#   stage 1  teachers     5 PPO CnnPolicy teachers (one per scenario), spread over the 4 GPUs
#   stage 2  rollouts     eps-greedy teacher rollouts -> uint8 shards (frames, teacher dist, returns)
#   stage 3  train        FlyNet distillation, two connectome backbones in parallel (GPU0 / GPU1)
#   stage 4  finish       final S3 sync, stop the instance
#
# Everything is logged to /opt/doomfly/logs and synced to s3://$DOOMFLY_BUCKET/logs every 60 s.
# Idempotent per stage: a stage is skipped when its ".done" marker exists, so re-running after a
# failure resumes where it stopped. Env knobs: DOOMFLY_BUCKET (required), FRAMES_PER_SCENARIO,
# TRAIN_STEPS, BATCH, STOP_WHEN_DONE (1/0), STAGES (e.g. "3 4" to re-run only training).
set -euo pipefail

: "${DOOMFLY_BUCKET:?DOOMFLY_BUCKET must be set}"
ROOT=/opt/doomfly
CODE=$ROOT/src
LOGS=$ROOT/logs
DATA=$ROOT/data
RUNS=$ROOT/runs
S3=s3://$DOOMFLY_BUCKET
IN=$S3            # code + connectomes always come from the bucket root
# DOOMFLY_PREFIX lets several boxes (e.g. an ablation across instance types) write to disjoint prefixes
[[ -n "${DOOMFLY_PREFIX:-}" ]] && S3=$S3/${DOOMFLY_PREFIX%/}
FRAMES_PER_SCENARIO=${FRAMES_PER_SCENARIO:-300000}
TRAIN_STEPS=${TRAIN_STEPS:-60000}
# default batch 256 on 48 GB cards (L40S); halve it on 24 GB cards (L4 / A10G fallback instances)
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
GPU_MIB=${GPU_MIB%%$'\n'*}; GPU_MIB=${GPU_MIB:-49140}   # first GPU only (no `| head`: pipefail + SIGPIPE on multi-GPU boxes)
if [[ -z "${BATCH:-}" && "${GPU_MIB:-49140}" -lt 40000 ]]; then BATCH=128; fi
BATCH=${BATCH:-256}
# adapt to however many GPUs / cores the box actually has (capacity fallback may give us 1 GPU)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}; [[ $NGPU -ge 1 ]] || NGPU=1
N_ENVS=${N_ENVS:-$(( $(nproc) / 5 ))}; [[ $N_ENVS -ge 2 ]] || N_ENVS=2; [[ $N_ENVS -le 8 ]] || N_ENVS=8
STOP_WHEN_DONE=${STOP_WHEN_DONE:-1}
STAGES=${STAGES:-"0 1 2 3 4"}
SCENARIOS="basic defend_the_center health_gathering deadly_corridor defend_the_line"
mkdir -p $LOGS $DATA/teachers $DATA/rollouts $RUNS $ROOT/markers
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd $ROOT

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOGS/pipeline.log; }
stage_wanted() { [[ " $STAGES " == *" $1 "* ]]; }
done_marker() { echo $ROOT/markers/stage$1.done; }

# background log shipper
TB=$ROOT/tb; mkdir -p $TB
( while true; do aws s3 sync $LOGS $S3/logs --only-show-errors || true; aws s3 sync $TB $S3/tb --only-show-errors || true; sleep 60; done ) &
SHIPPER=$!
trap 'kill $SHIPPER 2>/dev/null || true; aws s3 sync $LOGS $S3/logs --only-show-errors || true' EXIT

# ---------------------------------------------------------------- stage 0: bootstrap
if stage_wanted 0 && [[ ! -f $(done_marker 0) ]]; then
  log "stage 0: bootstrap"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq libgl1 libglu1-mesa libsdl2-2.0-0 libopenal1 tmux htop >/dev/null
  nvidia-smi | tee -a $LOGS/pipeline.log
  log "instance type: $(curl -s -H "X-aws-ec2-metadata-token: $(curl -sX PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" http://169.254.169.254/latest/meta-data/instance-type)"
  aws s3 cp $IN/code/doomfly.tar.gz $ROOT/doomfly.tar.gz
  rm -rf $CODE && mkdir -p $CODE && tar -xzf $ROOT/doomfly.tar.gz -C $CODE
  aws s3 sync $IN/connectomes $DATA/processed --only-show-errors
  ls -la $DATA/processed | tee -a $LOGS/pipeline.log
  # DLAMI ships a PyTorch venv with matching CUDA; reuse it, add our deps on top
  # (DLAMI's activate references an unset LD_LIBRARY_PATH, which trips `set -u`; relax it around the source)
  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
  set +u
  if [[ -f /opt/pytorch/bin/activate ]]; then
    source /opt/pytorch/bin/activate
  else
    python3 -m venv $ROOT/venv && source $ROOT/venv/bin/activate && pip install -q torch
  fi
  set -u
  pip install -q -e "$CODE" tensorboard 2>&1 | tail -3
  python - <<'EOF' | tee -a $LOGS/pipeline.log
import torch, vizdoom, stable_baselines3
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "gpus")
print("vizdoom", vizdoom.__version__, "sb3", stable_baselines3.__version__)
EOF
  touch $(done_marker 0)
fi
# DLAMI's activate script trips `set -u` (LD_LIBRARY_PATH unbound) -- relax it around the source
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
set +u; if [[ -f /opt/pytorch/bin/activate ]]; then source /opt/pytorch/bin/activate; else source $ROOT/venv/bin/activate; fi; set -u
cd $CODE

# TensorBoard on :6006 (reach it with scripts/tb.sh, which port-forwards over SSM; no inbound SG rule needed)
if ! pgrep -f "tensorboard --logdir" >/dev/null; then
  nohup tensorboard --logdir $TB --port 6006 --bind_all --reload_interval 15 > $LOGS/tensorboard.log 2>&1 < /dev/null &
fi

# ---------------------------------------------------------------- stage 1: teachers
if stage_wanted 1 && [[ ! -f $(done_marker 1) ]]; then
  log "stage 1: PPO teachers (5 scenarios in parallel)"
  i=0; pids=()
  for sc in $SCENARIOS; do
    if [[ ! -f $DATA/teachers/$sc.zip ]]; then
      CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python -u -m doomfly.doom.teacher --scenario $sc --n-envs $N_ENVS --tb $TB/teachers \
        --out $DATA/teachers > $LOGS/teacher_$sc.log 2>&1 &
      pids+=($!)
    fi
    i=$((i+1))
  done
  fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  [[ $fail == 0 ]] || { log "stage 1 FAILED (see teacher_*.log)"; exit 1; }
  aws s3 sync $DATA/teachers $S3/teachers --only-show-errors
  touch $(done_marker 1); log "stage 1 done"
fi

# ---------------------------------------------------------------- stage 2: rollouts
if stage_wanted 2 && [[ ! -f $(done_marker 2) ]]; then
  log "stage 2: rollouts ($FRAMES_PER_SCENARIO frames/scenario)"
  i=0; pids=()
  for sc in $SCENARIOS; do
    if [[ ! -f $DATA/rollouts/$sc/meta.json ]]; then
      rm -rf $DATA/rollouts/$sc
      CUDA_VISIBLE_DEVICES=$((i % NGPU)) nohup python -u -m doomfly.doom.record --scenario $sc --teacher $DATA/teachers/$sc.zip \
        --frames $FRAMES_PER_SCENARIO --eps 0.1 --out $DATA/rollouts/$sc > $LOGS/record_$sc.log 2>&1 &
      pids+=($!)
    fi
    i=$((i+1))
  done
  fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  [[ $fail == 0 ]] || { log "stage 2 FAILED (see record_*.log)"; exit 1; }
  du -sh $DATA/rollouts/* | tee -a $LOGS/pipeline.log
  for sc in $SCENARIOS; do cp $DATA/rollouts/$sc/meta.json $LOGS/rollout_meta_$sc.json; done
  ( aws s3 sync $DATA/rollouts $S3/rollouts --only-show-errors > $LOGS/rollout_upload.log 2>&1 & )
  touch $(done_marker 2); log "stage 2 done"
fi

# ---------------------------------------------------------------- stage 3: train both backbones
if stage_wanted 3 && [[ ! -f $(done_marker 3) ]]; then
  # two backbones: in parallel on GPU0/GPU1 when we have >= 2 GPUs, otherwise back-to-back on GPU0
  log "stage 3: FlyNet distillation, flywire783 + malecns49k on $NGPU GPU(s) ($TRAIN_STEPS steps, batch $BATCH)"
  pids=(); fail=0
  train_one() {  # $1 = run name, $2 = connectome npz, $3 = gpu index
    CUDA_VISIBLE_DEVICES=$3 python -u -m doomfly.train --connectome $DATA/processed/$2 --tb $TB/train_$1 \
      --rollouts $DATA/rollouts --out $RUNS/$1 --steps $TRAIN_STEPS --batch $BATCH \
      --s3 $S3/runs/$1 > $LOGS/train_$1.log 2>&1
  }
  if [[ $NGPU -ge 2 ]]; then
    train_one flywire783 connectome_783.npz 0 & pids+=($!)
    train_one malecns49k connectome_malecns49k.npz 1 & pids+=($!)
    for p in "${pids[@]}"; do wait $p || fail=1; done
  else
    [[ -f $RUNS/flywire783/final/model.safetensors ]] || train_one flywire783 connectome_783.npz 0 || fail=1
    [[ -f $RUNS/malecns49k/final/model.safetensors ]] || train_one malecns49k connectome_malecns49k.npz 0 || fail=1
  fi
  [[ $fail == 0 ]] || { log "stage 3 FAILED (see train_*.log)"; exit 1; }
  touch $(done_marker 3); log "stage 3 done"
fi

# ---------------------------------------------------------------- stage 4: finish
if stage_wanted 4; then
  log "stage 4: final sync"
  aws s3 sync $RUNS $S3/runs --only-show-errors
  aws s3 sync $LOGS $S3/logs --only-show-errors
  touch $(done_marker 4); log "ALL DONE"
  if [[ $STOP_WHEN_DONE == 1 ]]; then
    log "terminating instance (ASG desired capacity -> 0; all artifacts are in S3)"
    sleep 30
    TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
    IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
    ASG=$(aws autoscaling describe-auto-scaling-groups --query "AutoScalingGroups[?Instances[?InstanceId=='$IID']].AutoScalingGroupName" --output text)
    [[ -n "$ASG" ]] && aws autoscaling set-desired-capacity --auto-scaling-group-name "$ASG" --desired-capacity 0 || shutdown -h now
  fi
fi
