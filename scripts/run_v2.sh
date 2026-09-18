#!/usr/bin/env bash
# DoomFly v2 distillation on a FRESH box (no teachers, no recording). Runs ON the GPU instance via user-data.
#
#   1. bootstrap        run_all.sh stage 0 (venv, deps, code + connectomes from S3)
#   2. rollouts         pull the v2 rollout set from S3: retrained basic/deadly_corridor teachers
#                       (ROLLOUTS_V2) + the three scenarios whose v1 teachers were fine (ROLLOUTS_GOOD)
#   3. train            FlyNet distillation, one run per GPU, all in parallel
#   4. finish           final S3 sync, power off (instance-initiated-shutdown-behavior = terminate)
#
# Env knobs: DOOMFLY_BUCKET (required), DOOMFLY_PREFIX (default l40s-v2), TRAIN_STEPS, BATCH,
# ROLLOUTS_GOOD, ROLLOUTS_V2, STOP_WHEN_DONE (1/0), EXTRA_RUNS (1/0: add a batch-256 flywire783 when >= 3 GPUs).
set -euo pipefail
: "${DOOMFLY_BUCKET:?DOOMFLY_BUCKET must be set}"
export DOOMFLY_PREFIX=${DOOMFLY_PREFIX:-l40s-v2}
ROOT=/opt/doomfly; CODE=$ROOT/src; LOGS=$ROOT/logs; DATA=$ROOT/data; RUNS=$ROOT/runs; TB=$ROOT/tb
S3=s3://$DOOMFLY_BUCKET
OUT=$S3/${DOOMFLY_PREFIX%/}
ROLLOUTS_GOOD=${ROLLOUTS_GOOD:-$S3/g6-12xl-use2/rollouts}
ROLLOUTS_V2=${ROLLOUTS_V2:-$S3/g6-12xl-use2-v2/rollouts}
TRAIN_STEPS=${TRAIN_STEPS:-60000}
BATCH=${BATCH:-128}
STOP_WHEN_DONE=${STOP_WHEN_DONE:-1}
EXTRA_RUNS=${EXTRA_RUNS:-1}
mkdir -p $LOGS $DATA/rollouts $RUNS $TB $ROOT/markers
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOGS/pipeline.log; }

# background log shipper
( while true; do aws s3 sync $LOGS $OUT/logs --only-show-errors || true; aws s3 sync $TB $OUT/tb --only-show-errors || true; sleep 60; done ) &
SHIPPER=$!
trap 'kill $SHIPPER 2>/dev/null || true; aws s3 sync $LOGS $OUT/logs --only-show-errors || true' EXIT

# ---------------------------------------------------------------- 1: bootstrap (reuse run_all stage 0)
log "v2 pipeline: prefix $DOOMFLY_PREFIX, steps $TRAIN_STEPS, batch $BATCH"
if [[ ! -f $ROOT/markers/stage0.done ]]; then
  aws s3 cp $S3/code/run_all.sh $ROOT/run_all.sh --only-show-errors && chmod +x $ROOT/run_all.sh
  STAGES="0" STOP_WHEN_DONE=0 $ROOT/run_all.sh
fi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
set +u; if [[ -f /opt/pytorch/bin/activate ]]; then source /opt/pytorch/bin/activate; else source $ROOT/venv/bin/activate; fi; set -u
cd $CODE
if ! pgrep -f "tensorboard --logdir" >/dev/null; then
  nohup tensorboard --logdir $TB --port 6006 --bind_all --reload_interval 15 > $LOGS/tensorboard.log 2>&1 < /dev/null &
fi

# ---------------------------------------------------------------- 2: rollouts from S3
if [[ ! -f $ROOT/markers/rollouts.done ]]; then
  log "pulling rollouts: good=$ROLLOUTS_GOOD v2=$ROLLOUTS_V2"
  pids=()
  for sc in defend_the_center health_gathering defend_the_line; do
    aws s3 sync $ROLLOUTS_GOOD/$sc $DATA/rollouts/$sc --only-show-errors & pids+=($!)
  done
  for sc in basic deadly_corridor; do
    aws s3 sync $ROLLOUTS_V2/$sc $DATA/rollouts/$sc --only-show-errors & pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
  for sc in basic defend_the_center health_gathering deadly_corridor defend_the_line; do
    [[ -f $DATA/rollouts/$sc/meta.json ]] || { log "missing rollouts for $sc"; fail=1; }
    cp $DATA/rollouts/$sc/meta.json $LOGS/rollout_meta_$sc.json 2>/dev/null || true
  done
  [[ $fail == 0 ]] || { log "rollout pull FAILED"; exit 1; }
  du -sh $DATA/rollouts/* | tee -a $LOGS/pipeline.log
  touch $ROOT/markers/rollouts.done; log "rollouts ready"
fi

# ---------------------------------------------------------------- 3: train
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [[ $NGPU -ge 1 ]] || NGPU=1
log "training on $NGPU GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
train_one() {  # $1 = run name, $2 = connectome npz, $3 = gpu index, $4 = batch
  [[ -f $RUNS/$1/final/model.safetensors ]] && { log "$1 already final"; return 0; }
  CUDA_VISIBLE_DEVICES=$3 python -u -m doomfly.train --connectome $DATA/processed/$2 --tb $TB/train_$1 \
    --rollouts $DATA/rollouts --out $RUNS/$1 --steps $TRAIN_STEPS --batch $4 \
    --s3 $OUT/runs/$1 > $LOGS/train_$1.log 2>&1
}
pids=(); fail=0
train_one flywire783 connectome_783.npz $((0 % NGPU)) $BATCH & pids+=($!)
train_one malecns49k connectome_malecns49k.npz $((1 % NGPU)) $BATCH & pids+=($!)
if [[ $EXTRA_RUNS == 1 && $NGPU -ge 3 ]]; then
  train_one flywire783_b256 connectome_783.npz 2 256 & pids+=($!)
fi
for p in "${pids[@]}"; do wait $p || fail=1; done
[[ $fail == 0 ]] || log "training FAILED for at least one run (see train_*.log)"

# ---------------------------------------------------------------- 4: finish
log "final sync"
aws s3 sync $RUNS $OUT/runs --only-show-errors
aws s3 sync $LOGS $OUT/logs --only-show-errors
aws s3 sync $TB $OUT/tb --only-show-errors
touch $ROOT/markers/all.done; log "ALL DONE (v2) fail=$fail"
if [[ $STOP_WHEN_DONE == 1 ]]; then
  log "powering off (shutdown behaviour = terminate; artifacts are in $OUT)"
  sleep 30; shutdown -h now
fi
