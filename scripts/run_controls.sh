#!/usr/bin/env bash
# DoomFly connectome CONTROLS on a FRESH box. Runs ON the GPU instance via user-data (RUN_SCRIPT=run_controls.sh).
# Same distillation recipe as run_v2.sh (same rollouts, steps, batch, train.py defaults); the only change per run
# is the connectome null model:
#   --shuffle-edges S   degree-preserving shuffle of the MaleCNS wiring (seed S)   -> malecns49k_shuffled_sS
#   --no-connectome     stem -> decoder, no connectome layer at all                 -> malecns49k_noconn
#
#   1. bootstrap        run_all.sh stage 0 (venv, deps, code + connectomes from S3)
#   2. rollouts         identical pull to run_v2.sh (ROLLOUTS_GOOD for dtc/hg/dtl, ROLLOUTS_V2 for basic/dc)
#   3. train            one `python -m doomfly.train` per run, run i on GPU i % NGPU, all in parallel
#   4. finish           final S3 sync, power off (instance-initiated-shutdown-behavior = terminate)
#
# Env knobs:
#   DOOMFLY_BUCKET   required
#   DOOMFLY_PREFIX   S3 prefix for this box (default controls). One prefix per box.
#   CONNECTOME       npz under data/processed (default connectome_malecns49k.npz)
#   TRAIN_STEPS      default 60000 (= reference student)
#   BATCH            default 128   (= reference student)
#   CONTROL_RUNS     ';'-separated "name|extra args". Commas in the args become spaces (launch_region.sh EXTRA_ENV
#                    cannot carry spaces): "malecns49k_shuffled_s0|--shuffle-edges,0". Default = both controls.
#   ROLLOUTS_GOOD, ROLLOUTS_V2, STOP_WHEN_DONE (1/0, default 1)
set -euo pipefail
: "${DOOMFLY_BUCKET:?DOOMFLY_BUCKET must be set}"
export DOOMFLY_PREFIX=${DOOMFLY_PREFIX:-controls}
ROOT=/opt/doomfly; CODE=$ROOT/src; LOGS=$ROOT/logs; DATA=$ROOT/data; RUNS=$ROOT/runs; TB=$ROOT/tb
S3=s3://$DOOMFLY_BUCKET
OUT=$S3/${DOOMFLY_PREFIX%/}
ROLLOUTS_GOOD=${ROLLOUTS_GOOD:-$S3/g6-12xl-use2/rollouts}
ROLLOUTS_V2=${ROLLOUTS_V2:-$S3/g6-12xl-use2-v2/rollouts}
CONNECTOME=${CONNECTOME:-connectome_malecns49k.npz}
TRAIN_STEPS=${TRAIN_STEPS:-60000}
BATCH=${BATCH:-128}
CONTROL_RUNS=${CONTROL_RUNS:-"malecns49k_shuffled_s0|--shuffle-edges,0;malecns49k_noconn|--no-connectome"}
STOP_WHEN_DONE=${STOP_WHEN_DONE:-1}
mkdir -p $LOGS $DATA/rollouts $RUNS $TB $ROOT/markers
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOGS/pipeline.log; }

# background log shipper
( while true; do aws s3 sync $LOGS $OUT/logs --only-show-errors || true; aws s3 sync $TB $OUT/tb --only-show-errors || true; sleep 60; done ) &
SHIPPER=$!
trap 'kill $SHIPPER 2>/dev/null || true; aws s3 sync $LOGS $OUT/logs --only-show-errors || true' EXIT

# ---------------------------------------------------------------- 1: bootstrap (reuse run_all stage 0)
log "controls pipeline: prefix $DOOMFLY_PREFIX, connectome $CONNECTOME, steps $TRAIN_STEPS, batch $BATCH, runs: $CONTROL_RUNS"
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

# ---------------------------------------------------------------- 2: rollouts from S3 (same set as run_v2.sh)
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
train_one() {  # $1 = run name, $2 = gpu index, $3... = extra args
  local name=$1 gpu=$2; shift 2
  [[ -f $RUNS/$name/final/model.safetensors ]] && { log "$name already final"; return 0; }
  log "start $name on gpu $gpu: $*"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu python -u -m doomfly.train --connectome $DATA/processed/$CONNECTOME --tb $TB/train_$name \
    --rollouts $DATA/rollouts --out $RUNS/$name --steps $TRAIN_STEPS --batch $BATCH \
    --s3 $OUT/runs/$name "$@" > $LOGS/train_$name.log 2>&1
}
pids=(); fail=0; i=0
IFS=';' read -ra SPECS <<< "$CONTROL_RUNS"
for spec in "${SPECS[@]}"; do
  name=${spec%%|*}; extra=${spec#*|}; [[ "$spec" == *"|"* ]] || extra=""; extra=${extra//,/ }
  # shellcheck disable=SC2086
  train_one $name $((i % NGPU)) $extra & pids+=($!)
  i=$((i + 1))
done
for p in "${pids[@]}"; do wait $p || fail=1; done
[[ $fail == 0 ]] || log "training FAILED for at least one run (see train_*.log)"

# ---------------------------------------------------------------- 4: finish
log "final sync"
aws s3 sync $RUNS $OUT/runs --only-show-errors
aws s3 sync $LOGS $OUT/logs --only-show-errors
aws s3 sync $TB $OUT/tb --only-show-errors
touch $ROOT/markers/all.done; log "ALL DONE (controls) fail=$fail"
if [[ $STOP_WHEN_DONE == 1 ]]; then
  log "powering off (shutdown behaviour = terminate; artifacts are in $OUT)"
  sleep 30; shutdown -h now
fi
