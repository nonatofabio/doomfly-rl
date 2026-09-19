#!/usr/bin/env bash
# DoomFly GRPO fine-tuning on a FRESH box. Runs ON the GPU instance via user-data (RUN_SCRIPT=run_grpo.sh).
#
#   1. bootstrap        run_all.sh stage 0 (venv, deps, code + connectomes from S3)
#   2. student          pull the distilled student (pi_ref) final/ from S3
#   3. grpo             one `python -m doomfly.grpo` config per GPU, all in parallel
#   4. finish           final S3 sync, power off (instance-initiated-shutdown-behavior = terminate)
#
# Env knobs:
#   DOOMFLY_BUCKET   required
#   DOOMFLY_PREFIX   S3 prefix for this box (default grpo). One prefix per box.
#   STUDENT          S3 dir with config.json + model.safetensors (default $S3/l40s-v2/runs/malecns49k/final)
#   STUDENT_NAME     backbone name used in run names (default malecns49k)
#   CONNECTOME       npz under data/processed (default connectome_malecns49k.npz)
#   GRPO_COMMON      args shared by every run (default: --iters 300 --eval-every 25 --eval-episodes 10)
#   GRPO_RUNS        ';'-separated "what|extra args". Run i goes to GPU i % NGPU. Run name = ${STUDENT_NAME}_grpo_<what>.
#                    Commas in the args become spaces (launch_region.sh EXTRA_ENV cannot carry spaces), so
#                    "beta0.2|--beta,0.2" == "beta0.2|--beta 0.2". Default = 4-run grid, one knob per run (see below).
#   STOP_WHEN_DONE   1/0 (default 1)
set -euo pipefail
: "${DOOMFLY_BUCKET:?DOOMFLY_BUCKET must be set}"
export DOOMFLY_PREFIX=${DOOMFLY_PREFIX:-grpo}
ROOT=/opt/doomfly; CODE=$ROOT/src; LOGS=$ROOT/logs; DATA=$ROOT/data; RUNS=$ROOT/runs; TB=$ROOT/tb
S3=s3://$DOOMFLY_BUCKET
OUT=$S3/${DOOMFLY_PREFIX%/}
STUDENT=${STUDENT:-$S3/l40s-v2/runs/malecns49k/final}
STUDENT_NAME=${STUDENT_NAME:-malecns49k}
CONNECTOME=${CONNECTOME:-connectome_malecns49k.npz}
GRPO_COMMON=${GRPO_COMMON:-"--iters 300 --eval-every 25 --eval-episodes 10"}
# default grid: base (grpo.py defaults: beta 0.05, groups 4 x G 8, T 1.0, lr 3e-5) + one knob each
GRPO_RUNS=${GRPO_RUNS:-"base|;beta0.01|--beta 0.01;beta0.2|--beta 0.2;g16|--group-size 16"}
STOP_WHEN_DONE=${STOP_WHEN_DONE:-1}
mkdir -p $LOGS $DATA/student $RUNS $TB $ROOT/markers
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOGS/pipeline.log; }

# background log shipper
( while true; do aws s3 sync $LOGS $OUT/logs --only-show-errors || true; aws s3 sync $TB $OUT/tb --only-show-errors || true; sleep 60; done ) &
SHIPPER=$!
trap 'kill $SHIPPER 2>/dev/null || true; aws s3 sync $LOGS $OUT/logs --only-show-errors || true' EXIT

# ---------------------------------------------------------------- 1: bootstrap (reuse run_all stage 0)
log "grpo pipeline: prefix $DOOMFLY_PREFIX, student $STUDENT, runs: $GRPO_RUNS"
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

# ---------------------------------------------------------------- 2: student (pi_ref)
if [[ ! -f $ROOT/markers/student.done ]]; then
  aws s3 cp $STUDENT/config.json $DATA/student/config.json --only-show-errors
  aws s3 cp $STUDENT/model.safetensors $DATA/student/model.safetensors --only-show-errors
  [[ -s $DATA/student/model.safetensors ]] || { log "student pull FAILED from $STUDENT"; exit 1; }
  cp $DATA/student/config.json $LOGS/student_config.json
  echo "$STUDENT" > $LOGS/student_source.txt
  touch $ROOT/markers/student.done; log "student ready: $(du -h $DATA/student/model.safetensors | cut -f1)"
fi

# ---------------------------------------------------------------- 3: grpo
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [[ $NGPU -ge 1 ]] || NGPU=1
log "grpo on $NGPU GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
grpo_one() {  # $1 = run name, $2 = gpu index, $3... = extra args
  local name=$1 gpu=$2; shift 2
  [[ -f $RUNS/$name/final/model.safetensors ]] && { log "$name already final"; return 0; }
  log "start $name on gpu $gpu: $GRPO_COMMON $*"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$gpu python -u -m doomfly.grpo --ckpt $DATA/student --connectome $DATA/processed/$CONNECTOME \
    --out $RUNS/$name --tb $TB/grpo_$name --s3 $OUT/runs/$name $GRPO_COMMON "$@" > $LOGS/grpo_$name.log 2>&1
}
pids=(); fail=0; i=0
IFS=';' read -ra SPECS <<< "$GRPO_RUNS"
for spec in "${SPECS[@]}"; do
  what=${spec%%|*}; extra=${spec#*|}; [[ "$spec" == *"|"* ]] || extra=""; extra=${extra//,/ }
  # shellcheck disable=SC2086
  grpo_one ${STUDENT_NAME}_grpo_$what $((i % NGPU)) $extra & pids+=($!)
  i=$((i + 1))
done
for p in "${pids[@]}"; do wait $p || fail=1; done
[[ $fail == 0 ]] || log "grpo FAILED for at least one run (see grpo_*.log)"

# ---------------------------------------------------------------- 4: finish
log "final sync"
aws s3 sync $RUNS $OUT/runs --only-show-errors
aws s3 sync $LOGS $OUT/logs --only-show-errors
aws s3 sync $TB $OUT/tb --only-show-errors
touch $ROOT/markers/all.done; log "ALL DONE (grpo) fail=$fail"
if [[ $STOP_WHEN_DONE == 1 ]]; then
  log "powering off (shutdown behaviour = terminate; artifacts are in $OUT)"
  sleep 30; shutdown -h now
fi
