# Side-car retrain of the two collapsed teachers on idle GPUs 2-3 of a running 4-GPU box, then re-record + re-distill under prefix g6-12xl-use2-v2 (run via SSM, base64-encoded).
# --- part 1: teachers ---
set -eu
ROOT=/opt/doomfly_v2; S3=s3://doomfly-047472448415-us-west-2
mkdir -p $ROOT/code $ROOT/data/teachers $ROOT/logs && cd $ROOT
aws s3 cp $S3/code/doomfly.tar.gz doomfly.tar.gz --only-show-errors && tar -xzf doomfly.tar.gz -C code
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; set +u; source /opt/pytorch/bin/activate; set -u
export PYTHONPATH=$ROOT/code PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
python -c "import doomfly.doom.teacher as t, doomfly.doom.actions as a; print(t.__file__); print(a.SCENARIOS['deadly_corridor'])" | tee logs/check.log
i=2
for sc in basic deadly_corridor; do
  CUDA_VISIBLE_DEVICES=$i nohup python -u -m doomfly.doom.teacher --scenario $sc --n-envs 8 --tb $ROOT/tb --out $ROOT/data/teachers > logs/teacher_$sc.log 2>&1 &
  i=$((i+1))
done
nohup bash -c 'while true; do aws s3 sync /opt/doomfly_v2/logs '$S3'/teachers_v2/logs --only-show-errors; aws s3 sync /opt/doomfly_v2/data/teachers '$S3'/teachers_v2 --only-show-errors; sleep 60; done' > /dev/null 2>&1 &
sleep 20; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv; tail -3 logs/teacher_*.log

# --- part 2: chain (record -> train -> sync) ---
set -u
ROOT=/opt/doomfly_v2; V1=/opt/doomfly; S3=s3://doomfly-047472448415-us-west-2/g6-12xl-use2-v2
LOGS=$ROOT/logs; DATA=$ROOT/data; RUNS=$ROOT/runs; TB=$ROOT/tb; mkdir -p $RUNS $DATA/rollouts
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; set +u; source /opt/pytorch/bin/activate; set -u
export PYTHONPATH=$ROOT/code PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOGS/chain.log; }
# keep the v1 run_all's end-of-run `shutdown -h now` from killing us; we shut down ourselves at the end
systemctl mask poweroff.target halt.target >/dev/null 2>&1; log "masked poweroff/halt"
# background shipper
( while true; do aws s3 sync $LOGS $S3/logs --only-show-errors; aws s3 sync $TB $S3/tb --only-show-errors; aws s3 sync $DATA/teachers $S3/teachers --only-show-errors; sleep 60; done ) &
cd $ROOT
log "waiting for teachers"
while pgrep -f "doomfly.doom.teacher" >/dev/null; do sleep 30; done
for sc in basic deadly_corridor; do [[ -f $DATA/teachers/$sc.zip ]] || { log "missing $sc.zip"; exit 1; }; done
log "teachers done; recording"
i=2; pids=()
for sc in basic deadly_corridor; do
  rm -rf $DATA/rollouts/$sc
  CUDA_VISIBLE_DEVICES=$i nohup python -u -m doomfly.doom.record --scenario $sc --teacher $DATA/teachers/$sc.zip --frames 300000 --eps 0.1 --out $DATA/rollouts/$sc > $LOGS/record_$sc.log 2>&1 &
  pids+=($!); i=$((i+1))
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[[ $fail == 0 ]] || { log "record FAILED"; exit 1; }
for sc in defend_the_center health_gathering defend_the_line; do ln -sfn $V1/data/rollouts/$sc $DATA/rollouts/$sc; done
for sc in basic deadly_corridor; do cp $DATA/rollouts/$sc/meta.json $LOGS/rollout_meta_$sc.json; done
( aws s3 sync $DATA/rollouts/basic $S3/rollouts/basic --only-show-errors; aws s3 sync $DATA/rollouts/deadly_corridor $S3/rollouts/deadly_corridor --only-show-errors ) &
log "rollouts done; training"
train_one() { CUDA_VISIBLE_DEVICES=$3 python -u -m doomfly.train --connectome $V1/data/processed/$2 --tb $TB/train_$1 --rollouts $DATA/rollouts --out $RUNS/$1 --steps 60000 --batch 128 --s3 $S3/runs/$1 > $LOGS/train_$1.log 2>&1; }
pids=(); train_one flywire783 connectome_783.npz 2 & pids+=($!); train_one malecns49k connectome_malecns49k.npz 3 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
log "training finished fail=$fail; final sync"
aws s3 sync $RUNS $S3/runs --only-show-errors; aws s3 sync $LOGS $S3/logs --only-show-errors; aws s3 sync $TB $S3/tb --only-show-errors
log "ALL DONE (v2)"
# only power off once the v1 pipeline is also finished
while [[ ! -f $V1/markers/stage4.done ]] && pgrep -f "doomfly.train" >/dev/null; do sleep 60; done
systemctl unmask poweroff.target halt.target; sleep 30; shutdown -h now
