# doomfly

A neural network wired like a fruit-fly brain, trained to play Doom.

The model (`FlyNet`) reuses the recipe of [mlabonne/chessfly](https://huggingface.co/mlabonne/chessfly):
the recurrent core is the real connectome (FlyWire v783, 138,639 neurons, 15.09 M synapses), and training
only learns one gain per synapse. Sign and weight class come from the biology. The visual input neurons
receive the game frame, the output is read from the descending neurons, and a small decoder turns that
into one of 22 Doom actions. A second, smaller backbone (`malecns49k`, 49,393 neurons) is trained as an ablation.

Read `tutorial/index.html` for the full, self-contained walkthrough (open it in a browser, no server).

## Layout

| Path | What it is |
|---|---|
| `doomfly/connectome/` | Connectome ETL (`build.py`, `build_malecns49k.py`): FlyWire / MaleCNS tables to `data/processed/connectome_*.npz` |
| `doomfly/model/` | `sparse.py` (sparse recurrent op), `flynet.py` (the model) |
| `doomfly/doom/` | `env.py` (ViZDoom wrapper), `actions.py` (action vocabulary), `teacher.py` (PPO teachers), `record.py` (rollouts) |
| `doomfly/train.py` | Distils the teachers into FlyNet. `--tb DIR` writes TensorBoard scalars |
| `doomfly/evaluate.py` | Plays the trained model and writes GIFs |
| `infra/` | CDK stack `DoomFly` (S3 bucket, launch template, IAM, ASG) in `us-west-2` |
| `scripts/` | Ship, launch, watch (see below) |
| `tutorial/` | `generate.py` + `template.html` -> `index.html` infographic |

## Pipeline on the GPU box

`scripts/run_all.sh` runs on the instance at boot. Stages, each with a `markers/stageN.done` file:

0. bootstrap (venv, deps, pull code + connectomes from S3)
1. PPO teachers, one per scenario, in parallel
2. epsilon-greedy rollouts from each teacher
3. FlyNet distillation, `flywire783` then `malecns49k` (in parallel if there are 2+ GPUs)
4. final S3 sync, then the instance shuts down

Knobs (environment): `DOOMFLY_BUCKET`, `DOOMFLY_PREFIX`, `FRAMES_PER_SCENARIO`, `TRAIN_STEPS`, `BATCH`, `N_ENVS`, `STAGES`, `STOP_WHEN_DONE`.

## Operate

```bash
export AWS_PROFILE=fnp3
scripts/ship.sh                         # code + connectomes -> s3://doomfly-<acct>-us-west-2/{code,connectomes}
scripts/launch.sh                       # ASG in us-west-2 (desired=1)
scripts/launch_region.sh us-east-2      # no capacity at home? walk TYPES x AZs in another region
scripts/boxes.sh                        # every trainer box, all regions, with its S3 prefix
scripts/status.sh [train|teacher|record] # tail the S3 log mirror (DOOMFLY_PREFIX=... for a prefixed box)
scripts/tb.sh                           # TensorBoard: SSM port-forward from the box -> http://localhost:6006
scripts/tb.sh live i-0123 us-east-2     # ... a specific box (PORT=6007 for a second one)
scripts/tb.sh s3                        # TensorBoard from the S3 mirror, works after the box is gone
```

Two boxes must not share an S3 prefix. Give the second one `DOOMFLY_PREFIX=<name>` (and tag it `Prefix=<name>`).

## Local development

```bash
uv venv && uv pip install -e .
python -m doomfly.doom.teacher --scenario basic --steps 2000 --n-envs 2 --out /tmp/t --tb /tmp/tb/teachers
python -m doomfly.doom.record --scenario basic --teacher /tmp/t/basic.zip --frames 400 --out /tmp/r
python -m doomfly.train --connectome data/processed/connectome_783.npz --rollouts /tmp/r --out /tmp/run \
    --tb /tmp/tb/train --steps 60 --batch 8 --device cpu
tensorboard --logdir /tmp/tb
```
