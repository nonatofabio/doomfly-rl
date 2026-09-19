# doomfly-rl

A neural network wired like a fruit-fly brain, trained with gradient descent to play Doom.

> **Not [nftechie/doomfly](https://github.com/nftechie/doomfly).** That project runs the full
> MaleCNS connectome as a spiking network with fixed, biology-motivated input and output, and
> studies online plasticity in one real circuit. This one (`-rl`) treats the connectome as an
> inductive bias and *learns* the encoder, the decoder and one gain per synapse. Same name,
> opposite question; see [Relation to nftechie/doomfly](#relation-to-nftechiedoomfly) below and
> `docs/related-work-nftechie-doomfly.md` for the full comparison. The Python package is still
> `doomfly`.

The model (`FlyNet`) reuses the recipe of [mlabonne/chessfly](https://huggingface.co/mlabonne/chessfly):
the recurrent core is the real connectome (FlyWire v783, 138,639 neurons, 15.09 M synapses), and training
only learns one gain per synapse. Sign and weight class come from the biology. The visual input neurons
receive the game frame, the output is read from the descending neurons, and a small decoder turns that
into one of 22 Doom actions. A second, smaller backbone (`malecns49k`, 49,393 neurons) is trained as an ablation.

Read `tutorial/index.html` for the full, self-contained walkthrough (open it in a browser, no server).

## Relation to nftechie/doomfly

Both projects take "a fly brain plays Doom" literally, but they make opposite design choices, and it
helps to be explicit about why.

**What they do.** `nftechie/doomfly` simulates the whole MaleCNS graph (166,700 neurons, 25.6 M edges,
no pruning) as leaky integrate-and-fire neurons with biological time constants, in an event-driven
C++ kernel. Pixels are mapped onto R1–R6/R8 photoreceptors through inferred ommatidia; actions are read
off named descending neurons (DNp20 rate difference → turn, DNpe017 → move/fire). Nothing on the input
or output side is learned. The only plastic synapses are the 4,184 KC→MBON11 connections, updated
online by a dopamine-gated anti-Hebbian rule, with two PPL101 dopamine cells driven by in-game damage.
Their `AGENTS.md` forbids cropping circuits, pruning edges, or "replacing the network with a game
policy". They publish negative results: the v6 candidate did not pass its own survival gates, and the
repo says so.

**Why that is a coherent design.** It is a computational-neuroscience experiment. The question is
whether the *actual* wiring, run as a spiking system with plausible sensors and effectors, produces
useful behaviour, and whether a *known* plasticity mechanism in a *known* circuit can shape it. Every
lever we pull here — a learned conv stem, a learned readout, a free gain on every synapse, a value head,
distillation from PPO teachers — would contaminate that question, because a conv net and an MLP can
learn to play Doom on their own and the connectome in the middle could be doing nothing. Refusing those
levers is what makes "no learning demonstrated" a meaningful result rather than a bug.

**Why we do it differently.** `doomfly-rl` asks the ML question instead: if a recurrent network is
constrained to the fly's connectivity and synapse signs, can optimisation make it play, and how does
that compare with the same recipe on chess (`chessfly`)? Here the connectome is a structural prior, not
a claim about biology. Consequences we accept:

- The stem (4×72×96 frames → 10,855 visual neurons), the readout (33,788 central/descending neurons →
  512 → 22 actions) and the 64-bin value head are ordinary learned modules, 34.5 M of the 51.0 M
  parameters. They do real work. Any claim about "the fly brain" playing has to be read net of them,
  which is why the ablation backbone and the per-component parameter counts are reported.
- Dynamics are rate units unrolled for 5 steps with BatchNorm and homeostatic scale/shift, not spikes
  with millisecond time constants. This trains on a GPU in hours; a spiking simulation of the same graph
  runs at 0.16× wall time on their hardware.
- Learning is offline (PPO teachers → ε-greedy rollouts → distillation, optionally GRPO/RLOO), on the
  five standard ViZDoom scenarios, so results are comparable across backbones and to published RL
  baselines.
- We prune to the FlyWire v783 graph with class labels, and build a 49k-neuron MaleCNS subgraph as an
  ablation. Their contract disallows exactly this.

Neither project invalidates the other. Theirs tells you what the wiring does on its own; ours tells you
what the wiring is worth as a prior once you let gradient descent in. Read a `doomfly-rl` score as
"connectome-constrained RL agent", never as "a fly".

## Layout

| Path | What it is |
|---|---|
| `doomfly/connectome/` | Connectome ETL (`build.py`, `build_malecns49k.py`): FlyWire / MaleCNS tables to `data/processed/connectome_*.npz` |
| `doomfly/model/` | `sparse.py` (sparse recurrent op), `flynet.py` (the model) |
| `doomfly/doom/` | `env.py` (ViZDoom wrapper), `actions.py` (action vocabulary), `teacher.py` (PPO teachers), `record.py` (rollouts) |
| `doomfly/train.py` | Distils the teachers into FlyNet. `--tb DIR` writes TensorBoard scalars |
| `doomfly/grpo.py` | Critic-free GRPO / RLOO fine-tuning of a distilled student against a frozen reference |
| `doomfly/evaluate.py` | Plays the trained model and writes GIF/MP4 footage (`--tics`, `--pick`, `--fmt`) |
| `infra/` | CDK stack `DoomFly` (S3 bucket, launch template, IAM, ASG) in `us-west-2` |
| `scripts/` | Ship, launch, watch (see below) |
| `tutorial/` | `generate.py` + `template.html` -> `index.html` infographic |
| `docs/` | Related work, GRPO study notes, harness findings |
| `AGENTS.md` | What this project is for (harness capability test + GRPO study) and the rules agents follow |

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
