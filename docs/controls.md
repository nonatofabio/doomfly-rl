# Controls: does the connectome matter? (negative result)

**Question.** The FlyWire-783 (138k neurons) and MaleCNS-49k (49k neurons) students were statistically
indistinguishable after identical distillation (see the findings section of `README.md`). Either both connectomes carry the same
useful prior, or the connectome is acting as a fixed random projection and the learned stem/decoder do the work.
Two controls, same recipe as `l40s-v2/runs/malecns49k` (60k steps, batch 128, lr 3e-4 / lr_conn 3e-3, wd 0.01,
warmup 1000, seed 0, 5 dynamic steps, eval every 5k on 10 episodes):

| control | flag | what changes |
|---|---|---|
| `malecns49k_shuffled_s0` | `--shuffle-edges 0` | same 49,393 neurons, same 9,050,125 edges, edge endpoints shuffled degree-preserving (seed 0); all wiring structure destroyed, degree sequence kept |
| `malecns49k_noconn` | `--no-connectome` | stem → decoder directly; no neuron layer at all |

Runs: `s3://doomfly-047472448415-us-west-2/controls-shuffled-s0/` and `…/controls-noconn/`
(commit `6188701`, one g6e.8xlarge each, us-east-2c, 2026-09-21). Both finished with `fail=0` and self-terminated.

## Result at step 60,000 (10 eval episodes, mean ± std_return)

| run | params | train sps | wall | train acc | val acc | basic | defend_the_center | health_gathering | deadly_corridor | defend_the_line |
|---|---|---|---|---|---|---|---|---|---|---|
| connectome (malecns49k, `pi_ref`) | 48.2 M | 646 | 3.66 h | 0.952 | 0.949 | 78.2±7.5 | 18.6±1.8 | 1728±616 | 2280.5±2.4 | 21.0±5.9 |
| shuffled s0 | 48,160,765 | 1,454 | 1.68 h | 0.953 | 0.947 | 78.2±7.5 | 19.0±1.7 | 1602±597 | 2280.6±2.2 | 21.7±6.1 |
| no-connectome | 27,729,030 | 14,686 | 0.29 h | 0.953 | 0.943 | 77.5±8.5 | 18.3±1.6 | 1373±555 | 2279.9±2.4 | 23.0±5.0 |

Per-scenario validation top-1 agreement with the teacher at step 60k:

| run | basic | dtc | hg | dc | dtl |
|---|---|---|---|---|---|
| connectome | 0.987 | 0.959 | 0.971 | 0.916 | 0.913 |
| shuffled s0 | 0.984 | 0.951 | 0.962 | 0.904 | 0.933 |
| no-connectome | 0.987 | 0.956 | 0.945 | 0.926 | 0.908 |

Every difference is inside one standard deviation of the 10-episode eval. `basic` is bit-identical between
connectome and shuffled (78.2±7.5, same as every GRPO run: the scenario is saturated). `health_gathering` is the
only scenario with a visible gap (1728 vs 1602 vs 1373) but its eval std is ~600 and its trajectory over
training is noise: the no-connectome run hit 2043 and 2046 at steps 40k/45k, the shuffled run hit the 2100 ceiling
at 20k. The eval-over-training curves of the three runs overlap everywhere.

## What this says

1. **The connectome carries no measurable inductive bias for this task, at this scale, with this recipe.**
   Degree-preserving shuffling (same sparsity, same fan-in/fan-out, no biology) gives the same student.
   Removing the neuron layer entirely gives the same student with 58 % of the parameters.
2. The FlyWire/MaleCNS invariance reported in `README.md` is therefore explained by the null hypothesis: the
   dynamic layer is a fixed-sparsity recurrent random projection plus a learned gain, and the 64-wide stem and
   512-wide decoder (~27.7 M params) are sufficient to imitate a small CNN teacher at 95 % top-1.
3. Cost: the connectome makes training 22× slower (646 vs 14,686 sps) and inference ~12–27× slower than the
   teacher, for nothing measurable here.

## What it does not say

- Nothing about *RL from scratch* or *learning on tasks with headroom*. Distillation to 95 % agreement on five
  saturated scenarios is a low bar; every backbone clears it. The free-play GRPO plan (`docs/freeplay-plan.md`)
  runs the same three backbones on Freedoom II MAP01 where the student starts far from optimal — that is
  where an inductive bias would show, if it exists.
- One shuffle seed. `malecns49k_shuffled_s1` (`--shuffle-edges 1`) is queued to bound shuffle-seed variance;
  given the numbers above it would need to move by >2σ to change the conclusion.
- No claim about the chessfly setting (different task, different training).

## Reproduce

```
scripts/launch_region.sh us-east-2   # with RUN_SCRIPT=run_controls.sh PREFIX_TAG=controls-<name>
                                     # CONTROL_RUNS="malecns49k_shuffled_s0|--shuffle-edges,0" or
                                     #              "malecns49k_noconn|--no-connectome"
```
Numbers above are `metrics.jsonl` (last row) and `eval.jsonl` (`step == 60000`) of each run; `meta.json` has
the exact args and parameter counts.
