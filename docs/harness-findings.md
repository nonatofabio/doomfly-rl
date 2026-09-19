# Harness findings

Bugs and gotchas found in the training/eval harness, with the symptom that exposed them. Keep
this list short and concrete; one entry per finding.

## GRPO (`doomfly/grpo.py`)

1. **BatchNorm train/eval mismatch made the update policy differ from the sampling policy.**
   Symptom: iteration 1 reported `ratio != 1.0`, `kl > 0`, `clipfrac > 0` before any gradient
   step. Cause: the per-step `BatchNorm1d` layers (affine=False) were switched to `train()` for
   the update, so normalization statistics came from the minibatch instead of the running
   stats used at sampling time. Fix: keep `model.eval()` during the update (gradients still
   flow through the normalization). Commit `d8a73d2`.
2. **Observation aliasing in the rollout buffer.** `tr["obs"].append(obs[k])` stored a NumPy
   view into the vectorized env's observation array, which the env overwrote in place on the
   next step, so every stored observation was the final one. Fix: `.copy()`. Same commit.
   After both fixes iteration 1 gives `ratio 1.0, kl 0.0, clipfrac 0.0`.
3. **`basic` gives no GRPO signal from the distilled student.** Episodes last ~4 steps and all
   seed-matched episodes in a group score identically, so `adv_abs = 0` and `pg = 0`. The
   scenario is already near the ceiling (87 / 95), so this is expected; it costs ~2 s per
   iteration and is kept for the eval comparison.
4. **Update dominates wall time.** First A10G runs: `t_update` 65-88 s vs `t_sample` 13-16 s per
   iteration (2 epochs, minibatch 256, ~8k frames). About 100 s per non-basic iteration, so
   300 iterations is ~7-8 h per run. Worth profiling before scaling `--iters`.

## Launch scripts

5. **`launch_region.sh` `EXTRA_ENV` values were injected unquoted into user-data.** A value
   containing `|` or `;` (e.g. `GRPO_RUNS=a|--x,1;b|--y,2`) was parsed as shell operators,
   cloud-init failed (`--temperature,1.2: command not found`), and the box sat idle. Fix:
   values are now single-quoted (`export K='V'`). Values still cannot contain spaces, which
   is why `run_grpo.sh` accepts commas as argument separators.
6. **Retrying `launch_region.sh` across regions in a loop must `break` on the first landing**,
   otherwise you get one box per region with the same job. Happened once (2026-09-18); the
   duplicate was repurposed to a different grid via SSM instead of terminated.
