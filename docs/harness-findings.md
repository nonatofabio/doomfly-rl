# Harness findings

Bugs and gotchas found in the training/eval harness, with the symptom that exposed them. Keep
this list short and concrete; one entry per finding. Results (run table, GRPO vs. PPO
comparison, conclusions) live in `docs/grpo-study.md`; this file is only about the harness.

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

## Free play (`doomfly/grpo_freeplay.py`) and the local machine

7. **The in-run eval seeds are the first training seeds.** `grpo_freeplay.py` draws its 10 eval
   seeds from `default_rng(0)` so that they match `freeplay_zero_shot --seed 0`, and draws training
   seeds from `default_rng(--seed)`, default 0. The eval seeds are therefore exactly the training
   seeds of iterations 1-3 (4 per iteration). This was found after the three runs finished. The
   `eval.jsonl` curves are used for shape only. Endpoint numbers come from
   `doomfly/freeplay_eval.py`, which scores 30 episodes on seed stream 1 and refuses stream 0.
   `grpo_freeplay.py` is unchanged, so a rerun reproduces the launched recipe.
8. **The PPO ratio ignored the sampling temperature** (`doomfly/grpo.py`, fixed in `97528ef`).
   `logp_old` came from `logits / T`, but the update used `log_softmax(logits)`. With T ≠ 1 the
   ratio was ≠ 1 before any gradient step: at T = 1.2 on a CPU check, 14 % of samples were clipped
   on the first minibatch. Only the `temp1.2` sweep run was affected (`docs/grpo-study.md`).
9. **Two different step-60000 connectome students exist.** `run_grpo.sh` defaults `STUDENT` to
   `l40s-v2/runs/malecns49k/final` (ETag `815f4377…-50`). `checkpoints/malecns49k_v2_final`, the
   `pi_ref` of the zero-shot tables, is `g6-12xl-use2-v2/runs/malecns49k/final` (`4143e256…-50`).
   On 30 held-out seeds the l40s-v2 student starts 0.87 lower (p = 0.048). Name a checkpoint by
   S3 path and ETag whenever numbers from two documents are compared.
10. **Orphaned ViZDoom engines ignore SIGTERM.** Nine `vizdoom` processes outlived their Python
    parents (PPID 1, 6-10 days old) after interrupted local evals. They stayed alive after
    `kill -TERM` and needed `kill -KILL`. After any interrupted local run, check `pgrep -fl vizdoom`.
11. **`Error: read EADDRNOTAVAIL` after the laptop wakes from sleep.** A connection that the agent
    session opened before sleep read from a local address that no longer existed. The network was
    healthy after wake, and the AWS runs were not affected. Sending the message again was enough.
