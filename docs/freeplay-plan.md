# Free play on Freedoom II: plan

Status: steps 1 and 2 done (`docs/freeplay.md`). Step 3 implemented as
`doomfly/grpo_freeplay.py` and launched on all three backbones. Budget: 500 iterations,
not ~1000. At the sweep's measured L40S cost (update ≈4.4 ms/sample, sampling ≈75 ms per
32-env step), one 32 × 512 iteration takes ≈110 s, so 500 iterations ≈15 h already fills
the 10–15 h budget.

## Why

The five ViZDoom scenarios are saturated: the distilled student matches both PPO teachers, and twelve GRPO runs
could not move it (`docs/grpo-study.md`). The controls (`docs/controls.md`) show that in this regime the
connectome contributes nothing measurable: shuffled and no-connectome students are identical. To test whether
wiring matters for *learning* rather than *imitation*, we need a task where the student starts far from
optimal and has to improve by RL. A real level is that task.

## Environment

- `freedoom2.cfg`, `doom_map MAP01`, skill 3, frame skip 4, RGB 320×240 downsampled to the usual 4×72×96
  stack. `map_exit_reward = 1` is the only native reward.
- `FREEPLAY` scenario in `doomfly/doom/actions.py`: deadly_corridor's 7-button set (ATTACK, LEFT, RIGHT, FWD,
  BACK, TURN_L, TURN_R) so the student's trained action rows 0–16 map directly. Not in `SCENARIOS`, so
  `N_SCENARIOS` stays 5 and every existing checkpoint loads.
- `DoomEnv.step` returns `info["vars"]` (KILLCOUNT, ITEMCOUNT, SECRETCOUNT, DAMAGECOUNT, HEALTH, ARMOR,
  POSITION_X/Y) and `info["dead"]` when a map is set.
- Episodes are capped at a fixed number of decisions (512–1024) so groups are length-matched.

## Step 1: zero-shot baselines (this week)

Conditions, 10 episodes each, 1024 decisions, sampled (not greedy):
random uniform over the legal set; `pi_ref` borrowing `deadly_corridor`'s scenario id and mask;
`pi_ref` borrowing `health_gathering`'s (move/turn only). Report kills, items, secrets, damage, final health,
death, displacement, path length, exit. Output: `docs/results/freeplay_zero_shot_<backbone>.json`,
summary in `docs/freeplay.md`.

## Step 2: head surgery

- Add `SELECT_NEXT_WEAPON` (n_actions 22 → 23, zero-initialised policy row).
- Add a sixth `scenario_emb` row initialised to the mean of the five trained rows.
- Verify the surgery is a no-op on the five scenarios (eval must reproduce `pi_ref` numbers).

## Step 3: single-policy GRPO on MAP01

- Init from the scenario-distilled student (each backbone from its own checkpoint).
- 32 envs, seed-matched groups of 8, 512-decision segments, PPO-clip, **no KL to the scenario student**
  (β = 0, or lagged-KL to a slow EMA of the policy), entropy 0.01, temperature 1.2 → 1.0.
- Shaped reward, each component logged separately: +kills, +items, +secrets, +damage/100, −health loss/100,
  +exploration (new 128-unit POSITION cell), +exit bonus.
- Same recipe on three backbones: connectome, shuffled s0, no-connectome. The paper question is whether the
  learning curves differ, not the end point.
- Checkpoint eval: 10+ episodes, mean ± std for every reward component.
- Budget: ~1000 iterations, 10–15 h per backbone on one L40S.

## Step 4 (only if step 3 learns): fly-vs-fly

ViZDoom multiplayer duel, frozen-snapshot opponents / small league, frag-differential reward, Elo.
Risks: lock-stepped localhost throughput, one engine process per player, `multi_duel` is 1-D.

## Rules

One change per run. Report deltas with `std_return`. Negative results in `docs/`. Every number from a
checkpoint.
