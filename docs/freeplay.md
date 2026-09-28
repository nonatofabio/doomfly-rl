# Free play on Freedoom II MAP01: zero-shot baselines, head surgery, GRPO

Plan and rationale: `docs/freeplay-plan.md`. Raw per-episode data:
`docs/results/freeplay_zero_shot_malecns49k.json`.

Setup: `freedoom2` MAP01, skill 3, frame skip 4, episodes capped at 1024 decisions (4096 tics ≈ 2 min game
time), 10 episodes per condition with the same 10 seeds across conditions, actions sampled at temperature 1.0.
Policy is `pi_ref` (`checkpoints/malecns49k_v2_final`, step 60000, MaleCNS-49k connectome). No training on
MAP01. The map has no shaped reward; the native `map_exit_reward` was never earned, so `return` is 0 everywhere.

| condition | kills | items | damage dealt | final health | died | decisions | displacement | path length |
|---|---|---|---|---|---|---|---|---|
| random (deadly_corridor legal set) | 0.7±0.5 | 0.2±0.6 | 1.5±4.5 | 5.4±3.7 | 1.0±0.0 | 297±86 | 749±242 | 2348±694 |
| `pi_ref` as `deadly_corridor` (7 buttons incl. ATTACK) | 0.8±0.4 | 2.6±1.8 | 16.8±12.4 | 24.8±37.7 | 0.8±0.4 | 659±260 | 590±218 | 4036±1328 |
| `pi_ref` as `health_gathering` (move/turn only) | 0.2±0.4 | 1.1±1.3 | 0.0±0.0 | 73.2±41.6 | 0.3±0.5 | 901±189 | 602±149 | 13790±2934 |

Per-episode (10 seeds, same order in every row):

| condition | kills | died | decisions |
|---|---|---|---|
| random | 1 1 1 1 1 0 1 0 0 1 | all | 227 319 399 319 432 214 369 151 226 316 |
| as `deadly_corridor` | 1 1 1 1 0 0 1 1 1 1 | 1 1 1 1 0 0 1 1 1 1 | 930 655 718 300 1024 1024 512 698 305 419 |
| as `health_gathering` | 0 1 0 1 0 0 0 0 0 0 | 0 1 0 1 0 0 0 0 0 1 | 1024 591 1024 592 1024 1024 1024 1024 1024 657 |

## Reading

- **Random dies every time within ~300 decisions** (~1 min game time) with ≤1 kill. That is the floor.
- **`pi_ref` with the `deadly_corridor` prior survives 2.2× longer** (659 vs 297 decisions), collects items
  (2.6 vs 0.2) and deals 11× the damage (16.8 vs 1.5), but does not kill more (0.8 vs 0.7) and still dies in
  8/10 episodes. The two survivors are the two 1024-decision episodes with zero kills: it survives by not
  finding the monsters, not by beating them. (Which monster dies is not logged; MAP01's opening rooms hold
  zombiemen and imps.)
- **`pi_ref` with the `health_gathering` prior survives longest** (901 decisions, 7/10 episodes to the cap,
  final health 73) and covers 3.4× the path of the `deadly_corridor` prior at the same net displacement: it
  runs in circles fast, as trained, without attacking. The 0.2 kills happened with no ATTACK button in the
  legal set, so they came from something other than the player's gun (barrel or infighting; not logged). This is the "avoid everything" strategy and it is the best zero-shot survival policy.
- Displacement (~600–750 units from spawn) is the same for all three: nobody leaves the first two rooms.
  MAP01's exit is several rooms and a door away; `exit = 0` in all 30 episodes.

## What this fixes for training

1. The student transfers *something*: survival time and item pickup rise well above random with the
   `deadly_corridor` prior. But scenario skills do not compose: the aiming prior and the evasion prior live in
   different scenario embeddings and neither alone plays Doom. A sixth, trainable scenario embedding (head
   surgery, step 2 of the plan) initialised to the mean of the five is the right starting point.
2. Reward shaping has clear signal on every component from step one: kills (0–1 → headroom), items (0–4; *corrected 2026-09-27*: 4 is the maximum in every episode recorded, not 5),
   damage (0–40), survival (300–1024 decisions), exploration (~600 units of displacement, 2 rooms). Exit is
   unreachable for the initial policy and should stay a bonus, not the objective.
3. 1024 decisions is long enough: the good episodes hit the cap without leaving the start area. For GRPO
   segments 512 decisions is enough to separate the death-in-300 tail from survivors.
4. CPU cost: 18,565 decisions in 6,075 s = 327 ms per decision (49k-neuron model forward + 4 game tics) on
   Apple silicon; 30 episodes took 1 h 41 min. Training needs the GPU and vectorised envs (plan: 32 envs on one L40S).

## Backbone controls, same protocol

Same 10 seeds, same conditions, on the step-60000 control checkpoints from `docs/controls.md`. Raw data:
`docs/results/freeplay_zero_shot_malecns49k_noconn.json`, `…_shuffled_s0.json`. The `random` condition's
per-episode records are identical in all three files (same seeds, same legal set; only `wall_s` differs),
which is the determinism check for the protocol.

| backbone, prior | kills | items | damage dealt | final health | died | decisions | displacement | path length |
|---|---|---|---|---|---|---|---|---|
| connectome, `deadly_corridor` | 0.8±0.4 | 2.6±1.8 | 16.8±12.4 | 24.8±37.7 | 0.8±0.4 | 658±260 | 590±218 | 4036±1328 |
| no-connectome, `deadly_corridor` | 0.9±0.3 | 2.2±1.0 | 2.5±5.1 | 5.0±3.1 | 1.0±0.0 | 600±180 | 394±285 | 4560±910 |
| shuffled s0, `deadly_corridor` | 1.2±0.6 | 3.0±1.3 | 14.0±16.7 | 32.0±35.4 | 0.6±0.5 | 699±302 | 450±317 | 3315±933 |
| connectome, `health_gathering` | 0.2±0.4 | 1.1±1.3 | 0.0±0.0 | 73.2±41.6 | 0.3±0.5 | 901±189 | 602±148 | 13790±2934 |
| no-connectome, `health_gathering` | 0.2±0.4 | 0.4±0.8 | 0.0±0.0 | 82.1±36.3 | 0.2±0.4 | 934±205 | 287±228 | 7056±1603 |
| shuffled s0, `health_gathering` | 0.4±0.5 | 2.2±1.7 | 0.0±0.0 | 56.9±44.4 | 0.4±0.5 | 842±252 | 624±193 | 8729±3714 |

Per-episode (same seed order as above):

| backbone, prior | kills | died | decisions |
|---|---|---|---|
| no-connectome, `deadly_corridor` | 1 1 1 0 1 1 1 1 1 1 | all | 739 436 473 262 822 853 510 508 711 681 |
| no-connectome, `health_gathering` | 1 0 0 0 1 0 0 0 0 0 | 1 0 0 0 1 0 0 0 0 0 | 354 1024 1024 1024 799 1024 1024 1024 1024 1024 |
| shuffled s0, `deadly_corridor` | 1 2 1 0 2 1 1 1 1 2 | 1 0 1 0 0 1 1 1 1 0 | 374 1024 798 1024 1024 536 279 613 293 1024 |
| shuffled s0, `health_gathering` | 0 1 1 0 0 0 0 0 1 1 | 0 1 0 1 0 0 0 0 1 1 | 1024 484 1024 363 1024 1024 1024 1024 565 864 |

Reading: with n=10 and per-episode std this large, the three backbones are the same on kills, survival time
and deaths (kills 0.8 / 0.9 / 1.2 with std 0.3–0.6; decisions 658 / 600 / 699 with std 180–300). Shuffled
s0 has the best point estimates on kills (3 episodes with 2 kills, the only ones in any run) and survivals
(4/10), but its 95 % interval on kills (~0.8–1.6) overlaps the other two. The one gap above noise is damage
dealt with the `deadly_corridor` prior: 16.8±12.4 (connectome) and 14.0±16.7 (shuffled) vs 2.5±5.1
(no-connectome). Both sparse backbones fire at monsters; the dense one runs more (path 4560 vs 4036/3315)
and dies every time. Nobody kills more than random beyond noise. This is consistent with `docs/controls.md`:
the backbones are interchangeable on the trained scenarios, and they start free play from the same place.
Anything that separates them later has to come from the GRPO learning curves, not from the starting point.

*Correction (2026-09-27).* At n = 30 held-out seeds the start points are not all level. With the shaped
reward and the GRPO iter-0 init (see "GRPO on MAP01" below) applied to the zero-shot `pi_ref` above, shuffled s0
starts ahead of the connectome by +1.36 return (95 % CI +0.46 to +2.20, p = 0.007). The connectome and
no-connectome start level (−0.14, p = 0.78). The n = 10 tables above could not resolve this gap. The connectome
GRPO run started from a different, lower-scoring student; see "Start points" below.

Cost (`wall_s` in the JSON, 20 policy episodes each, Apple silicon CPU): connectome 5,974 s, shuffled s0
6,347 s, no-connectome 45 s. The dense backbone is ~130× cheaper per decision than the 49k-neuron sparse
recurrent one on CPU (batch 1, 4 game tics included); on the L40S at training batch size the gap was 23×
(646 vs 14,686 samples/s, `docs/controls.md`).

## Head surgery (plan step 2): done

`doomfly/surgery.py` widens a trained checkpoint for free play without touching what it learned:

- `n_actions` 22 → 23: `SELECT_NEXT_WEAPON` is global action 22, legal only in `FREEPLAY`. The new policy
  row (weight and bias) is zero. The legal-mask fill is `-1e4`, so on the five trained scenarios the new
  logit is masked and the softmax over the old 22 is unchanged in fp32.
- `n_scenarios` 5 → 6: `scenario_emb` row 5 (`FREEPLAY.id`) is the mean of the five trained rows. Rows 0–4
  are untouched and row 5 is only looked up when `scenario_id == 5`.
- `load_flynet(ckpt, connectome)` is now the one loader (`evaluate.py`, `grpo.py`, `freeplay_zero_shot.py`):
  it reads the checkpoint's own `config.json`, expands old 22/5 state dicts in memory, and loads strictly.
  `python -m doomfly.surgery --ckpt … --connectome … --out …` writes a converted copy with a `surgery`
  block in `config.json`. `FlyNetConfig` defaults are now 23/6, so new runs are born post-surgery.
- `train.py`: the legal table has 6 rows; stored 22-wide teacher arrays are zero-padded to 23 on load.
  `--resume` from a pre-surgery checkpoint is not supported (the optimizer state has the old shapes).

Verification (`tests/test_surgery.py`, 9 tests, plus the real checkpoints):

- Synthetic 64-neuron connectome, both backbones: old-action logits, value and readout bit-identical on
  scenario ids 0–4 before/after surgery; action 22 masked at `-1e4`; row 5 equals the mean; shrinking refused.
- `checkpoints/malecns49k_noconn` converted and run through `evaluate.py --greedy --episodes 10` (seed
  12345) on original and converted checkpoint: JSON output byte-identical (`basic` 78.2±7.5,
  `defend_the_center` 20.8±2.4, `health_gathering` 1018.4±619.7, `deadly_corridor` 2280.8±2.5,
  `defend_the_line` 24.3±6.1).
- `checkpoints/malecns49k_v2_final` (`pi_ref`, real 49k connectome) converted: forward pass on random
  frames bit-identical on scenario ids 0–4; the free-play row exposes exactly `{0..16, 22}`.

Param count after surgery: 48,161,837 (`pi_ref`), 27,730,055 (no-connectome); +1,025 each (512+1 for the
policy row, 512 for the embedding row).

## GRPO on MAP01 (plan step 3): all three backbones learn and plateau in the start area; the exit needs USE, which the action set lacks

Runs: `doomfly/grpo_freeplay.py` at commit `97528ef`, launched 2026-09-26, one g6e.4xlarge (1× L40S) per
backbone in us-east-2c, S3 prefixes `freeplay-grpo-{conn,shuffled-s0,noconn}`. The recipe is identical; only the
starting checkpoint differs. 500 iterations × 32 episodes (4 seed-matched groups of 8), 512-decision cap, RLOO
baseline, std-normalised advantage, PPO-clip 0.2, β = 0 (no KL), entropy 0.01, temperature 1.2 → 1.0, lr 3e-5 /
lr_conn 3e-4, training seed 0. Iter-0 policy (`apply_init`): the FREEPLAY embedding row is copied from
`deadly_corridor`, the SELECT_NEXT_WEAPON bias is −4. Reward: +1 per kill, item or secret; +0.01 per damage point
dealt; −0.01 per HP lost (death costs the remaining HP); +0.1 per new 128-unit cell; +10 for the exit. All three
boxes ended `ALL DONE (grpo) fail=0` and self-terminated.

| backbone | start checkpoint (S3 ETag) | final checkpoint (S3 ETag) |
|---|---|---|
| connectome | `l40s-v2/runs/malecns49k/final` (`815f4377…-50`) | `freeplay-grpo-conn/runs/malecns49k_grpo_freeplay/final` (`272682cf…-50`) |
| shuffled s0 | `controls-shuffled-s0/runs/malecns49k_shuffled_s0/final` (`f9387fe3…-50`) | `freeplay-grpo-shuffled-s0/runs/malecns49k_shuffled_s0_grpo_freeplay/final` (`41414780…-50`) |
| no-connectome | `controls-noconn/runs/malecns49k_noconn/final` (`806d5453…-14`) | `freeplay-grpo-noconn/runs/malecns49k_noconn_grpo_freeplay/final` (`38fcf488…-14`) |

Local copies: `checkpoints/malecns49k_l40s_v2_final`, `checkpoints/malecns49k{,_shuffled_s0,_noconn}_grpo_freeplay_final`
(sizes and ETags match S3). The connectome start is not the `pi_ref` of the zero-shot tables above; see
"Start points" below.

### Protocol

The in-run eval (`eval.jsonl`, 10 episodes every 25 iterations) reuses the zero-shot seeds, `default_rng(0)`. The
training seed stream is also `default_rng(0)`, so those 10 seeds are exactly the training seeds of iterations 1–3
(`docs/harness-findings.md`, finding 7). The in-run eval is used for curve shape only. Endpoint numbers come from
a held-out eval:

```
python -m doomfly.freeplay_eval --ckpt checkpoints/<ckpt> --connectome data/processed/connectome_malecns49k.npz \
    --out docs/results/freeplay_heldout_<tag>_<init|final>.json [--init] --device cpu
```

30 episodes on seed stream 1 (`default_rng(1)`; none of its seeds is among the 2,000 training seeds), 1024
decisions, sampled at temperature 1.0, same shaped reward. `--init` applies the iter-0 policy to a distilled
student; GRPO outputs are scored as saved. All files share the same 30 seeds, so differences are paired by seed.
Pairing removes map-RNG variance, not training-run variance: there is one GRPO run per backbone, so a
between-backbone gap is "this run minus that run". CI = bootstrap 95 % (20,000 resamples); p = two-sided
sign-flip permutation test, floor 5e-5. `hp lost` is the shaped-reward health total (HP lost; death counts the
remaining HP), not final HP. All tables below: `python scripts/freeplay_compare.py` (reads `docs/results/` and
the run logs pulled from S3 to `runs/`).

### Reading

1. **Every backbone learns.** Held-out return rises by +2.06 to +4.24 (p = 5e-5 for all three). Kills rise by
   +0.6 to +0.9, items to 3.8–4.0, exploration by +4.6 to +9.3 cells, and deaths fall by 0.27 to 0.40 (the
   connectome's −0.27 at p = 0.075). All three in-run curves are within 10 % of their final level by iteration 225.
2. **No backbone reaches the exit or a secret**: 0 of 180 held-out episodes, 0 of 48,000 training episodes
   (per-iteration maximum of `c_exit` and `c_secret` = 0). This is structural. MAP01 has one exit, linedef 774,
   special 11 (S1 switch exit), which needs USE, and no secret exit. 16 of its linedef specials are
   use-activated: the exit, doors (117 ×6, 31 ×4, 63, 103) and lifts (123 ×3). FREEPLAY's legal set
   `{0..16, 22}` has no USE action, so the policy cannot press the exit or open a use-activated door. The
   remaining specials are walk-triggered (90 ×2 WR door, 120 ×4 WR lift, 97 ×7 WR teleport, 19 W1 floor) or
   passive (48 ×3 scroller); whether any of them lies on a path from the start is not checked here
   (`python scripts/map01_specials.py`; `freedoom2.wad` sha256 `a8772e08…`; vizdoom 1.3.0 locally, 1.3.1 on the boxes).
3. **The reachable reward is small and nearly saturated.** Items never exceed 4 and kills never exceed 2 in any
   of the 180 held-out init/final episodes or the 90 zero-shot episodes. At iteration 500, 27–29 of 30 held-out
   episodes per backbone take all 4 items. What remains to learn in the start area is damage, survival and cells.
4. **Between backbones, the no-connectome run ends ahead; connectome and shuffled end level.** Final return:
   no-connectome − connectome +0.83 (p = 0.044), no-connectome − shuffled s0 +0.69 (p = 0.047), shuffled s0 −
   connectome +0.14 (p = 0.73). Exploration is the clearer gap: +3.8 and +3.5 cells (p ≤ 0.002). The return gaps
   are directional, not established: they are run-vs-run with one run per backbone, and they sit near p = 0.05
   among 24 between-backbone tests (8 metrics × 3 pairs). No p-value in this document is corrected for
   multiplicity.
5. **Learning curves: the wiring does not help.** The connectome and no-connectome runs start level (+0.73,
   p = 0.15) and gain the same (+0.10, p = 0.88). Shuffled s0 starts +2.23 ahead of the connectome
   (p = 0.00045) and gains 2.09 less (p = 0.005): it starts nearer the cap. The connectome run starts lowest in
   the in-run eval, reaches 90 % of its final level last (iteration 225, against 75 and 125) and has the lowest
   mean over the 21 evals (5.54, against 6.16 and 6.47). Its training entropy also fell furthest (0.25–0.26 in
   iterations 101–300, against ≥ 0.53 for the others) before it recovered to 0.56. With one run per backbone, a
   slower or collapsing curve is not yet a property of the wiring.
6. **Cost.** 500 iterations took 5.36 h (connectome), 6.02 h (shuffled s0) and 3.76 h (no-connectome) on one
   L40S. The update step took 15.0, 17.0 and 1.0 s per iteration. The dense backbone learns at least as well at
   70 % of the connectome's wall time.

### Start points

The connectome GRPO run started from `l40s-v2/runs/malecns49k/final`, the `run_grpo.sh` default. That is the run
whose recipe the two controls reproduce (`docs/controls.md`). The zero-shot tables above use
`checkpoints/malecns49k_v2_final` = `g6-12xl-use2-v2/runs/malecns49k/final`: same recipe and step, different
weights (`docs/grpo-study.md` already records this for the scenario GRPO sweep). On the held-out seeds, with the
iter-0 init, the `l40s-v2` student starts lower: return 2.42 against 3.29 (−0.87, p = 0.048), exploration 10.1
against 15.0 cells (p = 0.00015). Measured against the `g6-12xl-use2-v2` student, shuffled s0 leads by +1.36
(p = 0.007) and no-connectome is level (−0.14, p = 0.78). Measured against the student the run started from,
shuffled s0 leads by +2.23 and no-connectome by +0.73 (p = 0.15). Both sets are in the tables.

### Tables

Held-out eval: n=30 seeds, stream 1.

| backbone | stage | R | kill | item | damage | hp lost | explore | T | dead |
|---|---|---|---|---|---|---|---|---|---|
| connectome | init | 2.42±1.89 | 0.73±0.57 | 1.40±1.47 | 4.40±8.37 | 76.73±41.06 | 10.10±2.94 | 619.40±312.27 | 0.70±0.46 |
| connectome | final | 6.57±1.57 | 1.53±0.56 | 3.80±0.65 | 13.07±10.30 | 84.20±29.43 | 19.43±4.57 | 715.83±362.99 | 0.43±0.50 |
| shuffled s0 | init | 4.65±2.01 | 0.90±0.47 | 3.00±1.63 | 9.30±12.07 | 85.43±39.61 | 15.10±3.85 | 586.40±262.94 | 0.80±0.40 |
| shuffled s0 | final | 6.71±1.63 | 1.47±0.62 | 3.83±0.58 | 16.93±14.56 | 73.47±39.47 | 19.73±4.69 | 725.63±325.76 | 0.47±0.50 |
| no-connectome | init | 3.15±1.97 | 0.73±0.57 | 1.57±1.43 | 3.57±7.31 | 84.17±36.69 | 16.57±4.44 | 723.83±250.02 | 0.73±0.44 |
| no-connectome | final | 7.39±1.26 | 1.60±0.61 | 3.97±0.18 | 20.53±13.40 | 69.83±36.88 | 23.20±3.48 | 818.93±297.11 | 0.33±0.47 |

exit/secret any: {'connectome/init': False, 'connectome/final': False, 'shuffled s0/init': False, 'shuffled s0/final': False, 'no-connectome/init': False, 'no-connectome/final': False}

#### final - init, paired by seed
| metric | connectome | shuffled s0 | no-connectome |
|---|---|---|---|
| R | +4.15 [+3.30, +4.97], p=5e-05 | +2.06 [+1.24, +2.87], p=5e-05 | +4.24 [+3.34, +5.12], p=5e-05 |
| kill | +0.80 [+0.50, +1.10], p=0.0001 | +0.57 [+0.33, +0.80], p=0.00025 | +0.87 [+0.57, +1.17], p=0.0001 |
| item | +2.40 [+1.87, +2.93], p=5e-05 | +0.83 [+0.23, +1.47], p=0.02 | +2.40 [+1.90, +2.90], p=5e-05 |
| damage | +8.67 [+3.40, +13.60], p=0.0023 | +7.63 [+0.30, +15.03], p=0.058 | +16.97 [+11.60, +22.30], p=5e-05 |
| hp lost | +7.47 [-10.73, +26.47], p=0.45 | -11.97 [-32.23, +8.80], p=0.27 | -14.33 [-26.57, -2.07], p=0.029 |
| explore | +9.33 [+7.53, +11.10], p=5e-05 | +4.63 [+2.77, +6.50], p=0.00015 | +6.63 [+4.30, +8.83], p=5e-05 |
| T | +96.43 [-78.83, +274.74], p=0.3 | +139.23 [-19.50, +295.30], p=0.1 | +95.10 [-25.17, +216.10], p=0.14 |
| dead | -0.27 [-0.50, -0.03], p=0.075 | -0.33 [-0.57, -0.07], p=0.029 | -0.40 [-0.60, -0.20], p=0.0021 |

#### between backbones, paired by seed
| metric | no-connectome - connectome | no-connectome - shuffled s0 | shuffled s0 - connectome |
|---|---|---|---|
| final R | +0.83 [+0.08, +1.60], p=0.044 | +0.69 [+0.06, +1.32], p=0.047 | +0.14 [-0.62, +0.92], p=0.73 |
| final kill | +0.07 [-0.23, +0.33], p=0.83 | +0.13 [-0.13, +0.40], p=0.48 | -0.07 [-0.33, +0.20], p=0.8 |
| final item | +0.17 [-0.03, +0.43], p=0.37 | +0.13 [-0.03, +0.40], p=0.5 | +0.03 [-0.30, +0.37], p=1 |
| final damage | +7.47 [+1.07, +13.93], p=0.036 | +3.60 [-3.77, +10.73], p=0.35 | +3.87 [-1.70, +9.63], p=0.2 |
| final hp lost | -14.37 [-30.17, +0.53], p=0.081 | -3.63 [-21.63, +14.40], p=0.7 | -10.73 [-26.47, +5.00], p=0.21 |
| final explore | +3.77 [+1.53, +5.90], p=0.0019 | +3.47 [+1.80, +5.20], p=0.00075 | +0.30 [-2.17, +2.80], p=0.84 |
| final T | +103.10 [-66.73, +273.07], p=0.25 | +93.30 [-66.93, +249.57], p=0.27 | +9.80 [-135.97, +155.40], p=0.9 |
| final dead | -0.10 [-0.33, +0.17], p=0.61 | -0.13 [-0.37, +0.10], p=0.43 | +0.03 [-0.17, +0.27], p=1 |
| init R | +0.73 [-0.22, +1.68], p=0.15 | -1.50 [-2.48, -0.49], p=0.0077 | +2.23 [+1.23, +3.19], p=0.00045 |
| gain R | +0.10 [-1.17, +1.37], p=0.88 | +2.18 [+1.03, +3.37], p=0.0011 | -2.09 [-3.33, -0.78], p=0.0052 |

#### connectome students at init: l40s-v2 (GRPO start) - g6-12xl-use2-v2 (zero-shot pi_ref)
| metric | g6 v2 | l40s v2 | diff |
|---|---|---|---|
| R | 3.29 | 2.42 | -0.87 [-1.68, -0.05], p=0.048 |
| kill | 0.73 | 0.73 | +0.00 [-0.30, +0.30], p=1 |
| item | 1.90 | 1.40 | -0.50 [-1.10, +0.10], p=0.15 |
| damage | 3.53 | 4.40 | +0.87 [-2.50, +4.63], p=0.66 |
| hp lost | 87.93 | 76.73 | -11.20 [-30.57, +8.63], p=0.28 |
| explore | 15.00 | 10.10 | -4.90 [-6.83, -2.87], p=0.00015 |
| T | 687.00 | 619.40 | -67.60 [-217.94, +85.93], p=0.39 |
| dead | 0.77 | 0.70 | -0.07 [-0.30, +0.17], p=0.79 |

| init R, minus the g6 v2 student | shuffled s0 | no-connectome |
|---|---|---|
| diff | +1.36 [+0.46, +2.20], p=0.0071 | -0.14 [-1.13, +0.82], p=0.78 |

#### in-run eval R (10 episodes, seeds default_rng(0), every 25 iters)
connectome: iter0 2.53, last-4 mean 6.31, mean over 21 evals 5.54, first eval >= 90% of last-4 at iter 225
  2.5 3.7 5.3 5.0 4.9 5.1 5.5 4.9 5.4 6.5 5.9 5.6 6.4 6.0 6.2 6.2 6.1 6.3 6.1 6.6 6.3
shuffled s0: iter0 3.78, last-4 mean 6.56, mean over 21 evals 6.16, first eval >= 90% of last-4 at iter 75
  3.8 5.8 5.4 6.5 5.3 6.8 6.0 5.5 5.8 6.2 6.3 5.9 6.4 6.3 7.1 6.8 7.1 6.6 7.0 6.5 6.1
no-connectome: iter0 3.77, last-4 mean 7.17, mean over 21 evals 6.47, first eval >= 90% of last-4 at iter 125
  3.8 4.0 6.1 6.2 6.4 7.1 6.1 6.8 6.9 6.9 7.0 6.5 6.8 6.3 7.2 6.7 6.5 6.4 7.7 7.6 7.0

#### training blocks (100 iters = 3200 episodes, 512-decision cap)
| backbone | iters | R | ep_len | dead | ent | kill | item | damage | hp lost | explore |
|---|---|---|---|---|---|---|---|---|---|---|
| connectome | 1-100 | 4.70 | 346 | 0.82 | 0.45 | 0.96 | 3.46 | 3.9 | 96.0 | 12.1 |
| connectome | 101-200 | 5.53 | 366 | 0.78 | 0.25 | 1.08 | 3.87 | 6.7 | 96.8 | 14.7 |
| connectome | 201-300 | 6.03 | 403 | 0.61 | 0.26 | 1.18 | 3.94 | 9.5 | 93.9 | 17.5 |
| connectome | 301-400 | 6.11 | 411 | 0.54 | 0.42 | 1.25 | 3.92 | 11.5 | 89.6 | 17.3 |
| connectome | 401-500 | 6.34 | 421 | 0.48 | 0.56 | 1.37 | 3.90 | 13.9 | 85.6 | 17.9 |
| connectome | wall | 5.36 h, 6,229,921 frames, t_sample 22.9 s, t_update 15.0 s | | | | | | | | |
| connectome | max | exit 0, secret 0, item 4.00 | | | | | | | | |
| shuffled s0 | 1-100 | 5.75 | 415 | 0.57 | 0.69 | 1.16 | 3.63 | 11.0 | 84.8 | 17.0 |
| shuffled s0 | 101-200 | 6.01 | 435 | 0.45 | 0.53 | 1.24 | 3.66 | 10.0 | 80.7 | 18.2 |
| shuffled s0 | 201-300 | 5.86 | 414 | 0.52 | 0.58 | 1.30 | 3.61 | 11.6 | 84.4 | 16.7 |
| shuffled s0 | 301-400 | 6.17 | 437 | 0.40 | 0.74 | 1.39 | 3.60 | 15.9 | 78.8 | 18.1 |
| shuffled s0 | 401-500 | 6.16 | 441 | 0.40 | 0.86 | 1.35 | 3.58 | 15.1 | 78.0 | 18.6 |
| shuffled s0 | wall | 6.02 h, 6,856,433 frames, t_sample 25.5 s, t_update 17.0 s | | | | | | | | |
| shuffled s0 | max | exit 0, secret 0, item 4.00 | | | | | | | | |
| no-connectome | 1-100 | 5.34 | 448 | 0.46 | 0.83 | 1.08 | 3.09 | 6.3 | 83.2 | 19.4 |
| no-connectome | 101-200 | 6.33 | 461 | 0.36 | 0.62 | 1.28 | 3.63 | 11.2 | 79.0 | 21.0 |
| no-connectome | 201-300 | 6.49 | 463 | 0.33 | 0.65 | 1.28 | 3.73 | 12.7 | 78.3 | 21.3 |
| no-connectome | 301-400 | 6.59 | 459 | 0.36 | 0.67 | 1.31 | 3.74 | 13.6 | 77.6 | 21.9 |
| no-connectome | 401-500 | 6.86 | 465 | 0.31 | 0.70 | 1.42 | 3.77 | 18.2 | 75.6 | 22.4 |
| no-connectome | wall | 3.76 h, 7,344,724 frames, t_sample 25.4 s, t_update 1.0 s | | | | | | | | |
| no-connectome | max | exit 0, secret 0, item 4.00 | | | | | | | | |

### What this means for the plan

- Step 3 asked whether the learning curves differ by backbone. For this task the first answer is no advantage
  from the connectome: the dense backbone learns at least as well, for less compute. The answer rests on one
  training seed per backbone, on a task where every run plateaus in the start area and the exit is out of
  reach of the action set.
- Step 4 (fly-vs-fly) is gated on "step 3 learns". It learns, but only the start area. Two single changes would
  each sharpen the step-3 answer first: (a) add USE and FWD+USE (global actions 20 and 21, already in the head,
  untrained) to FREEPLAY, so doors and the exit become reachable; (b) two more training seeds per backbone on the
  current recipe, so between-backbone gaps become backbone-vs-backbone rather than run-vs-run. Neither is launched.
