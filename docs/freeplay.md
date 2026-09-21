# Free play on Freedoom II MAP01: zero-shot baselines

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
2. Reward shaping has clear signal on every component from step one: kills (0–1 → headroom), items (0–5),
   damage (0–40), survival (300–1024 decisions), exploration (~600 units of displacement, 2 rooms). Exit is
   unreachable for the initial policy and should stay a bonus, not the objective.
3. 1024 decisions is long enough: the good episodes hit the cap without leaving the start area. For GRPO
   segments 512 decisions is enough to separate the death-in-300 tail from survivors.
4. CPU cost: 18,565 decisions in 6,075 s = 327 ms per decision (49k-neuron model forward + 4 game tics) on
   Apple silicon; 30 episodes took 1 h 41 min. Training needs the GPU and vectorised envs (plan: 32 envs on one L40S).

## Backbone controls, same protocol

Same 10 seeds, same conditions, on the step-60000 control checkpoints from `docs/controls.md`. Raw data:
`docs/results/freeplay_zero_shot_malecns49k_noconn.json`, `…_shuffled_s0.json`. The `random` row is
bit-identical across runs (same seeds, same legal set), which is the determinism check for the protocol.

| backbone, prior | kills | items | damage dealt | final health | died | decisions | displacement | path length |
|---|---|---|---|---|---|---|---|---|
| connectome, `deadly_corridor` | 0.8±0.4 | 2.6±1.8 | 16.8±12.4 | 24.8±37.7 | 0.8±0.4 | 658±260 | 590±218 | 4036±1328 |
| no-connectome, `deadly_corridor` | 0.9±0.3 | 2.2±1.0 | 2.5±5.1 | 5.0±3.1 | 1.0±0.0 | 600±180 | 394±285 | 4560±910 |
| shuffled s0, `deadly_corridor` | _running_ | | | | | | | |
| connectome, `health_gathering` | 0.2±0.4 | 1.1±1.3 | 0.0±0.0 | 73.2±41.6 | 0.3±0.5 | 901±189 | 602±148 | 13790±2934 |
| no-connectome, `health_gathering` | 0.2±0.4 | 0.4±0.8 | 0.0±0.0 | 82.1±36.3 | 0.2±0.4 | 934±205 | 287±228 | 7056±1603 |
| shuffled s0, `health_gathering` | _running_ | | | | | | | |

Per-episode, no-connectome (same seed order as above):

| condition | kills | died | decisions |
|---|---|---|---|
| as `deadly_corridor` | 1 1 1 0 1 1 1 1 1 1 | all | 739 436 473 262 822 853 510 508 711 681 |
| as `health_gathering` | 1 0 0 0 1 0 0 0 0 0 | 1 0 0 0 1 0 0 0 0 0 | 354 1024 1024 1024 799 1024 1024 1024 1024 1024 |

Reading: with n=10 and per-episode std this large, the two backbones are the same on kills, survival time
and deaths. The one gap above noise is damage dealt with the `deadly_corridor` prior (16.8±12.4 vs 2.5±5.1):
the connectome student fires at monsters more, the no-connectome student runs more (path 4560 vs 4036) and
dies every time. Neither kills more than random. This is consistent with `docs/controls.md`: the backbones
are interchangeable on the trained scenarios, and they start free play from the same place. Anything that
separates them later has to come from the GRPO learning curves, not from the starting point.

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
