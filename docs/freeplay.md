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

Same protocol is due on the shuffled-s0 and no-connectome checkpoints before GRPO (the paper compares the
three backbones' *learning curves*, so their zero-shot starting points must be on record).
