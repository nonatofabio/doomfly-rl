# GRPO study: critic-free RL vs. the PPO teachers

Status after the first one-knob sweep (12 runs, 300 iterations each, backbone `malecns49k`).
Every number below comes from an `eval.jsonl`, `metrics.jsonl`, or teacher curve in
`s3://doomfly-047472448415-us-west-2/`. Paths are given per table.

## 1. Question

Project goal #2: does critic-free GRPO/RLOO let the connectome-constrained net improve *past*
the PPO teachers it was distilled from, and how does that depend on group size, KL coefficient,
learning rates, and the other knobs?

Short answer so far: **no**. After 300 iterations (~1.7 M frames per run) no GRPO run beats the
distilled student outside eval noise on any scenario. Several runs make `defend_the_center`
worse by more than 5 standard deviations. The KL penalty to the frozen student is the only thing
that keeps the policy from drifting away, and at the strength that stops the drift (β = 0.2) it
also stops any gain.

## 2. The two algorithms as they exist in this repo

| | PPO teachers (`doomfly/doom/teacher.py`) | GRPO on FlyNet (`doomfly/grpo.py`) |
|---|---|---|
| Policy | SB3 `CnnPolicy`, one model per scenario, trained from scratch | FlyNet (49,393 neurons, 9.05 M edges, 48.2 M params), one model for all five scenarios, initialised from the distilled student |
| Critic | Yes: shared-trunk value head, GAE (λ 0.95, γ 0.99), `vf_coef 0.5` | None. The student's 64-bin value head is its weakest part, so the advantage comes from the group instead |
| Advantage | GAE per step, bootstrapped | Per episode. Group = G envs reset with the same ViZDoom seed. RLOO `A_i = R_i − mean_{j≠i} R_j` (default) or GRPO `A_i = (R_i − mean)/(std+ε)`; normalised by group std |
| Loss | clipped ratio · A + `vf_coef`·value MSE − `ent_coef`·H | clipped ratio · A − β · KL(π‖π_ref) − `ent_coef`·H. No value loss, no bootstrap |
| Reference | none | π_ref = frozen copy of the student (`ref = deepcopy(model).eval()`) |
| Clip | 0.1 | 0.2 |
| Entropy coef | 0.01 | 0.0 |
| LR | 2.5e-4 | 3e-5 (stem/readout), 3e-4 (connectome edge weights) |
| Batch | 8 envs × 256 steps, 4 epochs, minibatch 512 | 4 groups × 8 episodes (≤ 600 steps each), 2 epochs, minibatch 256 |
| Frames per run | basic 1 M, dtc 1.5 M, hg 1.5 M, dc 8 M, dtl 1.5 M | ≈ 1.64–1.87 M total across all five scenarios (300 iterations, round-robin) |
| Reward | `reward_scale 0.01` on basic and dc, shaping on dc (doom_skill 3) | raw scenario return |
| Throughput | basic: 1 M frames / 677 s ≈ 1.5 k fps; dc: 8 M / 4151 s ≈ 1.9 k fps | 1.64 M / 6.4 h ≈ 71 fps (four runs sharing one 4×A10G box); 1.87 M / 4.15 h ≈ 125 fps (one run per A10G) |

The last row is the practical difference. FlyNet runs 5 recurrent steps over a 9 M-edge sparse
matrix per decision, so sampling and especially the update are 12–27× slower per frame than the
teacher CNN. At iteration 300 on `defend_the_line` (the longest episodes) `t_sample 16.6 s`,
`t_update 81.6 s`: the update is ~5× the sampling time. Frames are not the bottleneck; the
gradient step is.

## 3. Return ladder: teacher → ε-greedy rollouts → student → GRPO

Mean return, 10 episodes unless stated. `±` is `std_return` across episodes.

| Scenario | PPO teacher (train curve, last / best) | Teacher rollouts, ε = 0.1 | Distilled student (step 60 k) | Best GRPO run at iter 300 |
|---|---|---|---|---|
| basic | 80.9 / 84.5 (t = 1.0 M) | −279.4 ± 117.0 | 78.2 ± 7.5 | 78.2 ± 7.5 (unchanged, all runs) |
| defend_the_center | — | 8.3 ± 1.5 | 18.6 ± 1.8 | 21.4 ± 1.9 (`lrconn1e-4`) |
| health_gathering | — | 1403.0 ± 719.3 | 1728.0 ± 615.5 | 2100.0 ± 0.0 (`temp1.2`; 2100 = timeout ceiling) |
| deadly_corridor | 2262.1 / 2281.1 (t = 8.0 M) | 155.8 ± 152.2 | 2280.5 ± 2.4 | 2281.8 ± 1.8 (`baselinemean`) |
| defend_the_line | — | 19.95 ± 5.01 | 21.0 ± 5.9 | 24.3 ± 3.7 (`base`) |

Sources: teacher curves `g6-12xl-use2-v2/teachers/{basic,deadly_corridor}_curve.json`
(`raw_return`, last point and max over the curve); rollouts `rollouts/<scenario>/meta.json`
(`ep_return_mean` over 824–17 864 episodes, 15 shards, ~300 k frames per scenario); student `l40s-v2/runs/malecns49k/eval.jsonl`
(step 60000); GRPO `grpo-*/runs/malecns49k_grpo_<knob>/eval.jsonl` (iter 300).

Caveats on the teacher column:

- **Only `basic` and `deadly_corridor` have teacher curves.** The `defend_the_center`,
  `health_gathering` and `defend_the_line` teachers were trained in the first `g6-12xl-use2`
  run, before the `Monitor` wrapper was added (harness-findings, teacher section); those runs
  left 0-byte logs and no curves. Only the `.zip` models exist. A greedy 10-episode eval of those
  three teachers is the missing measurement and needs a box with ViZDoom + SB3.
- The teacher numbers are training-time episode returns from the SB3 rollout workers (stochastic
  policy, 8 envs), not a held-out eval. The ε = 0.1 rollout column is the data the student was
  actually distilled from; it is a lower bound on the teacher, not the teacher.
- The student already matches the two teachers we can see (78.2 vs 80.9 on basic, 2280.5 vs
  2262.1 on deadly_corridor) and beats the ε-greedy rollout mean on all five scenarios.

## 4. GRPO run table

All runs: `--ckpt l40s-v2/runs/malecns49k/final --iters 300 --eval-every 25 --eval-episodes 10`,
defaults from §2 unless the knob column says otherwise. Eval is sampled (not greedy), 10 episodes,
fixed env seeds. Δ is GRPO iter-300 mean minus student mean. Student std for reference:
basic 7.5, dtc 1.8, hg 615.5, dc 2.4, dtl 5.9.

| Run (`malecns49k_grpo_*`) | S3 prefix | Knob | basic | dtc | hg | dc | dtl |
|---|---|---|---|---|---|---|---|
| student (iter 0) | `l40s-v2` | — | 78.2 ± 7.5 | 18.6 ± 1.8 | 1728 ± 616 | 2280.5 ± 2.4 | 21.0 ± 5.9 |
| `base` | `grpo-use2` | defaults | 78.2 (+0.0) | 10.3 ± 0.9 (**−8.3**) | 1992 ± 324 (+264) | 2281.6 ± 1.8 (+1.1) | 24.3 ± 3.7 (+3.3) |
| `beta0.01` | `grpo-use2` | `--beta 0.01` | 78.2 (+0.0) | 7.5 ± 1.5 (**−11.1**) | 968 ± 532 (**−760**) | 1698 ± 929 (**−583**) | 21.1 ± 4.9 (+0.1) |
| `beta0.2` | `grpo-use2` | `--beta 0.2` | 74.8 ± 13.5 (−3.4) | 17.5 ± 1.9 (−1.1) | 1697 ± 591 (−31) | 2281.2 ± 2.5 (+0.7) | 21.8 ± 3.9 (+0.8) |
| `g16` | `grpo-use2` | `--group-size 16` | 78.2 (+0.0) | 13.8 ± 1.5 (**−4.8**) | 1914 ± 302 (+186) | 2280.2 ± 3.4 (−0.3) | 23.8 ± 4.7 (+2.8) |
| `temp1.2` | `grpo-use1-1gpu` | `--temperature 1.2` | 78.2 (+0.0) | 14.4 ± 1.9 (**−4.2**) | 2100 ± 0 (+372) | 1167 ± 747 (**−1114**) | 18.9 ± 6.1 (−2.1) |
| `lrconn1e-4` | `grpo-use1-1gpu` | `--lr-conn 1e-4` | 78.2 (+0.0) | 21.4 ± 1.9 (+2.8) | 1802 ± 465 (+74) | 2281.5 ± 1.9 (+1.0) | 22.9 ± 4.4 (+1.9) |
| `lr1e-4` | `grpo-use1-1gpu` | `--lr 1e-4` | 78.2 (+0.0) | 12.8 ± 1.0 (**−5.8**) | 1674 ± 687 (−54) | 2129 ± 450 (−151) | 20.9 ± 6.3 (−0.1) |
| `ent0.01` | `grpo-use1-1gpu` | `--ent-coef 0.01` | 78.2 (+0.0) | 14.8 ± 0.9 (**−3.8**) | 1986 ± 343 (+258) | 2279.7 ± 2.4 (−0.8) | 22.5 ± 5.1 (+1.5) |
| `seed1` | `grpo-usw2-1gpu` | `--seed 1` | 78.2 (+0.0) | 3.9 ± 1.4 (**−14.7**) | 1770 ± 485 (+42) | 2281.2 ± 2.5 (+0.7) | 21.6 ± 5.3 (+0.6) |
| `clip0.1` | `grpo-usw2-1gpu` | `--clip 0.1` | 78.2 (+0.0) | 18.1 ± 1.8 (−0.5) | 1787 ± 411 (+59) | 2278.6 ± 1.9 (−1.9) | 23.0 ± 4.0 (+2.0) |
| `groups8` | `grpo-usw2-1gpu` | `--groups 8` | 78.2 (+0.0) | 17.5 ± 3.0 (−1.1) | 2072 ± 84 (+344) | 2281.4 ± 2.7 (+1.0) | 20.4 ± 4.7 (−0.6) |
| `baselinemean` | `grpo-usw2-1gpu` | `--baseline mean` | 78.2 (+0.0) | 18.3 ± 2.9 (−0.3) | 1967 ± 309 (+239) | 2281.8 ± 1.8 (+1.3) | 23.3 ± 5.0 (+2.3) |

Bold = |Δ| larger than 2 × the student std for that scenario. Mean of the last four evals
(iters 225–300), which smooths the 10-episode noise:

| Run | dtc | hg | dc | dtl |
|---|---|---|---|---|
| student | 18.6 | 1728 | 2280.5 | 21.0 |
| `base` | 14.2 | 1685 | 2280.4 | 22.0 |
| `beta0.01` | 10.8 | 1033 | 2134.8 | 23.7 |
| `beta0.2` | 15.5 | 1874 | 2280.0 | 22.2 |
| `g16` | 16.3 | 2019 | 2281.2 | 21.8 |
| `temp1.2` | 12.3 | 1997 | 856.2 | 19.5 |
| `lrconn1e-4` | 21.0 | 1859 | 2280.4 | 22.4 |
| `lr1e-4` | 12.7 | 1655 | 2210.1 | 20.6 |
| `ent0.01` | 15.3 | 1923 | 2280.5 | 23.0 |
| `seed1` | 7.0 | 1912 | 2281.0 | 23.4 |
| `clip0.1` | 15.2 | 1770 | 2134.8 | 23.0 |
| `groups8` | 15.8 | 1970 | 2281.1 | 21.9 |
| `baselinemean` | 19.8 | 1616 | 2281.0 | 22.8 |

## 5. What the training curves say

Per-scenario `metrics.jsonl`, averaged over 60-iteration blocks (`R` = sampled group return,
`kl` = KL(π‖π_ref), `ent` = policy entropy). `base` run:

| Scenario | iters 1–60 | 61–120 | 121–180 | 181–240 | 241–300 |
|---|---|---|---|---|---|
| dtc | R 17.9, kl 0.11, ent 0.42 | 15.8, 0.21, 0.47 | 15.8, 0.24, 0.41 | 16.9, 0.21, 0.40 | 14.6, 0.41, 0.51 |
| hg | R 1320, kl 0.09, ent 0.29 | 1170, 0.19, 0.15 | 1442, 0.23, 0.12 | 1484, 0.21, 0.15 | 1602, 0.21, 0.19 |
| dc | R 2263, kl 0.03, ent 0.29 | 2257, 0.06, 0.24 | 2274, 0.10, 0.25 | 2236, 0.13, 0.21 | 2230, 0.14, 0.22 |
| dtl | R 21.0, kl 0.05, ent 0.89 | 21.8, 0.09, 0.78 | 21.1, 0.14, 0.76 | 21.3, 0.24, 0.71 | 22.3, 0.26, 0.64 |

Same for the runs that mark the corners:

- `seed1` dtc: kl 0.11 → 0.11 → 0.17 → 0.46 → **1.94**, R 18.8 → 19.3 → 18.4 → 15.5 → **5.3**.
  The policy left the reference and the return followed. Nothing in the loss stopped it: at
  β = 0.05 a KL of 1.9 costs 0.1 in loss units, the same order as the clipped policy term.
- `beta0.01` hg: ent 0.44 → 0.32 → 0.17 → 0.10 → **0.05**, R 1775 → 1731 → 1657 → 1430 → **904**.
  Entropy collapse; the policy became near-deterministic and stopped gathering.
- `beta0.2`: kl stays ≤ 0.14 on every scenario for all 300 iterations, entropy is flat, and the
  return is flat too (dtc 18.3 → 16.8, hg 1775 → 1771, dc 2280 → 2259, dtl 20.7 → 21.7).
- `lrconn1e-4` (the only run that ends above the student on dtc): kl grows to 0.61 by the last
  block but R rises with it (18.9 → 14.9 → 17.3 → 19.7 → 20.4). One seed; not yet reproduced.
- `adv_abs` sits at 0.9–1.0 on dtc/hg/dc/dtl in every run (it is std-normalised, so this only
  says the groups are not degenerate) and is exactly 0 on `basic` in every run: all eight
  seed-matched episodes end identically in ~4–6 steps, so `pg = 0` and the basic policy never
  changes. This is why `basic` reads 78.2 ± 7.5 in every row of §4.

## 6. Conclusions so far

1. **GRPO from the distilled student does not beat the student on any scenario in 300
   iterations.** The candidate gains (hg +186 to +372, dtl +2 to +3) are inside one student std
   with n = 10 episodes and are not consistent across knobs. The losses on dtc (−3.8 to −14.7) are
   2–8 std and are consistent: 7 of 12 runs regress it by more than 2 std, 11 of 12 end at or
   below the student.
2. **The KL term is doing the work, and it is too weak at the default.** On every scenario the KL to
   the frozen student keeps growing over the 300 iterations at β = 0.05 (0.03–0.11 in the first
   block, 0.14–1.94 in the last). Where it crosses ~0.4 the return drops.
   β = 0.2 pins the KL and the return; β = 0.01 lets the policy collapse (entropy → 0.05 on hg).
   A β schedule or a KL target (adaptive β, as in PPO-penalty) is the obvious next single change.
3. **The group advantage is not the limiting factor.** `g16`, `groups8`, `baselinemean` and
   `clip0.1` all land within noise of `base`. Doubling the group or the number of groups did not
   change the outcome. This matches PPO's advantage estimate not being what the teachers were
   short on either.
4. **Returns are near the ceiling on three of five scenarios.** dc is deterministic at 2280–2282
   (the corridor is solved), hg's 2100 is the episode timeout, and basic is at zero advantage.
   Only dtc and dtl have room, and dtc is where GRPO hurts. The remaining headroom is small
   enough that 10-episode evals cannot resolve it; the eval budget needs to go up (≥ 50
   episodes) before the next sweep.
5. **PPO's data efficiency is not the comparison to make; wall clock is.** The dc teacher saw
   8 M frames in 69 minutes. A GRPO run sees ~1.7 M frames in 4–7 hours because the 48 M-param
   recurrent net dominates the update. Any GRPO gain has to be worth a 20× slower loop.
6. **The comparison to the teachers is incomplete.** Three of five teachers have no measured
   return (§3). Until they are evaluated greedily, "past the PPO teachers" can only be checked on
   basic and deadly_corridor, and there the student is already at the teacher's level with
   nothing left for GRPO to add.

## 6a. Side-by-side footage: student vs GRPO `base` iter 300

Fresh CPU evals, 5 episodes per scenario, env seed 12345, stochastic policy, both checkpoints on
the same MaleCNS 49k backbone (`--connectome data/processed/connectome_malecns49k.npz`). Left =
`l40s-v2/runs/malecns49k/final` (the GRPO student, iter 0), right = `grpo-use2/runs/malecns49k_grpo_base/final`
(iter 300). Per-episode clips and `eval_all.json` are in `assets/videos/malecns49k_student_iter0/` and
`assets/videos/malecns49k_grpo_base_iter300/`; the hstack clips (median-return episode per side,
labels burned in) come from `scripts/compare_clips.sh` and are section 10 of the tutorial page.

| scenario | student iter 0 | GRPO `base` iter 300 | Δ mean | episodes (student / GRPO) |
|---|---|---|---|---|
| basic | 79.8 ± 4.7 | 79.8 ± 4.7 | +0.0 | identical returns and lengths on all 5 episodes |
| defend_the_center | 18.0 ± 1.4 | 11.2 ± 1.8 | **−6.8** | 233–261 vs 161–207 decisions |
| health_gathering | 1852.0 ± 496.0 | 2100.0 ± 0.0 | +248.0 | student had one 860 episode (died at 240 decisions); GRPO 5/5 at the 2100 timeout |
| deadly_corridor | 2279.3 ± 0.8 | 2281.7 ± 2.3 | +2.4 | both at the corridor cap |
| defend_the_line | 20.8 ± 4.1 | 26.0 ± 4.9 | +5.2 | 141–252 vs 182–323 decisions |

The 5-episode numbers agree with the 10-episode §4 row for `base` in direction on every scenario
(dtc down, hg and dtl up, basic and dc flat). What the footage adds on dtc: by ~20 s the GRPO side
has spent all 26 rounds and stands at low health with an empty pistol while the student still has
ammo; the shorter GRPO episodes (161–207 vs 233–261 decisions) are the same effect. A policy that fires more
would cost on dtc (26 rounds, no pickups) and could pay on dtl; the dtl footage is consistent with
that but 5 episodes cannot confirm it.

**Checkpoint provenance.** `checkpoints/malecns49k_v2_final/model.safetensors` (the footage in
tutorial sections 1–9 and the numbers in `docs/related-work-nftechie-doomfly.md`) is byte-identical to
`s3://doomfly-047472448415-us-west-2/g6-12xl-use2-v2/runs/malecns49k/final/model.safetensors`
(S3 multipart ETag `4143e256…-50` matches), not to `l40s-v2/runs/malecns49k/final`, which is the
student every GRPO run started from. Both are step-60000 distillations with near-identical 10-episode
evals but different weights. That is why the student side above was re-recorded from `l40s-v2`
rather than reused from the existing footage.

## 7. Next single changes (one per run)

- Adaptive β with a KL target of ~0.1 (`--kl-target`), starting from `base`.
- `lrconn1e-4` with seeds 1 and 2, to see if the dtc +2.8 survives.
- Eval with 50 episodes at iters 0 and 300 only, so the eval cost stays constant.
- Greedy 10-episode eval of the five teacher `.zip`s on a ViZDoom box; write the numbers into §3.
- Drop `basic` from the GRPO scenario list (zero signal, §5) and reallocate its iterations.
