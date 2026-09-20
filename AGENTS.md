# AGENTS.md — doomfly-rl

Read this before you change anything. It says what the project is for, what it is not, and the
rules that keep results honest.

## What this project is

`doomfly-rl` exists for two reasons. Keep both in view; a change that serves one and breaks the
other is not an improvement.

1. **A capability test for our agent harness.** This repo is built and operated end to end by the
   agentic coding harness whose session state lives in `.agent/` (gitignored). The harness has to
   handle the whole loop without a human at the keyboard: connectome ETL, model code, PPO teachers,
   distillation, GPU fleet in AWS (CDK, ASGs, SSM, S3 mirrors), TensorBoard, evaluation, footage,
   the tutorial site, and the write-ups. Every gap you hit — a missing script, a step that needs a
   human, a result that cannot be reproduced from S3 — is a finding about the harness. Record it
   (see *Findings* below) rather than working around it silently.

2. **A study of GRPO for connectome-constrained models.** The scientific question is narrow: once
   a recurrent network constrained to the fruit-fly wiring diagram has been distilled from PPO
   teachers, does critic-free group-relative policy optimisation (GRPO, and its RLOO baseline) let
   it improve *past* the teachers, and how does that depend on the backbone (FlyWire v783 vs.
   MaleCNS-49k), the group size, the KL coefficient to the frozen student, and which parameters are
   unfrozen (synaptic gains only vs. gains + stem + readout)? `doomfly/grpo.py` is the instrument;
   the distilled checkpoints are the starting points. The rationale for GRPO here (the 64-bin value
   head is the weakest part of the model, and PPO's GAE leans on it) is in that file's docstring.

The Doom scores themselves are the measurement, not the goal. Do not tune for a leaderboard.

## What this project is not

- **Not `nftechie/doomfly`.** That project runs the full MaleCNS connectome as a spiking network
  with fixed biological I/O and studies online plasticity in one circuit. It forbids learned
  encoders, learned decoders, pruning, and "replacing the network with a game policy". We do all of
  those on purpose; see the README section *Relation to nftechie/doomfly* and
  `docs/related-work-nftechie-doomfly.md`. Do not import their framing, their claims, or their
  code. Do not describe our results in their vocabulary.
- **Not a claim about biology.** The connectome is a structural prior. 34.5 M of the 51.0 M
  parameters are an ordinary conv stem, readout, policy MLP, and value head. Write "a network
  constrained to the fly connectome", never "a fly" or "the fly brain plays Doom".
- **Not a product.** No spectator UI, no deployment story beyond the training fleet and the static
  tutorial page.

## Rules

Reproducibility

- Every number that appears in the README, the tutorial, or `docs/` must come from a checkpoint
  that exists in `checkpoints/` or under `s3://doomfly-047472448415-us-west-2/<prefix>/runs/`, with
  the exact `evaluate.py` invocation (episodes, `--greedy` or not, seed) written next to it.
- Two boxes never share an S3 prefix. New run → new `DOOMFLY_PREFIX` and a `Prefix=` tag.
- `final/` under a run prefix is the contract that the run finished. Do not terminate a box whose
  run has no `final/`.
- Connectome builds record the SHA-256 of their raw inputs in the connectome's meta file (see
  `build_malecns49k.py`). Any new data source gets the same treatment before it is used.

GRPO study protocol

- The reference policy `pi_ref` is always the distilled student the run started from. Never
  fine-tune a GRPO output with itself as reference without saying so in the run's `meta.json`.
- Change one thing per run. Name runs by what changed: `<backbone>_grpo_<what>` (for example
  `malecns49k_grpo_g8_beta0.05`, `flywire783_rloo_gainsonly`).
- Report GRPO results as the delta over the student's own eval, same scenarios, same episode count,
  same seed, with `std_return`. A single scenario going up is not a result.
- Negative results are results. If GRPO does not beat distillation, that goes in `docs/` with the
  run prefix, not in the bin.

Repository hygiene

- The Python package stays `doomfly`. The distribution and the project name are `doomfly-rl`.
- Footage (`assets/videos/`, `tutorial/assets/videos/*.mp4`) is never committed. Regenerate with
  `evaluate.py --tics --pick median --fmt mp4` and `scripts/make_clips.sh`; back up to
  `s3://…/site-assets/`.
- Checkpoints (`*.safetensors`, `*.pt`) are never committed. S3 is the store.
- Keep README, `docs/related-work-nftechie-doomfly.md`, and the tutorial in sync when the model,
  the parameter counts, or the results change. `tutorial/generate.py` produces `index.html`; edit
  `template.html`, not the output.
- Prefer one script that does the job over a framework. The codebase is ~3 k lines; keep it there.

Working with other sessions

- More than one harness session may hold this working tree. Before committing, run `git status`
  and `git diff` and stage only files you changed. Do not commit hunks you do not recognise.
- Do not push, rename the GitHub repo, terminate instances, or delete S3 prefixes without an
  explicit instruction in the current session.

## Findings

Harness findings and GRPO results are written down, not remembered.

- Harness: `docs/harness-findings.md` — one bullet per gap or surprise, with the session id from
  `.agent/sessions/` and what was done about it.
- GRPO: `docs/grpo-study.md` — the run table (prefix, backbone, what changed, student eval, GRPO
  eval, delta) and the conclusions so far.

Both files may be empty or missing early on. Create them the first time you have something to say.

## Where things are

| | |
|---|---|
| Model | `doomfly/model/flynet.py`, `doomfly/model/sparse.py` |
| Distillation | `doomfly/train.py` |
| GRPO / RLOO | `doomfly/grpo.py` |
| Evaluation + footage | `doomfly/evaluate.py`, `scripts/make_clips.sh` |
| Fleet | `infra/` (CDK), `scripts/{ship,launch,launch_region,boxes,status,tb}.sh`, `scripts/run_all.sh` |
| Site | `tutorial/`, `scripts/publish_site.sh {sync,start,status,stop}` |
| Data | `data/processed/connectome_*.npz` (gitignored), `s3://…/connectomes/` |
| Runs so far | `l40s-v2/{flywire783,malecns49k}`, `l40s-v2-b256/…`, `g6-12xl-use2/malecns49k`; local `checkpoints/malecns49k_v2_final` (== `g6-12xl-use2-v2/runs/malecns49k/final`, not the GRPO student `l40s-v2/runs/malecns49k/final`; see `docs/grpo-study.md` §6a) |
