# doomfly-rl vs. nftechie/doomfly — same name, opposite question

Two repositories are called `doomfly`. They are not forks of each other. This note records
what each one is, where they differ, and what is worth borrowing. Snapshot taken 2026‑09‑18.

- **This repo** — `nonatofabio/doomfly`, to be renamed `doomfly-rl` (private, `fnp`). A neural network wired like a
  fruit‑fly brain, trained to play Doom, following the
  [mlabonne/chessfly](https://huggingface.co/mlabonne/chessfly) recipe.
- **The other one** — `nftechie/doomfly` (public, 361 stars / 56 forks at snapshot time). A
  simulation of the whole fly connectome as a spiking brain‑computer interface that drives Doom,
  with a live synaptic‑plasticity experiment.

## Side by side

| | `nonatofabio/doomfly` (this repo) | `nftechie/doomfly` |
|---|---|---|
| **One‑liner** | Network constrained to the fly wiring diagram, trained with gradient descent to play Doom | Whole connectome run as a spiking network with fixed biological I/O; a game is the assay |
| **Connectome** | FlyWire FAFB v783 (Shiu et al.): 138,639 neurons / 15,091,983 edges / 54.5 M synapses. Ablation backbone MaleCNS‑49k (49,393 neurons) | MaleCNS v1.0, full retained graph: 166,700 neurons / 25,582,938 edges / 124 M contacts. No cropping or pruning (`AGENTS.md` forbids it) |
| **Neuron model** | Rate units, 5 unrolled steps: `h ← (1‑a)h + a·relu(γ·norm(Wh+u)+β)`; BatchNorm plus per‑neuron/per‑step homeostatic scale/shift | LIF spiking, dt = 0.1 ms, τ_m = 20 ms, τ_g = 5 ms, rest −52 mV, threshold −45 mV, refractory 2.2 ms, delay 1.8 ms; event‑driven native C++ kernel |
| **Vision → brain** | Learned conv stem (3 conv + 2 linear) on 4×72×96 grayscale stacks → currents into 10,855 visual sensory neurons | Pixels mapped to 3,335 R1–R6 photoreceptors + 811 R8 via inferred hex columns; low‑pass + saturating photocurrent. Nothing learned |
| **Brain → buttons** | Readout of 33,788 central/descending/motor neurons → learned `Linear(→512)` + MLP policy over 22 actions with per‑scenario legality mask, plus a 64‑bin value head | Hand‑picked decoder: DNp20 (R−L rate) → turn; summed DNpe017 rate → move; DNpe017 spike → fire. Fixed, not learned |
| **What learns** | `W_ij = sign_ij · log1p(syn) · exp(θ_ij)` — one gain per synapse (15.09 M) + 1.39 M homeostatic + 34.5 M stem/decoder/heads = **51.0 M params** | **4,184 KC→MBON11 synapses only**, anti‑Hebbian rule adapted from Huang et al. 2024 (η = 0.001, bounds 0.1–2× original, 1,800 s decay). Everything else frozen |
| **How it learns** | Offline: PPO teachers (SB3) → ε‑greedy rollouts → distillation (KL + value CE) → optional GRPO/RLOO fine‑tune with KL‑to‑reference | Online, in the live stream: non‑fatal damage → +4 mV into two PPL101 dopamine cells for 200 ms; weights update during play, state persists across rounds |
| **Environment** | 5 standard ViZDoom scenarios (`basic`, `deadly_corridor`, `defend_the_center`, `defend_the_line`, `health_gathering`), frame‑skip 4 | Custom `combat_survival` arena (spawning enemies, ammo pickups, death‑only termination), plus legacy `defend_the_center` |
| **Results posture** | Ships checkpoints + eval videos. `malecns49k_v2_final`, mean return over 10 episodes: basic 78.2, defend_the_center 18.1, health_gathering 1817.6, deadly_corridor 2280.1, defend_the_line 22.7 | Ships negative results: v6 candidate failed visual‑recovery, conditioning, and survival gates; labelled UNVALIDATED; "not demonstrated learned survival"; sim runs at 0.158× wall time |
| **Code size** | ≈2.6 k lines Python/shell, single `doomfly` package | 623+ files: `doom/` sim + C++ kernel, `doom_learning…_v6`, Next.js spectator UI (`doom-ui`), Docker `deploy/`, `tests/`, 13 docs, evidence + provenance manifests |
| **Infra** | AWS CDK stack (S3, launch template, ASG, us‑west‑2), TensorBoard via SSM tunnel, static tutorial site behind Midway | Local‑only server on `127.0.0.1:8766` with `/state` and `/health`, broadcaster audit log, container deploy |
| **Dependencies** | Python ≥3.10 <3.13, torch, SB3, gymnasium, vizdoom, safetensors, hatchling | Python 3.11 pinned, Brian2 2.5.1 / numpy 1.24.4 env for the Shiu reference model, C++ compiler, Node ≥22 |
| **Licensing / provenance** | Data: Shiu et al. (MIT), Schlegel annotations, HF `fly-connectome-49k` (CC BY 4.0); SHA‑256 checks against chessfly's `connectome_meta.json` | MIT code, separate `THIRD_PARTY*.md`, `licenses/`, `data-provenance/…/source.lock.json`, explicit software disclaimer |

## The deeper difference

Both take "fly brain plays Doom" literally, but they answer opposite questions.

- **This repo asks:** if you constrain a network to the fly's wiring diagram and synapse signs,
  can gradient descent make it play? It is machine learning. The connectome is an inductive
  bias; the stem, readout, and 15 M synaptic gains are free parameters, and a conv net plus MLP
  heads do real work at both ends. The honest framing, which the README already uses, is
  "wired like a fruit‑fly brain", not "a fruit fly".
- **Theirs asks:** if you run the actual connectome as a spiking network with fixed,
  biology‑motivated I/O, does anything useful happen — and can a plausible dopamine rule on one
  real circuit (KC→MBON) learn? It is computational neuroscience with a game as the assay. They
  deliberately refuse the levers this repo pulls (no learned encoder, no learned decoder, no
  replacement policy), and the published answer so far is "no learning demonstrated".

## Things worth noting

- **Name collision, not a fork.** If this repo goes public, expect readers to conflate the two.
  One line in the README pointing at this note saves the same question being asked repeatedly.
- **Their contract reads as adversarial to ours.** Their `AGENTS.md` bans exactly what this
  approach does: "Do not crop circuits, prune weak/self edges, or replace the network with a
  game policy." That is not a criticism of this repo — it is a different contract — but it is
  how a neuroscience audience will read a 51 M‑parameter model. Say up front what is learned and
  what is inherited from the connectome.
- **Reusable ideas from theirs:**
  - The SHA‑256 `source.lock.json` provenance pattern for every data input.
  - The "what is measured / inferred / chosen / unresolved" split in the docs. Cheap to add to
    the tutorial site and it pre‑empts the "how much of this is the fly?" question.
  - The spectator‑stream UX is a lot of code this repo does not need.
- **Workspace state at snapshot time (2026‑09‑18):** 13 commits, 5 unpushed; uncommitted edits
  to `evaluate.py` and `pyproject.toml`; `assets/` untracked with ~1.1 GB of videos. Resolved
  since: footage is gitignored (`assets/videos/`, `tutorial/assets/videos/*.mp4`) and backed up
  to `s3://doomfly-047472448415-us-west-2/site-assets/`; regenerate with `evaluate.py --tics`
  and `scripts/make_clips.sh` (see `tutorial/assets/videos/README.md`).

## In the README

The README (`# doomfly-rl`) carries a short disambiguation box at the top and a
"Relation to nftechie/doomfly" section that explains why their design is coherent for their question
and why ours differs. Keep the two in sync with this note.
