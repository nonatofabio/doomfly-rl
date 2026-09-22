---
license: mit
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - vizdoom
  - doom
  - connectome
  - drosophila
  - distillation
datasets:
  - fernandofernandes/fly-connectome-49k
---

<p align="center"><img src="doomfly_icon.png" width="160" alt="doomfly-rl icon"></p>

# doomfly-rl: a network wired like a fruit fly's brain, playing Doom

Three checkpoints from [nonatofabio/doomfly-rl](https://github.com/nonatofabio/doomfly-rl), all
step 60,000 of the same distillation recipe on the five standard ViZDoom scenarios, plus the
connectome file they need. The project was built end to end by the
[Strands harness](https://github.com/strands-agents/harness-sdk); the story is
[here](https://nonatofabio.github.io/blog/posts/doomfly_autonomous.html).

**Read a score as "connectome-constrained RL agent", never as "a fly".** Two thirds of the
parameters are an ordinary conv stem and MLP readout, and the controls below show the wiring
adds nothing measurable on these scenarios.

<video src="doomfly_reel.mp4" controls autoplay muted loop playsinline width="100%"></video>

*The `malecns49k_v2_final` checkpoint on the five ViZDoom scenarios, one clip each, median episode of ten.*

## Files

| Path | What it is |
|---|---|
| `malecns49k_v2_final/` | The real connectome: 49,393 MaleCNS neurons, 9,050,172 edges, frozen wiring and signs, one learned gain per synapse. 48.2 M params. The footage in the blog post. |
| `malecns49k_shuffled_s0/` | Control. Same graph with edges shuffled degree-preserving (seed 0): same sparsity, same fan-in/out, same signs, no biology. 48.2 M params. |
| `malecns49k_noconn/` | Control. No neuron layer at all, stem wired straight to the decoder. 27.7 M params. |
| `connectome_malecns49k.npz` | The connectome the first two checkpoints run on. Derived from [fernandofernandes/fly-connectome-49k](https://huggingface.co/datasets/fernandofernandes/fly-connectome-49k), CC BY 4.0. `connectome_malecns49k.json` has the source revision and manifest hash. |

Each checkpoint directory holds `model.safetensors` and `config.json` (`FlyNetConfig` plus the
training step).

## Results (10 episodes, mean ± std of return, sampled policy)

| checkpoint | basic | defend_the_center | health_gathering | deadly_corridor | defend_the_line |
|---|---|---|---|---|---|
| `malecns49k_v2_final` (connectome) | 78.2 ± 7.5 | 18.6 ± 1.8 | 1728 ± 616 | 2280.5 ± 2.4 | 21.0 ± 5.9 |
| `malecns49k_shuffled_s0` | 78.2 ± 7.5 | 19.0 ± 1.7 | 1602 ± 597 | 2280.6 ± 2.2 | 21.7 ± 6.1 |
| `malecns49k_noconn` | 77.5 ± 8.5 | 18.3 ± 1.6 | 1373 ± 555 | 2279.9 ± 2.4 | 23.0 ± 5.0 |

Every gap is inside one eval standard deviation. Full protocol and training curves:
`docs/controls.md` in the repo.

## Use

```bash
git clone https://github.com/nonatofabio/doomfly-rl && cd doomfly-rl
uv venv && uv pip install -e . huggingface_hub
hf download fabiononato/doomfly-rl --local-dir hf

# play 10 episodes of every scenario and write the median one per scenario as MP4
python -m doomfly.evaluate --ckpt hf/malecns49k_v2_final --connectome hf/connectome_malecns49k.npz \
    --episodes 10 --gif-dir /tmp/clips --tics --pick median --fmt mp4 --device cpu

# the no-connectome control needs no connectome file but the loader still takes the path
python -m doomfly.evaluate --ckpt hf/malecns49k_noconn --connectome hf/connectome_malecns49k.npz --episodes 10
```

In Python:

```python
from doomfly.surgery import load_flynet
model, cfg = load_flynet("hf/malecns49k_v2_final", "hf/connectome_malecns49k.npz", "cpu")
model.eval()
```

`load_flynet` widens the stored 22-action, 5-scenario head to 23/6 in memory so the same
checkpoint runs on the five scenarios and in Freedoom II free play. On the five scenarios the
output is bit-identical to the original head.

## Model

<p align="center"><img src="connectome_rotate.gif" width="360" alt="The MaleCNS-49k connectome, 49,393 neurons, rotating"></p>

`FlyNet` follows [mlabonne/chessfly](https://huggingface.co/mlabonne/chessfly). Input: a 4-frame
stack of 72×96 grayscale Doom frames through a conv stem onto the visual sensory neurons. Core: the
connectome as a sparse recurrent layer unrolled 5 steps, with BatchNorm and homeostatic
scale/shift, sign and topology frozen, one log-gain per edge learned. Readout: central,
descending and motor neurons → 512 → 22 actions with a per-scenario legality mask, plus a 64-bin
value head.

Training: PPO teachers (Stable-Baselines3, one per scenario) → ε-greedy rollouts → distillation
for 60k steps, batch 128, lr 3e-4 (stem/readout) and 3e-3 (connectome gains), on one L40S. Details
and every launch script are in the repo.

## Limitations

- Five saturated ViZDoom scenarios only. The controls show no connectome effect there; whether
  a wiring prior matters when the policy has to learn rather than imitate is the open question
  (free play on Freedoom II MAP01 is next in the repo).
- No FlyWire-783 checkpoint here yet; the 138k-neuron backbone trains 22× slower and its final
  run is in S3, not yet published.
- Nothing here is a claim about biology.
