"""Held-out evaluation of a free-play policy on Freedoom II MAP01.

The in-run eval of doomfly.grpo_freeplay reuses the 10 zero-shot seeds (default_rng(0)), and
those are also the first training seeds of every run (same default_rng(0) stream, --seed 0).
This script scores a checkpoint on seeds drawn from a different stream, with the same shaped
reward and the same protocol (1024 decisions, sampled at temperature 1.0), and writes every
episode so that runs can be compared pairwise, seed by seed.

--init applies the iter-0 policy of doomfly.grpo_freeplay (FREEPLAY row from deadly_corridor,
SELECT_NEXT_WEAPON bias -4) to a scenario-distilled student; use it to score the start point of
a run. A GRPO output already carries its trained rows, so score it without --init.

    python -m doomfly.freeplay_eval --ckpt checkpoints/malecns49k_grpo_freeplay_final \\
        --connectome data/processed/connectome_malecns49k.npz --out docs/results/x.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .doom.actions import FREEPLAY
from .doom.env import DoomEnv
from .grpo import load_student, sample_groups
from .grpo_freeplay import W, ShapedReward, apply_init, summarise
from .train import legal_mask_table


def heldout_seeds(n, seed):
    """n engine seeds from default_rng(seed). seed 0 is the in-run eval and training stream."""
    if seed == 0:
        raise SystemExit("seed 0 is the in-run eval/training stream; pick another")
    return [int(s) for s in np.random.default_rng(seed).integers(0, 2**31 - 1, size=n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--init", action="store_true", help="apply the grpo_freeplay iter-0 init first")
    ap.add_argument("--emb-from", default="deadly_corridor")
    ap.add_argument("--nextw-bias", type=float, default=-4.0)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1, help="seed stream for the episodes (not 0)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    seeds = heldout_seeds(a.episodes, a.seed)
    model, cfg = load_student(a.ckpt, a.connectome, dev)
    if a.init:
        apply_init(model, a.emb_from, a.nextw_bias)
    model.eval()
    legal = legal_mask_table(dev)[FREEPLAY.id]
    envs = [DoomEnv(FREEPLAY) for _ in seeds]
    t0 = time.time()
    trajs = sample_groups(model, envs, seeds, FREEPLAY.id, legal, dev, a.max_steps, a.temperature,
                          [ShapedReward() for _ in envs])
    for e in envs:
        e.close()
    rec = {
        "ckpt": str(a.ckpt), "config_iter": json.loads((a.ckpt / "config.json").read_text()).get("iter"),
        "init": {"emb_from": a.emb_from, "nextw_bias": a.nextw_bias} if a.init else None,
        "map": FREEPLAY.doom_map, "skill": FREEPLAY.doom_skill, "max_steps": a.max_steps,
        "temperature": a.temperature, "seed_stream": a.seed, "seeds": seeds, "reward_weights": W,
        "summary": summarise(trajs),
        "episodes": [{"seed": s, "R": t["R"], "T": t["T"], "dead": t["dead"], **t["comps"]}
                     for s, t in zip(seeds, trajs)],
        "wall_s": time.time() - t0,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({"ckpt": rec["ckpt"], "init": rec["init"], "wall_s": round(rec["wall_s"], 1),
                      **{k: round(v["mean"], 3) for k, v in rec["summary"].items()}}))


if __name__ == "__main__":
    main()
