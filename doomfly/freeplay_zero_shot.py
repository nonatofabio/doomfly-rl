"""Zero-shot baseline for free play on Freedoom II MAP01.

The distilled student has never seen a full level. Before any RL on real maps we need to know
where it starts: does it move, does it kill anything, does it die, does it ever reach the exit?
Because there is no free-play scenario embedding yet, the student borrows the id and legal action
set of a trained scenario (`--prior`); we compare against a uniform-random policy over the same
legal set. Every number here comes from the given checkpoint; results go to docs/.

  python -m doomfly.freeplay_zero_shot --ckpt checkpoints/malecns49k_v2_final \\
      --connectome data/processed/connectome_malecns49k.npz --episodes 10 --max-steps 1024

Per episode we report: env return (map_exit_reward=1 -> reached exit), kills, items, secrets,
damage dealt, final health, dead, decisions taken, straight-line displacement from spawn and path
length (map units), all from ViZDoom game variables.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .doom.actions import SCENARIOS, FREEPLAY, N_ACTIONS
from .doom.env import DoomEnv


@torch.no_grad()
def play(model, env: DoomEnv, sid: int | None, legal: np.ndarray, device, max_steps: int, rng: np.random.Generator,
         temperature: float = 1.0):
    """sid=None -> uniform random over `legal` (global indices). Returns per-episode stats."""
    obs, _ = env.reset()
    g2l = {int(g): i for i, g in enumerate(env.sc.actions)}
    legal_row = torch.zeros(N_ACTIONS, dtype=torch.bool, device=device)
    legal_row[list(legal)] = True
    R, n, done = 0.0, 0, False
    path, x0 = 0.0, None
    prev = None
    info = {"vars": {}}
    while not done and n < max_steps:
        if sid is None:
            a = int(rng.choice(legal))
        else:
            fr = torch.from_numpy(obs[None]).to(device)
            sc = torch.tensor([sid], device=device)
            logits = model(fr, sc, legal_mask=legal_row[None])["policy"].float()[0] / temperature
            a = int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, done, _, info = env.step(g2l[a])
        R += r; n += 1
        v = info["vars"]
        p = (v["POSITION_X"], v["POSITION_Y"])
        if x0 is None:
            x0 = p
        if prev is not None:
            path += math.dist(prev, p)
        prev = p
    v = info["vars"]
    return {
        "return": R, "exit": bool(R >= 1.0), "kills": v.get("KILLCOUNT", 0.0), "items": v.get("ITEMCOUNT", 0.0),
        "secrets": v.get("SECRETCOUNT", 0.0), "damage": v.get("DAMAGECOUNT", 0.0), "health": v.get("HEALTH", 0.0),
        "dead": bool(info.get("dead", False)), "decisions": n,
        "displacement": math.dist(x0, prev) if x0 and prev else 0.0, "path": path,
    }


def summarise(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        xs = np.array([float(r[k]) for r in rows])
        out[k] = {"mean": float(xs.mean()), "std": float(xs.std())}
    return out


def main():
    from safetensors.torch import load_file
    from .model.flynet import FlyNet, FlyNetConfig, load_connectome

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1024, help="decisions per episode (frame skip 4 -> ~2 min game time)")
    ap.add_argument("--priors", nargs="*", default=["deadly_corridor", "health_gathering"],
                    help="trained scenarios whose id + legal set the student borrows")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    a = ap.parse_args()

    dev = torch.device(a.device)
    cfg = FlyNetConfig(**json.loads((a.ckpt / "config.json").read_text())["flynet"])
    model = FlyNet(load_connectome(a.connectome), cfg).to(dev).eval()
    model.load_state_dict(load_file(a.ckpt / "model.safetensors"))
    rng = np.random.default_rng(a.seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=a.episodes)]

    conditions = [("random_dc_legal", None, np.array(SCENARIOS["deadly_corridor"].actions))]
    conditions += [(f"pi_ref_as_{p}", SCENARIOS[p].id, np.array(SCENARIOS[p].actions)) for p in a.priors]

    results = {"ckpt": str(a.ckpt), "step": json.loads((a.ckpt / "config.json").read_text()).get("step"),
               "map": FREEPLAY.doom_map, "skill": FREEPLAY.doom_skill, "max_steps": a.max_steps,
               "episodes": a.episodes, "seeds": seeds, "temperature": a.temperature, "conditions": {}}
    for name, sid, legal in conditions:
        rows, t0 = [], time.time()
        for s in seeds:
            env = DoomEnv(FREEPLAY, seed=s)
            rows.append(play(model, env, sid, legal, dev, a.max_steps, rng, a.temperature))
            env.close()
            print(f"[{name}] seed={s} {json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in rows[-1].items()})}", flush=True)
        results["conditions"][name] = {"episodes": rows, "summary": summarise(rows), "wall_s": time.time() - t0}
        print(f"[{name}] summary: " + " ".join(f"{k}={v['mean']:.1f}±{v['std']:.1f}" for k, v in results["conditions"][name]["summary"].items()), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(results, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
