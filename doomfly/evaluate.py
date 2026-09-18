"""Live evaluation: let FlyNet play ViZDoom scenarios, report per-scenario mean
episode return, and save a GIF of the best episode per scenario.

Usage (standalone):
  python -m doomfly.evaluate --ckpt runs/flywire783/final --connectome data/processed/connectome_783.npz \
      --episodes 10 --gif-dir runs/flywire783/gifs/final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .doom.actions import SCENARIOS, N_ACTIONS
from .doom.env import DoomEnv


@torch.no_grad()
def play_episode(model, env: DoomEnv, sid: int, legal_row: torch.Tensor, device, greedy=False, record=False, max_steps=2100):
    obs, _ = env.reset()
    frames, R, done, n = [], 0.0, False, 0
    local = np.asarray(env.sc.actions)
    g2l = {int(g): i for i, g in enumerate(local)}
    while not done and n < max_steps:
        if record:
            frames.append(env.render())
        fr = torch.from_numpy(obs[None]).to(device)
        sc = torch.tensor([sid], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(fr, sc, legal_mask=legal_row[None])["policy"].float()[0]
        a = int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, done, _, _ = env.step(g2l[a])
        R += r; n += 1
    return R, n, frames


def evaluate(model, legal, episodes=10, gif_dir: Path | None = None, device=torch.device("cpu"), scenarios=None, greedy=False):
    was_training = model.training
    model.eval()
    out = {}
    for name in scenarios or list(SCENARIOS):
        sc = SCENARIOS[name]
        env = DoomEnv(name, render=True, seed=12345)
        rets, lens, best = [], [], (-np.inf, None)
        for ep in range(episodes):
            R, n, frames = play_episode(model, env, sc.id, legal[sc.id], device, greedy=greedy, record=gif_dir is not None)
            rets.append(R); lens.append(n)
            if R > best[0]:
                best = (R, frames)
        env.close()
        out[name] = {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
                     "max_return": float(np.max(rets)), "mean_len": float(np.mean(lens))}
        if gif_dir is not None and best[1]:
            gif_dir.mkdir(parents=True, exist_ok=True)
            # 35 game tics/s with frame_skip 4 -> ~8.75 decisions/s; play back at that rate
            imageio.mimsave(gif_dir / f"{name}.gif", best[1][::1], duration=1 / 8.75, loop=0)
    if was_training:
        model.train()
    return out


if __name__ == "__main__":
    from safetensors.torch import load_file
    from .model.flynet import FlyNet, FlyNetConfig, load_connectome
    from .train import legal_mask_table

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--gif-dir", type=Path, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cfg = FlyNetConfig(**json.loads((a.ckpt / "config.json").read_text())["flynet"])
    model = FlyNet(load_connectome(a.connectome), cfg).to(dev)
    model.load_state_dict(load_file(a.ckpt / "model.safetensors"))
    print(json.dumps(evaluate(model, legal_mask_table(dev), a.episodes, a.gif_dir, dev, greedy=a.greedy), indent=1))
