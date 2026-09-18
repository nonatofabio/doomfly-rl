"""Stage 2 of the Doom ETL: roll out the PPO teacher (with epsilon exploration so
the dataset covers off-policy states) and record supervised shards:

  frames   uint8 [T, 4, 72, 96]   stacked grayscale observations
  scenario int8  [T]              scenario id
  action   int8  [T]              GLOBAL action index actually taken
  teacher  f16   [T, N_ACTIONS]   teacher policy over the global vocabulary (0 on illegal)
  ret      f32   [T]              discounted return-to-go (gamma=0.99) from this state
  episode  int32 [T]              episode counter (for splitting train/val by episode)

Usage:  python -m doomfly.doom.record --scenario basic --teacher data/teachers/basic.zip \
            --frames 500000 --out data/rollouts/basic --eps 0.1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .actions import SCENARIOS, N_ACTIONS
from .env import DoomEnv

GAMMA = 0.99


def teacher_probs(model: PPO, obs: np.ndarray) -> np.ndarray:
    """obs [B,4,72,96] uint8 -> probs [B, n_local] float32."""
    with torch.no_grad():
        t = torch.as_tensor(obs, device=model.device)
        dist = model.policy.get_distribution(t)
        return dist.distribution.probs.float().cpu().numpy()


def record(scenario: str, teacher_path: Path, n_frames: int, out: Path, eps: float = 0.1,
           shard: int = 20_000, seed: int = 0, device: str = "auto"):
    sc = SCENARIOS[scenario]
    model = PPO.load(teacher_path, device=device)
    env = DoomEnv(scenario, seed=seed)
    local = np.asarray(sc.actions)
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    buf = {k: [] for k in ("frames", "action", "teacher", "ret", "episode")}
    n_total, ep_id, shard_id, t0 = 0, 0, 0, time.time()
    ep_returns, ep_lens = [], []

    def flush():
        nonlocal shard_id, buf
        if not buf["frames"]:
            return
        np.savez(out / f"shard_{shard_id:04d}.npz",
                 frames=np.stack(buf["frames"]).astype(np.uint8),
                 scenario=np.full(len(buf["frames"]), sc.id, np.int8),
                 action=np.array(buf["action"], np.int8),
                 teacher=np.stack(buf["teacher"]).astype(np.float16),
                 ret=np.array(buf["ret"], np.float32),
                 episode=np.array(buf["episode"], np.int32))
        shard_id += 1
        buf = {k: [] for k in buf}

    while n_total < n_frames:
        obs, _ = env.reset()
        frames, acts, probs, rews = [], [], [], []
        done = False
        while not done:
            p_local = teacher_probs(model, obs[None])[0]
            p_global = np.zeros(N_ACTIONS, np.float32)
            p_global[local] = p_local
            if rng.random() < eps:
                a_local = int(rng.integers(len(local)))
            else:
                a_local = int(rng.choice(len(local), p=p_local / p_local.sum()))
            frames.append(obs); acts.append(int(local[a_local])); probs.append(p_global)
            obs, r, done, _, _ = env.step(a_local)
            rews.append(r)
        # discounted return-to-go
        ret = np.zeros(len(rews), np.float32)
        g = 0.0
        for i in range(len(rews) - 1, -1, -1):
            g = rews[i] + GAMMA * g
            ret[i] = g
        buf["frames"] += frames; buf["action"] += acts; buf["teacher"] += probs
        buf["ret"] += ret.tolist(); buf["episode"] += [ep_id] * len(frames)
        ep_returns.append(float(sum(rews))); ep_lens.append(len(rews))
        n_total += len(frames); ep_id += 1
        if len(buf["frames"]) >= shard:
            flush()
            print(json.dumps({"frames": n_total, "episodes": ep_id, "mean_ep_return": float(np.mean(ep_returns[-50:])),
                              "fps": n_total / (time.time() - t0)}), flush=True)
    flush()
    env.close()
    all_ret = np.concatenate([np.load(f)["ret"] for f in sorted(out.glob("shard_*.npz"))])
    meta = {"scenario": scenario, "frames": int(n_total), "episodes": ep_id, "eps": eps, "gamma": GAMMA,
            "ret_lo": float(np.percentile(all_ret, 0.5)), "ret_hi": float(np.percentile(all_ret, 99.5)),
            "ep_return_mean": float(np.mean(ep_returns)), "ep_return_std": float(np.std(ep_returns)),
            "ep_len_mean": float(np.mean(ep_lens)), "shards": shard_id, "wall_s": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=500_000)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    record(a.scenario, a.teacher, a.frames, a.out, a.eps, seed=a.seed, device=a.device)
