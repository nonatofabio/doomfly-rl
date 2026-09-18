"""Stage 1 of the Doom ETL: train a small CNN PPO "teacher" per scenario with
Stable-Baselines3. Analogous to chessfly's Lichess games: the teacher supplies
the (frames -> action, return) supervision that FlyNet is later distilled from.

Usage:  python -m doomfly.doom.teacher --scenario basic --steps 300000 --out data/teachers
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from .actions import SCENARIOS
from .env import make_env


class Log(BaseCallback):
    def __init__(self, path: Path, every: int = 20_000):
        super().__init__()
        self.path, self.every, self.t0, self.rows = path, every, time.time(), []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every < self.training_env.num_envs:
            ep = self.model.ep_info_buffer
            if len(ep):
                r = float(np.mean([e["r"] for e in ep]))
                l = float(np.mean([e["l"] for e in ep]))
                row = {"t": self.num_timesteps, "ep_reward": r, "ep_len": l, "wall": time.time() - self.t0}
                self.rows.append(row)
                print(json.dumps(row), flush=True)
                self.path.write_text(json.dumps(self.rows))
        return True


def train(scenario: str, steps: int, out: Path, n_envs: int = 8, seed: int = 0, device: str = "auto", subproc: bool = True, tb: Path | None = None):
    sc = SCENARIOS[scenario]
    out.mkdir(parents=True, exist_ok=True)
    VecCls = SubprocVecEnv if subproc else DummyVecEnv
    env = VecCls([make_env(scenario, seed=seed * 100 + i) for i in range(n_envs)])
    model = PPO(
        "CnnPolicy", env, n_steps=256, batch_size=512, n_epochs=4, learning_rate=2.5e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=0.01, vf_coef=0.5,
        policy_kwargs=dict(normalize_images=True), seed=seed, device=device, verbose=0,
        tensorboard_log=str(tb) if tb else None,
    )
    model.learn(total_timesteps=steps, callback=Log(out / f"{scenario}_curve.json"), tb_log_name=scenario)
    model.save(out / f"{scenario}.zip")
    env.close()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--tb", type=Path, default=None, help="TensorBoard root dir (optional)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/teachers"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-subproc", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(4)
    train(a.scenario, a.steps or SCENARIOS[a.scenario].ppo_steps, a.out, a.n_envs, device=a.device, subproc=not a.no_subproc, tb=a.tb)
