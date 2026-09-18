"""Stage 1 of the Doom ETL: train a small CNN PPO "teacher" per scenario with
Stable-Baselines3. Analogous to chessfly's Lichess games: the teacher supplies
the (frames -> action, return) supervision that FlyNet is later distilled from.

Usage:  python -m doomfly.doom.teacher --scenario basic --steps 300000 --out data/teachers
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
import vizdoom as vzd

from .actions import SCENARIOS
from .env import make_env


class TeacherReward(gym.Wrapper):
    """Teacher-only reward transform: optional health/kill/ammo shaping (raw units), then reward_scale.
    The env itself keeps raw rewards so recorder returns and in-env eval stay comparable across runs.
    Raw episode return is reported in info["raw_r"] at episode end."""

    def __init__(self, env, scale: float = 1.0, shaping: bool = False):
        super().__init__(env)
        self.scale, self.shaping = scale, shaping
        self.game = env.unwrapped.game
        self.raw = 0.0

    def _vars(self):
        g = self.game
        return (g.get_game_variable(vzd.GameVariable.HEALTH), g.get_game_variable(vzd.GameVariable.KILLCOUNT),
                g.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.raw = 0.0
        self.prev = self._vars()
        return obs, info

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        self.raw += r
        if self.shaping and not term:
            h, k, am = self._vars()
            ph, pk, pam = self.prev
            r += 10.0 * min(h - ph, 0) + 200.0 * max(k - pk, 0) + 5.0 * min(am - pam, 0)
            self.prev = (h, k, am)
        if term or trunc:
            info = {**info, "raw_r": self.raw}
        return obs, r * self.scale, term, trunc, info


def make_teacher_env(scenario: str, seed: int):
    sc = SCENARIOS[scenario]
    def _f():
        # Monitor must be here: SB3 does not auto-wrap a VecEnv, and without it ep_info_buffer stays
        # empty (that is why the first teacher run produced no curves and 0-byte logs).
        return Monitor(TeacherReward(make_env(scenario, seed=seed)(), scale=sc.reward_scale, shaping=sc.shaping))
    return _f


class Log(BaseCallback):
    def __init__(self, path: Path, every: int = 20_000):
        super().__init__()
        self.path, self.every, self.t0, self.rows = path, every, time.time(), []
        self.raw = deque(maxlen=100)

    def _on_training_end(self) -> None:
        self._log(force=True)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            if "raw_r" in info:
                self.raw.append(info["raw_r"])
        self._log()
        return True

    def _log(self, force: bool = False):
        if force or self.num_timesteps % self.every < self.training_env.num_envs:
            ep = self.model.ep_info_buffer
            if len(ep):
                r = float(np.mean([e["r"] for e in ep]))
                l = float(np.mean([e["l"] for e in ep]))
                row = {"t": self.num_timesteps, "ep_reward": r, "ep_len": l, "wall": time.time() - self.t0,
                       "raw_return": float(np.mean(self.raw)) if self.raw else None}
                self.rows.append(row)
                print(json.dumps(row), flush=True)
                self.path.write_text(json.dumps(self.rows))


def train(scenario: str, steps: int, out: Path, n_envs: int = 8, seed: int = 0, device: str = "auto", subproc: bool = True, tb: Path | None = None):
    sc = SCENARIOS[scenario]
    out.mkdir(parents=True, exist_ok=True)
    VecCls = SubprocVecEnv if subproc else DummyVecEnv
    env = VecCls([make_teacher_env(scenario, seed=seed * 100 + i) for i in range(n_envs)])
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
