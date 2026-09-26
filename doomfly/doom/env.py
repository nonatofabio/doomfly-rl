"""Gymnasium wrapper around ViZDoom with the global action vocabulary."""
from __future__ import annotations

import os
from collections import deque

import gymnasium as gym
import numpy as np
import vizdoom as vzd
from gymnasium import spaces

from .actions import SCENARIOS, Scenario, action_to_buttons, ACTIONS, GAME_VARS

FRAME_H, FRAME_W, FRAME_STACK = 72, 96, 4


def _resize_gray(img: np.ndarray) -> np.ndarray:
    """RGB [H,W,3] (or gray [H,W]) uint8 -> gray [72,96] uint8 via area-average.

    ViZDoom is run at RES_320X240 -> 240x320 -> integer factor 240/72 is not
    integral, so we use 216x288 crop from the centre then 3x3 pooling.
    """
    if img.ndim == 3:
        g = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])
    else:
        g = img.astype(np.float32)
    H, W = g.shape
    ch, cw = FRAME_H * 3, FRAME_W * 3
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    g = g[y0:y0 + ch, x0:x0 + cw]
    g = g.reshape(FRAME_H, 3, FRAME_W, 3).mean(axis=(1, 3))
    return g.astype(np.uint8)


class DoomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, scenario: str | Scenario, frame_skip: int | None = None, render: bool = False, seed: int | None = None,
                 record_tics: bool = False):
        self.sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
        self.frame_skip = frame_skip or self.sc.frame_skip
        # record_tics: advance the engine one tic at a time inside the frame skip and keep every
        # screen buffer in self.tic_frames (35 fps footage instead of one frame per decision).
        # Same reward/observation semantics as make_action(a, frame_skip); only used for video.
        self.record_tics = record_tics
        self.tic_frames: list[np.ndarray] = []
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, self.sc.cfg))
        g.set_window_visible(False)
        g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_available_buttons([getattr(vzd.Button, b) for b in self.sc.buttons])
        g.set_mode(vzd.Mode.PLAYER)
        g.set_render_hud(True)
        if self.sc.doom_map is not None:  # free play on a full WAD level
            g.set_doom_map(self.sc.doom_map)
            g.set_audio_buffer_enabled(False)
            g.set_automap_buffer_enabled(False)
            g.set_available_game_variables([getattr(vzd.GameVariable, v) for v in GAME_VARS])
        if self.sc.doom_skill is not None:
            g.set_doom_skill(self.sc.doom_skill)
        if seed is not None:
            g.set_seed(seed)
        g.init()
        self.game = g
        self.render_mode = "rgb_array" if render else None
        self.local_actions = list(self.sc.actions)  # global indices legal here
        self.action_space = spaces.Discrete(len(self.local_actions))
        self.observation_space = spaces.Box(0, 255, (FRAME_STACK, FRAME_H, FRAME_W), np.uint8)
        self.frames = deque(maxlen=FRAME_STACK)
        self.last_rgb = None
        self.last_vars = np.zeros(len(GAME_VARS))

    def _obs(self):
        st = self.game.get_state()
        if st is not None:
            self.last_rgb = st.screen_buffer
            self.frames.append(_resize_gray(st.screen_buffer))
        return np.stack(self.frames)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:  # reseed the engine so seed-matched envs replay the same episode (GRPO groups)
            self.game.set_seed(int(seed))
        self.game.new_episode()
        self.frames.clear()
        self.tic_frames.clear()
        st = self.game.get_state()
        self.last_rgb = st.screen_buffer
        if self.record_tics:
            self.tic_frames.append(st.screen_buffer)
        f = _resize_gray(st.screen_buffer)
        for _ in range(FRAME_STACK):
            self.frames.append(f)
        info = {}
        if self.sc.doom_map is not None:  # spawn-time vars, so reward shapers see the starting health/position
            self.last_vars = st.game_variables
            info["vars"] = dict(zip(GAME_VARS, map(float, self.last_vars)))
            info["dead"] = False
        return np.stack(self.frames), info

    def step(self, local_action: int):
        gidx = self.local_actions[int(local_action)]
        buttons = action_to_buttons(self.sc, gidx)
        if self.record_tics:
            self.game.set_action(buttons)
            r = 0.0
            for _ in range(self.frame_skip):
                self.game.advance_action(1, True)
                r += self.game.get_last_reward()
                st = self.game.get_state()
                if st is not None:
                    self.tic_frames.append(st.screen_buffer)
                if self.game.is_episode_finished():
                    break
        else:
            r = self.game.make_action(buttons, self.frame_skip)
        done = self.game.is_episode_finished()
        obs = np.stack(self.frames) if done else self._obs()
        info = {"global_action": gidx}
        if self.sc.doom_map is not None:
            st = self.game.get_state()
            if st is not None:
                self.last_vars = st.game_variables
            info["vars"] = dict(zip(GAME_VARS, map(float, self.last_vars)))
            info["dead"] = bool(self.game.is_player_dead())
        return obs, float(r), done, False, info

    def render(self):
        return self.last_rgb

    def close(self):
        self.game.close()


def make_env(scenario: str, seed: int = 0, render: bool = False):
    def _f():
        e = DoomEnv(scenario, render=render, seed=seed)
        return e
    return _f
