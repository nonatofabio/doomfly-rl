"""Doom action vocabulary shared across scenarios (the analogue of chessfly's
1968-move vocabulary with legal-move masking).

Every scenario exposes a subset of ViZDoom buttons. We define one global set of
discrete actions (button combinations) and a per-scenario legality mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# button names as in vizdoom.Button
B_ATTACK, B_USE, B_SPEED = "ATTACK", "USE", "SPEED"
B_FWD, B_BACK, B_LEFT, B_RIGHT = "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT"
B_TL, B_TR = "TURN_LEFT", "TURN_RIGHT"

# global vocabulary: tuples of buttons pressed together
ACTIONS: list[tuple[str, ...]] = [
    (),  # no-op
    (B_ATTACK,),
    (B_FWD,), (B_BACK,), (B_LEFT,), (B_RIGHT,), (B_TL,), (B_TR,),
    (B_FWD, B_ATTACK), (B_LEFT, B_ATTACK), (B_RIGHT, B_ATTACK),
    (B_TL, B_ATTACK), (B_TR, B_ATTACK),
    (B_FWD, B_TL), (B_FWD, B_TR),
    (B_FWD, B_LEFT), (B_FWD, B_RIGHT),
    (B_FWD, B_SPEED), (B_FWD, B_TL, B_SPEED), (B_FWD, B_TR, B_SPEED),
    (B_USE,), (B_FWD, B_USE),
]
N_ACTIONS = len(ACTIONS)
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class Scenario:
    name: str
    cfg: str  # vizdoom scenario config file name
    buttons: tuple[str, ...]  # buttons the scenario allows
    actions: tuple[int, ...]  # indices into ACTIONS legal here (the "legal moves")
    frame_skip: int = 4
    # reward normalisation for the value head (per-scenario, from teacher rollouts)
    reward_lo: float = 0.0
    reward_hi: float = 1.0
    ppo_steps: int = 1_000_000
    # teacher-side only: scale env reward before PPO sees it (Sample Factory uses 0.01 for the two
    # scenarios whose rewards are in the +-100s; unscaled, SB3 PPO w/ clip 0.1 collapsed to a constant action)
    reward_scale: float = 1.0
    # teacher-side only: add health/kill/ammo delta shaping (in raw reward units, before reward_scale)
    shaping: bool = False
    # applied in the env itself (teacher, recorder, eval all see it). None = scenario .cfg default (5).
    doom_skill: int | None = None
    # free play on a full WAD level (None = the scenario .cfg's own map). Env disables audio/automap
    # buffers and exposes GAME_VARS in info; frees us from writing a new .cfg per level.
    doom_map: str | None = None
    id: int = field(default=-1)


def _legal(buttons):
    bs = set(buttons)
    return tuple(i for i, a in enumerate(ACTIONS) if set(a) <= bs and (a or True))


SCENARIOS: dict[str, Scenario] = {}
for i, s in enumerate([
    Scenario("basic", "basic.cfg", (B_LEFT, B_RIGHT, B_ATTACK), (), ppo_steps=1_000_000, reward_scale=0.01),
    Scenario("defend_the_center", "defend_the_center.cfg", (B_TL, B_TR, B_ATTACK), (), ppo_steps=1_500_000),
    Scenario("health_gathering", "health_gathering.cfg", (B_TL, B_TR, B_FWD), (), ppo_steps=1_500_000),
    Scenario("deadly_corridor", "deadly_corridor.cfg", (B_ATTACK, B_LEFT, B_RIGHT, B_FWD, B_BACK, B_TL, B_TR), (),
             ppo_steps=8_000_000, reward_scale=0.01, shaping=True, doom_skill=3),
    Scenario("defend_the_line", "defend_the_line.cfg", (B_TL, B_TR, B_ATTACK), (), ppo_steps=1_500_000),
]):
    s.id = i
    s.actions = _legal(s.buttons)
    SCENARIOS[s.name] = s

N_SCENARIOS = len(SCENARIOS)

# Free play on Freedoom II (ships with vizdoom). Deliberately NOT in SCENARIOS: it has no teacher,
# no rollouts and no scenario embedding row; the model must borrow the id/legal-set of a trained
# scenario (zero-shot) until a proper free-play head is added. Buttons = deadly_corridor's 7, the
# widest set the distilled student was ever trained on (legal global actions 0..16).
FREEPLAY = Scenario("freedoom2_map01", "freedoom2.cfg", (B_ATTACK, B_LEFT, B_RIGHT, B_FWD, B_BACK, B_TL, B_TR), (),
                    doom_skill=3, doom_map="MAP01")
FREEPLAY.actions = _legal(FREEPLAY.buttons)

# game variables the env reports in info (order matters: indexes into state.game_variables)
GAME_VARS = ("KILLCOUNT", "ITEMCOUNT", "SECRETCOUNT", "DAMAGECOUNT", "HEALTH", "ARMOR", "POSITION_X", "POSITION_Y")


def legal_mask(scenario: Scenario) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[list(scenario.actions)] = True
    return m


def action_to_buttons(scenario: Scenario, action_idx: int) -> list[bool]:
    """ViZDoom wants a bool per *scenario* button (in scenario button order)."""
    pressed = set(ACTIONS[action_idx])
    return [b in pressed for b in scenario.buttons]


def local_to_global(scenario: Scenario) -> np.ndarray:
    """map from the scenario's local discrete action index -> global action index"""
    return np.array(scenario.actions, dtype=np.int64)
