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
    id: int = field(default=-1)


def _legal(buttons):
    bs = set(buttons)
    return tuple(i for i, a in enumerate(ACTIONS) if set(a) <= bs and (a or True))


SCENARIOS: dict[str, Scenario] = {}
for i, s in enumerate([
    Scenario("basic", "basic.cfg", (B_LEFT, B_RIGHT, B_ATTACK), (), ppo_steps=300_000),
    Scenario("defend_the_center", "defend_the_center.cfg", (B_TL, B_TR, B_ATTACK), (), ppo_steps=1_500_000),
    Scenario("health_gathering", "health_gathering.cfg", (B_TL, B_TR, B_FWD), (), ppo_steps=1_500_000),
    Scenario("deadly_corridor", "deadly_corridor.cfg", (B_ATTACK, B_LEFT, B_RIGHT, B_FWD, B_BACK, B_TL, B_TR), (), ppo_steps=3_000_000),
    Scenario("defend_the_line", "defend_the_line.cfg", (B_TL, B_TR, B_ATTACK), (), ppo_steps=1_500_000),
]):
    s.id = i
    s.actions = _legal(s.buttons)
    SCENARIOS[s.name] = s

N_SCENARIOS = len(SCENARIOS)


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
