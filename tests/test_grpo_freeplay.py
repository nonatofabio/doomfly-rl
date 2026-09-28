"""ShapedReward: each component is a delta of an engine counter; totals are raw counts, the
returned reward is sum_k W[k] * delta_k with health lost entering negatively."""
import math

from doomfly.grpo_freeplay import ShapedReward, W


def info(k=0, i=0, s=0, d=0, h=100, x=0.0, y=0.0, dead=False):
    return {"vars": {"KILLCOUNT": k, "ITEMCOUNT": i, "SECRETCOUNT": s, "DAMAGECOUNT": d,
                     "HEALTH": h, "ARMOR": 0, "POSITION_X": x, "POSITION_Y": y}, "dead": dead}


def test_components_and_total():
    sh = ShapedReward()
    sh.reset(info())
    assert sh.totals == {k: 0.0 for k in W}
    r1 = sh(0.0, info(k=1, d=50, h=80, x=10))                   # same cell as spawn
    assert math.isclose(r1, 1.0 + 0.5 - 0.2)
    r2 = sh(0.0, info(k=1, i=1, d=50, h=90, x=200))            # health gain is not rewarded
    assert math.isclose(r2, 1.0 + 0.1)                          # item + one new cell
    r3 = sh(1.0, info(k=1, i=1, d=50, h=90, x=200))            # native exit reward
    assert math.isclose(r3, 10.0)
    t = sh.totals
    assert t == {"kill": 1, "item": 1, "secret": 0, "damage": 50, "health": 20, "explore": 1, "exit": 1}
    weighted = sum((-1 if k == "health" else 1) * W[k] * v for k, v in t.items())
    assert math.isclose(weighted, r1 + r2 + r3)
    assert not sh.dead


def test_revisit_and_death():
    sh = ShapedReward()
    sh.reset(info())
    assert math.isclose(sh(0.0, info(x=300)), 0.1)
    assert sh(0.0, info(x=0)) == 0.0                            # spawn cell already counted
    # on the death step the engine vars can be stale (health still > 0): death costs all of it
    assert math.isclose(sh(0.0, info(h=35, x=0, dead=True)), -1.0)
    assert sh.dead and sh.totals["health"] == 100


def test_reset_clears():
    sh = ShapedReward()
    sh.reset(info())
    sh(0.0, info(k=2, x=500))
    sh.reset(info(k=2, x=500))                                  # counters carry across episodes
    assert sh.totals == {k: 0.0 for k in W} and not sh.dead
    assert sh(0.0, info(k=2, x=500)) == 0.0                     # new spawn cell is visited, no kill delta


def test_apply_init_and_heldout_seeds():
    """apply_init copies the prior's embedding row into FREEPLAY's and sets the NEXTW bias, and
    touches nothing else; freeplay_eval refuses the seed stream the runs trained and evaluated on."""
    import pytest
    import torch
    from doomfly.doom.actions import FREEPLAY, SCENARIOS
    from doomfly.freeplay_eval import heldout_seeds
    from doomfly.grpo_freeplay import NEXTW, apply_init
    from doomfly.model.flynet import FlyNet, FlyNetConfig
    from tests.test_surgery import tiny_connectome

    torch.manual_seed(0)
    m = FlyNet(tiny_connectome(), FlyNetConfig(d_model=32, stem_width=8))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    apply_init(m, "deadly_corridor", -4.0)
    after = m.state_dict()
    emb, bias = "scenario_emb.weight", "heads.policy.2.bias"
    assert torch.equal(after[emb][FREEPLAY.id], before[emb][SCENARIOS["deadly_corridor"].id])
    assert after[bias][NEXTW] == -4.0
    rows = [i for i in range(after[emb].shape[0]) if i != FREEPLAY.id]
    assert torch.equal(after[emb][rows], before[emb][rows])
    assert torch.equal(after[bias][:NEXTW], before[bias][:NEXTW])
    assert all(torch.equal(after[k], before[k]) for k in before if k not in (emb, bias))
    apply_init(m, "mean", -4.0)                                  # "mean" leaves the row alone
    assert torch.equal(m.state_dict()[emb][FREEPLAY.id], before[emb][SCENARIOS["deadly_corridor"].id])

    assert len(heldout_seeds(30, 1)) == 30
    with pytest.raises(SystemExit):
        heldout_seeds(10, 0)
