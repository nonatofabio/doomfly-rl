"""Head surgery must be a no-op on the five trained scenarios (docs/freeplay-plan.md, step 2)."""
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from doomfly.doom.actions import FREEPLAY, N_ACTIONS, N_SCENARIOS, N_SCENARIO_IDS, SCENARIOS
from doomfly.model.flynet import FRAME_H, FRAME_STACK, FRAME_W, FlyNet, FlyNetConfig
from doomfly.surgery import EMB_W, POLICY_B, POLICY_W, expand_head, load_flynet, needs_surgery
from doomfly.train import legal_mask_table

N_NEURONS = 64


def tiny_connectome(seed=0) -> dict:
    """A 64-neuron random wiring with the same keys as data/processed/connectome_*.npz."""
    g = np.random.default_rng(seed)
    n_syn = 400
    src = g.integers(0, N_NEURONS, n_syn).astype(np.int64)
    dst = g.integers(0, N_NEURONS, n_syn).astype(np.int64)
    _, keep = np.unique(dst * N_NEURONS + src, return_index=True)
    is_input = np.zeros(N_NEURONS, bool); is_input[:16] = True
    is_readout = np.zeros(N_NEURONS, bool); is_readout[32:] = True
    return dict(n=np.array(N_NEURONS), src=src[keep], dst=dst[keep],
                sign=g.choice([-1.0, 1.0], len(keep)).astype(np.float32),
                syn_count=g.integers(1, 10, len(keep)).astype(np.float32),
                is_input=is_input, is_readout=is_readout)


OLD = FlyNetConfig(stem_width=8, d_model=32, n_actions=22, n_scenarios=5)


@pytest.fixture(params=[False, True], ids=["connectome", "no_connectome"])
def old_model(request):
    torch.manual_seed(0)
    cfg = FlyNetConfig(**{**asdict(OLD), "no_connectome": request.param})
    m = FlyNet(tiny_connectome(), cfg).eval()
    with torch.no_grad():  # make the head + embedding clearly non-trivial
        for p in m.parameters():
            p.add_(0.1 * torch.randn_like(p))
    return m


def frames(b=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, FRAME_STACK, FRAME_H, FRAME_W), generator=g, dtype=torch.uint8)


def test_constants():
    assert N_ACTIONS == 23 and N_SCENARIOS == 5 and N_SCENARIO_IDS == 6
    assert FREEPLAY.id == N_SCENARIOS and 22 in FREEPLAY.actions
    for s in SCENARIOS.values():
        assert 22 not in s.actions, s.name
    assert needs_surgery(OLD) and not needs_surgery(FlyNetConfig())


def test_expand_head_shapes_and_values(old_model):
    sd, cfg = expand_head(old_model.state_dict(), old_model.cfg)
    assert (cfg.n_actions, cfg.n_scenarios) == (N_ACTIONS, N_SCENARIO_IDS)
    old = old_model.state_dict()
    assert sd[POLICY_W].shape == (23, OLD.d_model) and sd[POLICY_B].shape == (23,)
    assert torch.equal(sd[POLICY_W][:22], old[POLICY_W]) and torch.equal(sd[POLICY_B][:22], old[POLICY_B])
    assert sd[POLICY_W][22].abs().sum() == 0 and sd[POLICY_B][22] == 0
    assert sd[EMB_W].shape == (6, OLD.d_model)
    assert torch.equal(sd[EMB_W][:5], old[EMB_W])
    assert torch.allclose(sd[EMB_W][5], old[EMB_W].mean(0))
    # input untouched
    assert old[POLICY_W].shape == (22, OLD.d_model)
    with pytest.raises(ValueError):
        expand_head(sd, cfg, n_actions=22)


def test_forward_is_identical_on_trained_scenarios(old_model):
    sd, cfg = expand_head(old_model.state_dict(), old_model.cfg)
    new = FlyNet(tiny_connectome(), cfg).eval()
    new.load_state_dict(sd, strict=True)
    fr = frames()
    legal_new = legal_mask_table("cpu")
    legal_old = legal_new[:5, :22]
    with torch.no_grad():
        for sid in range(N_SCENARIOS):
            sc = torch.full((fr.shape[0],), sid, dtype=torch.long)
            o = old_model(fr, sc, legal_mask=legal_old[sc])
            n = new(fr, sc, legal_mask=legal_new[sc])
            assert torch.equal(o["policy"], n["policy"][:, :22]), sid
            assert torch.all(n["policy"][:, 22] == -1e4), sid  # new action masked everywhere it is illegal
            assert torch.equal(o["value"], n["value"]), sid
            assert torch.equal(o["readout"], n["readout"]), sid
            # sampling distributions agree (masked logit contributes exp(-1e4) == 0 in fp32)
            assert torch.equal(o["policy"].softmax(-1), n["policy"].softmax(-1)[:, :22])
        # the free-play row is usable: finite logits, exactly the FREEPLAY legal set unmasked
        sc = torch.full((fr.shape[0],), FREEPLAY.id, dtype=torch.long)
        p = new(fr, sc, legal_mask=legal_new[sc])["policy"]
        assert torch.isfinite(p).all()
        assert set(torch.nonzero(p[0] > -1e4).flatten().tolist()) == set(FREEPLAY.actions)


def test_load_flynet_expands_old_checkpoint(old_model, tmp_path):
    ck = tmp_path / "ckpt"; ck.mkdir()
    save_file({k: v.contiguous() for k, v in old_model.state_dict().items()}, str(ck / "model.safetensors"))
    (ck / "config.json").write_text(json.dumps({"flynet": asdict(old_model.cfg), "step": 1}))
    conn = tiny_connectome()
    raw, cfg_raw = load_flynet(ck, conn, expand=False)
    assert (cfg_raw.n_actions, cfg_raw.n_scenarios) == (22, 5)
    m, cfg = load_flynet(ck, conn)
    raw.eval(); m.eval()  # loaders return train mode; BatchNorm in the connectome path must be frozen to compare
    assert (cfg.n_actions, cfg.n_scenarios) == (23, 6)
    assert m.heads["policy"][2].weight.shape == (23, OLD.d_model)
    assert m.scenario_emb.weight.shape == (6, OLD.d_model)
    fr = frames(b=3, seed=2)
    sc = torch.tensor([0, 2, 4])
    with torch.no_grad():
        assert torch.equal(raw(fr, sc)["policy"], m(fr, sc)["policy"][:, :22])
    # config.json on disk is unchanged
    assert json.loads((ck / "config.json").read_text())["flynet"]["n_actions"] == 22


def test_cli_writes_converted_checkpoint(old_model, tmp_path, monkeypatch):
    ck = tmp_path / "ckpt"; ck.mkdir()
    save_file({k: v.contiguous() for k, v in old_model.state_dict().items()}, str(ck / "model.safetensors"))
    (ck / "config.json").write_text(json.dumps({"flynet": asdict(old_model.cfg), "step": 7}))
    np.savez(tmp_path / "conn.npz", **tiny_connectome())
    out = tmp_path / "out"
    import sys
    from doomfly import surgery
    monkeypatch.setattr(sys, "argv", ["surgery", "--ckpt", str(ck), "--connectome", str(tmp_path / "conn.npz"), "--out", str(out)])
    surgery.main()
    meta = json.loads((out / "config.json").read_text())
    assert meta["flynet"]["n_actions"] == 23 and meta["flynet"]["n_scenarios"] == 6 and meta["step"] == 7
    assert meta["surgery"]["n_actions"] == [22, 23]
    m, cfg = load_flynet(out, tiny_connectome())
    m.eval()
    assert not needs_surgery(cfg)
    fr = frames(b=2, seed=3); sc = torch.tensor([1, 3])
    with torch.no_grad():
        assert torch.equal(old_model(fr, sc)["policy"], m(fr, sc)["policy"][:, :22])
    with pytest.raises(SystemExit):  # refuses to overwrite
        surgery.main()
