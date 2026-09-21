"""Head surgery: grow a scenario-distilled student for free play (docs/freeplay-plan.md, step 2).

The five-scenario students were trained with a 22-action policy head and a 5-row scenario embedding.
Free play on MAP01 needs one more action (SELECT_NEXT_WEAPON, global index 22) and its own scenario id
(FREEPLAY.id == 5). `expand_head` grows both without changing what the network computes on the five
scenarios:

  policy last layer   [22, d] -> [23, d]   old rows copied, new row zero weight + zero bias
  scenario_emb        [5, d]  -> [6, d]    old rows copied, new row = mean of the five trained rows

The new action is masked out (-1e4) on every trained scenario, and the new embedding row is only ever
looked up with FREEPLAY.id, so logits/value on scenarios 0..4 are bit-identical before and after.

`load_flynet` is the one loader every script should use: it builds the model from the checkpoint's own
config, loads the weights, and expands the head in memory if the checkpoint predates the surgery.

CLI: write the expanded checkpoint to disk once, so later runs (GRPO on MAP01) start from a normal
checkpoint directory:

    python -m doomfly.surgery --ckpt checkpoints/malecns49k --connectome data/processed/connectome_malecns49k.npz \\
        --out checkpoints/malecns49k_fp
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .doom.actions import N_ACTIONS, N_SCENARIO_IDS
from .model.flynet import FlyNet, FlyNetConfig, load_connectome

POLICY_W, POLICY_B = "heads.policy.2.weight", "heads.policy.2.bias"
EMB_W = "scenario_emb.weight"


def needs_surgery(cfg: FlyNetConfig) -> bool:
    return cfg.n_actions < N_ACTIONS or cfg.n_scenarios < N_SCENARIO_IDS


def expand_head(sd: dict[str, torch.Tensor], cfg: FlyNetConfig,
                n_actions: int = N_ACTIONS, n_scenarios: int = N_SCENARIO_IDS) -> tuple[dict, FlyNetConfig]:
    """Return (new state dict, new config). Input tensors are not modified."""
    if n_actions < cfg.n_actions or n_scenarios < cfg.n_scenarios:
        raise ValueError(f"cannot shrink: {cfg.n_actions}->{n_actions} actions, {cfg.n_scenarios}->{n_scenarios} scenarios")
    sd = dict(sd)
    w, b = sd[POLICY_W], sd[POLICY_B]
    assert w.shape[0] == cfg.n_actions == b.shape[0], (w.shape, b.shape, cfg.n_actions)
    if n_actions > cfg.n_actions:
        pad = n_actions - cfg.n_actions
        sd[POLICY_W] = torch.cat([w, torch.zeros(pad, w.shape[1], dtype=w.dtype)], 0)
        sd[POLICY_B] = torch.cat([b, torch.zeros(pad, dtype=b.dtype)], 0)
    e = sd[EMB_W]
    assert e.shape[0] == cfg.n_scenarios, (e.shape, cfg.n_scenarios)
    if n_scenarios > cfg.n_scenarios:
        mean = e.float().mean(0, keepdim=True).to(e.dtype)
        sd[EMB_W] = torch.cat([e, mean.expand(n_scenarios - cfg.n_scenarios, -1).clone()], 0)
    return sd, replace(cfg, n_actions=n_actions, n_scenarios=n_scenarios)


def load_flynet(ckpt: Path, connectome: Path | dict, device="cpu", expand: bool = True) -> tuple[FlyNet, FlyNetConfig]:
    """Build + load a checkpoint directory (config.json, model.safetensors). Pre-surgery checkpoints are
    expanded in memory when `expand` is set; the on-disk files are never touched."""
    ckpt = Path(ckpt)
    cfg = FlyNetConfig(**json.loads((ckpt / "config.json").read_text())["flynet"])
    sd = load_file(str(ckpt / "model.safetensors"))
    if expand and needs_surgery(cfg):
        sd, cfg = expand_head(sd, cfg)
    conn = connectome if isinstance(connectome, dict) else load_connectome(connectome)
    model = FlyNet(conn, cfg).to(device)
    model.load_state_dict(sd, strict=True)
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="pre-surgery checkpoint directory")
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new checkpoint directory (must not exist)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; refusing to overwrite")
    meta = json.loads((a.ckpt / "config.json").read_text())
    old = FlyNetConfig(**meta["flynet"])
    if not needs_surgery(old):
        raise SystemExit(f"{a.ckpt} already has {old.n_actions} actions / {old.n_scenarios} scenario rows")
    sd, cfg = expand_head(load_file(str(a.ckpt / "model.safetensors")), old)
    # loading through FlyNet validates the shapes against the connectome
    model = FlyNet(load_connectome(a.connectome), cfg)
    model.load_state_dict(sd, strict=True)
    a.out.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(a.out / "model.safetensors"))
    meta["flynet"] = asdict(cfg)
    meta["surgery"] = {"from": str(a.ckpt), "n_actions": [old.n_actions, cfg.n_actions],
                       "n_scenarios": [old.n_scenarios, cfg.n_scenarios],
                       "new_policy_rows": "zero", "new_scenario_rows": "mean of trained rows"}
    (a.out / "config.json").write_text(json.dumps(meta, indent=1))
    n = sum(p.numel() for p in model.parameters())
    print(f"wrote {a.out}: actions {old.n_actions}->{cfg.n_actions}, scenario rows {old.n_scenarios}->{cfg.n_scenarios}, "
          f"{n:,} params")


if __name__ == "__main__":
    main()
