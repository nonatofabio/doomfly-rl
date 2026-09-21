"""FlyNet for Doom.

Pipeline (mirrors mlabonne/chessfly):
  frames [B, 4, 72, 96] uint8
    -> conv stem + linear      -> currents on the 10,855 visual sensory neurons
    -> 5 steps of connectome dynamics over all 138,639 neurons
         h <- (1-a) h + a * relu( gamma_t * norm(W h + u) + beta_t )
       W_ij = sign_ij * prior_ij * exp(theta_ij)  (wiring + sign frozen, gain learned)
    -> read the central-brain + descending + motor neurons (33,788)
    -> policy head (23 global actions, legality mask per scenario)
    -> value head (64 bins over normalised return)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse import Connectome
from ..doom.actions import N_ACTIONS, N_SCENARIO_IDS

FRAME_H, FRAME_W, FRAME_STACK = 72, 96, 4
N_VALUE_BINS = 64


@dataclass
class FlyNetConfig:
    steps: int = 5
    leak: float = 0.5
    stem_width: int = 64
    d_model: int = 512
    n_actions: int = N_ACTIONS
    n_value_bins: int = N_VALUE_BINS
    n_scenarios: int = N_SCENARIO_IDS  # five trained scenarios + the free-play row
    input_gain: float = 1.0
    # controls (docs/findings.md): -1 = real wiring; >= 0 = degree-preserving shuffle with that seed
    shuffle_seed: int = -1
    # stem -> decoder directly, no neurons, no dynamics (same stem/heads, decoder reads the stem output)
    no_connectome: bool = False


def shuffle_connectome(conn: dict, seed: int, max_rounds: int = 200) -> dict:
    """Degree-preserving null model of the wiring.

    Keeps `dst` fixed (exact in-degree per neuron) and permutes the presynaptic column together
    with its sign (exact out-degree per neuron, sign stays with the neuron that emits it);
    `syn_count` stays with the postsynaptic slot, so per-neuron input weight mass is unchanged.
    Input / readout neuron sets are untouched.  Duplicate (dst, src) pairs created by the
    permutation are resolved by re-permuting `src` among the colliding edges only, so both
    degree sequences are exact; whatever is still colliding after `max_rounds` is dropped.
    Self-loops are kept.
    """
    g = np.random.default_rng(seed)
    src, dst = np.asarray(conn["src"]).copy(), np.asarray(conn["dst"])
    sign, syn = np.asarray(conn["sign"]).copy(), np.asarray(conn["syn_count"])
    n = int(conn["n"])
    perm = g.permutation(len(src))
    src, sign = src[perm], sign[perm]
    dst64 = dst.astype(np.int64) * n
    for _ in range(max_rounds):
        _, first = np.unique(dst64 + src, return_index=True)
        dup = np.ones(len(src), bool); dup[first] = False
        idx = np.flatnonzero(dup)
        if len(idx) == 0:
            break
        p = g.permutation(len(idx))
        src[idx], sign[idx] = src[idx][p], sign[idx][p]
    _, keep = np.unique(dst64 + src, return_index=True)
    keep.sort()
    out = dict(conn)
    out.update(src=src[keep], dst=dst[keep], sign=sign[keep], syn_count=syn[keep])
    out["shuffled"] = np.array([seed, len(src) - len(keep)])  # seed, dropped duplicates
    return out


class Stem(nn.Module):
    """Retina: frames -> a compact feature vector -> input currents."""

    def __init__(self, n_inputs: int, width: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, width, 8, stride=4), nn.GELU(),          # 72x96 -> 17x23
            nn.Conv2d(width, width * 2, 4, stride=2), nn.GELU(),            # -> 7x10
            nn.Conv2d(width * 2, width * 2, 3, stride=1), nn.GELU(),        # -> 5x8
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W)).shape[1]
        self.proj = nn.Sequential(nn.Linear(flat, 1024), nn.GELU(), nn.Linear(1024, n_inputs))

    def forward(self, frames):  # uint8 [B,4,H,W]
        x = frames.float() / 255.0 - 0.5
        return self.proj(self.conv(x))


class FlyNet(nn.Module):
    def __init__(self, connectome: dict, cfg: FlyNetConfig):
        super().__init__()
        self.cfg = cfg
        n = int(connectome["n"])
        self.n = n
        if cfg.shuffle_seed >= 0 and not cfg.no_connectome:
            connectome = shuffle_connectome(connectome, cfg.shuffle_seed)
        self.n_dropped = int(connectome["shuffled"][1]) if "shuffled" in connectome else 0
        self.connectome = None if cfg.no_connectome else Connectome(
            connectome["src"], connectome["dst"], connectome["sign"], connectome["syn_count"], n)
        input_idx = torch.as_tensor(np.flatnonzero(connectome["is_input"]), dtype=torch.long)
        readout_idx = torch.as_tensor(np.flatnonzero(connectome["is_readout"]), dtype=torch.long)
        self.register_buffer("input_idx", input_idx)
        self.register_buffer("readout_idx", readout_idx)
        self.n_inputs, self.n_readout = len(input_idx), len(readout_idx)

        self.encoder = Stem(self.n_inputs, cfg.stem_width)
        if cfg.no_connectome:
            # no neurons: the decoder reads the stem's n_inputs currents directly
            self.norms = nn.ModuleList()
            dec_in = self.n_inputs
        else:
            # homeostatic per-neuron, per-step gain (scale) and threshold (shift), on top of a
            # non-affine batch-norm of the presynaptic drive (mu, sigma tracked as running stats)
            self.scale = nn.Parameter(torch.ones(cfg.steps, n))
            self.shift = nn.Parameter(torch.zeros(cfg.steps, n))
            self.norms = nn.ModuleList([nn.BatchNorm1d(n, affine=False, momentum=0.05) for _ in range(cfg.steps)])
            dec_in = self.n_readout
        self.scenario_emb = nn.Embedding(cfg.n_scenarios, cfg.d_model)
        self.decoder = nn.Sequential(nn.Linear(dec_in, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))
        self.heads = nn.ModuleDict({
            "policy": nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.n_actions)),
            "value": nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.n_value_bins)),
        })

    def dynamics(self, u_in, return_trace=False):
        """u_in: [B, n_inputs] currents. Returns final state [N, B] (and per-step trace)."""
        B = u_in.shape[0]
        n, a = self.n, self.cfg.leak
        u = torch.zeros(n, B, device=u_in.device, dtype=torch.float32)
        u[self.input_idx] = (u_in.float() * self.cfg.input_gain).t()
        h = torch.zeros(n, B, device=u_in.device, dtype=torch.float32)
        trace = []
        for t in range(self.cfg.steps):
            drive = self.connectome(h) + u                      # [N, B]
            z = self.norms[t](drive.t()).t()                    # normalise over batch, per neuron
            z = self.scale[t][:, None] * z + self.shift[t][:, None]
            h = (1 - a) * h + a * F.relu(z)
            if return_trace:
                trace.append(h.detach())
        return (h, trace) if return_trace else h

    def forward(self, frames, scenario_id, legal_mask=None, return_trace=False):
        u_in = self.encoder(frames)
        if self.cfg.no_connectome:
            r, trace = u_in.float(), None                       # [B, n_inputs]: stem -> decoder
        else:
            out = self.dynamics(u_in, return_trace)
            h, trace = out if return_trace else (out, None)
            r = h[self.readout_idx].t()                         # [B, n_readout]
        z = self.decoder(r) + self.scenario_emb(scenario_id)
        logits = self.heads["policy"](z)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e4)
        value_logits = self.heads["value"](z)
        res = {"policy": logits, "value": value_logits, "readout": r}
        if return_trace:
            res["trace"] = trace
        return res

    def param_groups(self):
        conn = [] if self.connectome is None else [self.connectome.log_gain]
        homeo = [] if self.connectome is None else [self.scale, self.shift]
        ids = {id(p) for p in conn + homeo}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return conn, homeo, rest

    def meta(self):
        edges = 0 if self.connectome is None else int(self.connectome.log_gain.numel())
        return {"config": asdict(self.cfg), "neurons": 0 if self.connectome is None else self.n, "edges": edges,
                "dropped_duplicates": self.n_dropped, "inputs": self.n_inputs,
                "readout": self.n_inputs if self.connectome is None else self.n_readout,
                "params": sum(p.numel() for p in self.parameters())}


def value_to_bin(v: torch.Tensor, n_bins: int = N_VALUE_BINS) -> torch.Tensor:
    """v in [0,1] -> bin index"""
    return (v.clamp(0, 1 - 1e-6) * n_bins).long()


def bin_to_value(logits: torch.Tensor) -> torch.Tensor:
    centers = (torch.arange(logits.shape[-1], device=logits.device) + 0.5) / logits.shape[-1]
    return (logits.softmax(-1) * centers).sum(-1)


def load_connectome(path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}
