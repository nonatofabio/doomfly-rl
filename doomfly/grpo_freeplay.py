"""GRPO on Freedoom II MAP01 (docs/freeplay-plan.md, step 3).

Same critic-free update as doomfly.grpo (seed-matched groups, RLOO baseline, PPO-clip), on one
scenario (FREEPLAY) with a shaped reward, because MAP01's only native reward is the exit.

Shaped reward per decision (weights in W, every component logged separately):

  kill     +1      per KILLCOUNT increment
  item     +1      per ITEMCOUNT increment
  secret   +1      per SECRETCOUNT increment
  damage   +1/100  per point of DAMAGECOUNT (damage dealt)
  health   -1/100  per point of health lost (death counts the remaining health)
  explore  +0.1    per new 128x128-unit POSITION cell
  exit     +10     x native map_exit_reward (1 on exit)

Init. The surgery row for FREEPLAY (mean of the five trained embeddings) gives a close-to-uniform
policy on all three backbones: entropy 2.2-2.5 nats over the 18 legal actions (log 18 = 2.89).
With deadly_corridor's row the entropy is 0.8-1.1 nats. So by default the free-play row is
overwritten with deadly_corridor's row: the run starts from the same behaviour as
`pi_ref_as_deadly_corridor` in docs/freeplay.md, whose button set it shares.
The zero-initialised SELECT_NEXT_WEAPON row has logit 0, which is on par with the best trained
action, so it starts at bias -4. P(NEXTW) is then 1.8% (connectome), 0.9% (shuffled s0) and
6.7% (no-connectome); logits 0..16 equal deadly_corridor's exactly.
(All numbers: 41 frames of a random-action MAP01 walk, reset(seed=123), CPU.)
Both are flags and go in meta.json.

Eval: the zero-shot protocol of docs/freeplay.md (the same 10 seeds, default_rng(0), 1024
decisions, temperature 1.0, sampled), at iter 0 and every --eval-every iterations. Envs here are
reseeded through reset(seed=...), not built with DoomEnv(seed=...), so episodes need not replay
the zero-shot files one-for-one; iter 0 of eval.jsonl is the baseline for the run.

    python -m doomfly.grpo_freeplay --ckpt checkpoints/malecns49k_v2_final \\
        --connectome data/processed/connectome_malecns49k.npz --out runs/malecns49k_grpo_freeplay
"""
from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .doom.actions import FREEPLAY, SCENARIOS
from .doom.env import DoomEnv
from .grpo import advantages, flat_batch, grpo_step, load_student, sample_groups
from .train import legal_mask_table, s3_sync

W = {"kill": 1.0, "item": 1.0, "secret": 1.0, "damage": 0.01, "health": 0.01, "explore": 0.1, "exit": 10.0}
CELL = 128.0
NEXTW = 22


class ShapedReward:
    """Per-env shaper for sample_groups. totals holds raw per-component counts for the episode
    (kills, items, secrets, damage points, health points lost, new cells, exits); the returned
    reward is sum_k W[k] * delta_k (health enters with a minus sign)."""

    def __init__(self, w=W):
        self.w = w

    @staticmethod
    def _cell(v):
        return int(v["POSITION_X"] // CELL), int(v["POSITION_Y"] // CELL)

    def reset(self, info):
        v = info["vars"]
        self.prev = dict(v)
        self.cells = {self._cell(v)}
        self.totals = {k: 0.0 for k in self.w}
        self.dead = False

    def __call__(self, r, info):
        v, p = info["vars"], self.prev
        self.dead = bool(info["dead"])
        h = 0.0 if self.dead else max(v["HEALTH"], 0.0)   # vars are stale on the death step
        d = {"kill": v["KILLCOUNT"] - p["KILLCOUNT"], "item": v["ITEMCOUNT"] - p["ITEMCOUNT"],
             "secret": v["SECRETCOUNT"] - p["SECRETCOUNT"], "damage": v["DAMAGECOUNT"] - p["DAMAGECOUNT"],
             "health": max(0.0, max(p["HEALTH"], 0.0) - h), "explore": 0.0, "exit": float(r)}
        c = self._cell(v)
        if c not in self.cells:
            self.cells.add(c)
            d["explore"] = 1.0
        self.prev = {**v, "HEALTH": h}
        out = 0.0
        for k, x in d.items():
            self.totals[k] += x
            out += (-1.0 if k == "health" else 1.0) * self.w[k] * x
        return out


def summarise(trajs):
    """mean/std over episodes of R, T, dead and every raw component."""
    cols = {"R": [t["R"] for t in trajs], "T": [t["T"] for t in trajs], "dead": [float(t["dead"]) for t in trajs]}
    for k in trajs[0]["comps"]:
        cols[k] = [t["comps"][k] for t in trajs]
    return {k: {"mean": float(np.mean(x)), "std": float(np.std(x))} for k, x in cols.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="distilled student (pre- or post-surgery)")
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tb", type=Path, default=None)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-ep-steps", type=int, default=512, help="decisions per training segment")
    ap.add_argument("--baseline", choices=["loo", "mean"], default="loo")
    ap.add_argument("--adv-norm", choices=["std", "none"], default="std")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.0, help="KL to the (re-initialised) start policy; plan: 0")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-conn", type=float, default=3e-4)
    ap.add_argument("--temp-start", type=float, default=1.2)
    ap.add_argument("--temp-end", type=float, default=1.0, help="linear anneal over --iters")
    ap.add_argument("--emb-from", default="deadly_corridor", help="scenario whose embedding row seeds FREEPLAY, or 'mean'")
    ap.add_argument("--nextw-bias", type=float, default=-4.0, help="initial SELECT_NEXT_WEAPON logit bias")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--eval-steps", type=int, default=1024)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    model, cfg = load_student(a.ckpt, a.connectome, dev)
    sid = FREEPLAY.id
    with torch.no_grad():
        emb = model.get_parameter("scenario_emb.weight")
        if a.emb_from != "mean":
            emb[sid] = emb[SCENARIOS[a.emb_from].id]
        model.get_parameter("heads.policy.2.bias")[NEXTW] = a.nextw_bias
    model.eval()   # BatchNorm1d per recurrent step: stay in eval for sampling and update (see grpo.py)
    ref = None
    if a.beta:
        ref = deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    legal = legal_mask_table(dev)[sid]
    conn_p, homeo_p, rest_p = model.param_groups()
    opt = torch.optim.AdamW([{"params": conn_p, "lr": a.lr_conn}, {"params": homeo_p, "lr": a.lr},
                             {"params": rest_p, "lr": a.lr}], betas=(0.9, 0.95), weight_decay=0.0)
    init = {"emb_from": a.emb_from, "nextw_bias": a.nextw_bias}
    (a.out / "meta.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()},
                                                 "model": model.meta(), "reward_weights": W, "cell": CELL,
                                                 "init": init, "scenario": FREEPLAY.name}, indent=1))

    n_env = a.groups * a.group_size
    envs = [DoomEnv(FREEPLAY) for _ in range(n_env)]
    assert a.eval_episodes <= n_env
    eval_seeds = np.random.default_rng(0).integers(0, 2**31 - 1, size=a.eval_episodes)   # seeds of freeplay_zero_shot --seed 0
    rng = np.random.default_rng(a.seed)
    tb = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(a.tb))
    log = open(a.out / "metrics.jsonl", "a")

    def save(it, tag):
        d = a.out / tag
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
        (d / "config.json").write_text(json.dumps({"flynet": asdict(cfg), "connectome": a.connectome.name,
                                                   "iter": it, "init": init}, indent=1))

    def run_eval(it):
        te = time.time()
        trajs = sample_groups(model, envs[:a.eval_episodes], eval_seeds, sid, legal, dev, a.eval_steps,
                              1.0, [ShapedReward() for _ in range(a.eval_episodes)])
        rec = {"iter": it, "episodes": a.eval_episodes, "max_steps": a.eval_steps, "temperature": 1.0,
               "seeds": [int(s) for s in eval_seeds], **summarise(trajs), "t_eval": time.time() - te}
        with open(a.out / "eval.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("EVAL", json.dumps({k: (v["mean"] if isinstance(v, dict) else v) for k, v in rec.items() if k != "seeds"}), flush=True)
        if tb:
            for k, v in rec.items():
                if isinstance(v, dict):
                    tb.add_scalar(f"eval/{k}", v["mean"], it)
        if a.s3:
            s3_sync(a.out, a.s3)

    t0 = time.time()
    frames = 0
    run_eval(0)
    for it in range(1, a.iters + 1):
        temp = a.temp_start + (a.temp_end - a.temp_start) * (it - 1) / max(1, a.iters - 1)
        seeds = rng.integers(0, 2**31 - 1, size=a.groups)
        ts = time.time()
        trajs = sample_groups(model, envs, seeds, sid, legal, dev, a.max_ep_steps, temp,
                              [ShapedReward() for _ in range(n_env)])
        t_sample = time.time() - ts
        adv = advantages(trajs, a.baseline, a.adv_norm)
        batch = flat_batch(trajs, adv)
        frames += len(batch[1])
        ts = time.time()
        st = grpo_step(model, ref, opt, batch, sid, legal, dev, a.clip, a.beta, a.ent_coef, a.epochs,
                       a.minibatch, temperature=temp)
        t_update = time.time() - ts
        s = summarise(trajs)
        rec = {"iter": it, "temperature": temp, "return_mean": s["R"]["mean"], "return_std": s["R"]["std"],
               "return_max": max(t["R"] for t in trajs), "ep_len": s["T"]["mean"], "dead": s["dead"]["mean"],
               **{f"c_{k}": s[k]["mean"] for k in W}, "adv_abs": float(np.abs(adv).mean()), "frames": frames,
               "t_sample": t_sample, "t_update": t_update, "elapsed": time.time() - t0, **st}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if tb:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and k != "iter":
                    tb.add_scalar(f"freeplay/{k}", v, it)
        if it % a.ckpt_every == 0 or it == a.iters:
            save(it, "latest")
        if it % a.eval_every == 0 or it == a.iters:
            run_eval(it)
    save(a.iters, "final")
    if a.s3:
        s3_sync(a.out, a.s3)
    for e in envs:
        e.close()


if __name__ == "__main__":
    main()
