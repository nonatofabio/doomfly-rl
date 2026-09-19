"""Critic-free in-env fine-tuning of a distilled FlyNet with GRPO / RLOO.

Why no critic: FlyNet's value head is a 64-bin classifier bolted onto a sparse
connectome readout; it is the weakest part of the model and PPO's GAE leans on
it hard. GRPO (Shao et al. 2024) replaces the learned baseline with the mean
return of a *group* of episodes sampled from the same start state. Here a group
is G environments reset with the same ViZDoom seed, so every member sees the
same map layout / monster spawns and only the policy's own stochasticity
differs. Advantage per episode:

    RLOO   A_i = R_i - mean_{j != i} R_j              (Kool et al. 2019)
    GRPO   A_i = (R_i - mean_j R_j) / (std_j R_j + eps)

broadcast to every step of episode i (outcome supervision). Update = PPO-clipped
ratio * A - beta * KL(pi || pi_ref) - ent_coef * H, with pi_ref the frozen
distilled student, so the fly can improve on the teachers without forgetting
what the teachers taught. No value loss, no GAE, no bootstrapping.

Usage:
  python -m doomfly.grpo --ckpt runs/flywire783/final --connectome data/processed/connectome_783.npz \
      --out runs/flywire783_grpo --iters 300 --groups 4 --group-size 8
"""
from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .doom.actions import SCENARIOS, N_ACTIONS
from .doom.env import DoomEnv
from .evaluate import evaluate
from .model.flynet import FlyNet, FlyNetConfig, load_connectome
from .train import legal_mask_table, s3_sync


# ----------------------------------------------------------------------------- sampling
@torch.no_grad()
def sample_groups(model, envs, seeds, sid, legal_row, device, max_steps, temperature=1.0):
    """Roll one episode in every env (env k of group g gets seeds[g]). Returns a
    flat list of trajectories: dict(obs uint8 [T,4,H,W], act int64 [T], logp f32 [T], R, T, group)."""
    G = len(envs) // len(seeds)
    obs = np.stack([e.reset(seed=int(seeds[k // G]))[0] for k, e in enumerate(envs)])
    n = len(envs)
    local = np.asarray(envs[0].sc.actions)
    g2l = {int(g): i for i, g in enumerate(local)}
    alive = np.ones(n, dtype=bool)
    traj = [{"obs": [], "act": [], "logp": [], "rew": []} for _ in range(n)]
    for t in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        fr = torch.from_numpy(obs[idx]).to(device)
        sc = torch.full((idx.size,), sid, device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(fr, sc, legal_mask=legal_row[None].expand(idx.size, -1))["policy"].float()
        logits = logits / temperature
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        logp = dist.log_prob(act).cpu().numpy()
        act_np = act.cpu().numpy()
        for j, k in enumerate(idx):
            o, r, done, _, _ = envs[k].step(g2l[int(act_np[j])])
            tr = traj[k]
            tr["obs"].append(obs[k].copy()); tr["act"].append(int(act_np[j])); tr["logp"].append(float(logp[j])); tr["rew"].append(r)
            obs[k] = o
            if done:
                alive[k] = False
    out = []
    for k, tr in enumerate(traj):
        out.append({"obs": np.stack(tr["obs"]), "act": np.asarray(tr["act"], np.int64),
                    "logp": np.asarray(tr["logp"], np.float32), "R": float(np.sum(tr["rew"])),
                    "T": len(tr["act"]), "group": k // G})
    return out


def advantages(trajs, baseline="loo", norm="std", eps=1e-6):
    """Episode-level advantage from within-group returns; returns one float per trajectory."""
    groups = {}
    for i, tr in enumerate(trajs):
        groups.setdefault(tr["group"], []).append(i)
    adv = np.zeros(len(trajs), np.float32)
    for idxs in groups.values():
        R = np.array([trajs[i]["R"] for i in idxs], np.float64)
        G = len(R)
        if baseline == "loo" and G > 1:
            base = (R.sum() - R) / (G - 1)
        else:
            base = np.full(G, R.mean())
        a = R - base
        if norm == "std" and G > 1:
            a = a / (R.std() + eps)
        adv[idxs] = a
    return adv


# ----------------------------------------------------------------------------- update
def flat_batch(trajs, adv):
    obs = np.concatenate([t["obs"] for t in trajs])
    act = np.concatenate([t["act"] for t in trajs])
    logp = np.concatenate([t["logp"] for t in trajs])
    A = np.concatenate([np.full(t["T"], adv[i], np.float32) for i, t in enumerate(trajs)])
    return obs, act, logp, A


def grpo_step(model, ref, opt, batch, sid, legal_row, device, clip=0.2, beta=0.05, ent_coef=0.0,
              epochs=2, minibatch=256, max_grad_norm=1.0, use_bf16=True):
    obs, act, logp_old, A = batch
    N = len(act)
    stats = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "ent": 0.0, "clipfrac": 0.0, "ratio": 0.0}
    n_mb = 0
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for s in range(0, N, minibatch):
            mb = perm[s:s + minibatch]
            fr = torch.from_numpy(obs[mb]).to(device)
            a = torch.from_numpy(act[mb]).to(device)
            lp_old = torch.from_numpy(logp_old[mb]).to(device)
            adv = torch.from_numpy(A[mb]).to(device)
            sc = torch.full((len(mb),), sid, device=device, dtype=torch.long)
            mask = legal_row[None].expand(len(mb), -1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16 and device.type == "cuda"):
                logits = model(fr, sc, legal_mask=mask)["policy"].float()
                with torch.no_grad():
                    ref_logits = ref(fr, sc, legal_mask=mask)["policy"].float()
            logp = F.log_softmax(logits, -1)
            lp = logp.gather(1, a[:, None])[:, 0]
            ratio = torch.exp(lp - lp_old)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            ref_logp = F.log_softmax(ref_logits, -1)
            p = logp.exp()
            kl = (p * (logp - ref_logp)).masked_fill(~mask, 0.0).sum(-1).mean()   # KL(pi || pi_ref), full-distribution
            ent = -(p * logp).masked_fill(~mask, 0.0).sum(-1).mean()
            loss = pg + beta * kl - ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["loss"] += loss.item(); stats["pg"] += pg.item(); stats["kl"] += kl.item(); stats["ent"] += ent.item()
                stats["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item(); stats["ratio"] += ratio.mean().item()
            n_mb += 1
    return {k: v / max(1, n_mb) for k, v in stats.items()}


# ----------------------------------------------------------------------------- main
def load_student(ckpt: Path, connectome: Path, device):
    cfg = FlyNetConfig(**json.loads((ckpt / "config.json").read_text())["flynet"])
    model = FlyNet(load_connectome(connectome), cfg).to(device)
    model.load_state_dict(load_file(ckpt / "model.safetensors"))
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="distilled student dir (config.json + model.safetensors)")
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tb", type=Path, default=None)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--iters", type=int, default=300, help="one iteration = one scenario, groups x group-size episodes, one update")
    ap.add_argument("--groups", type=int, default=4, help="seed-matched groups per iteration")
    ap.add_argument("--group-size", type=int, default=8, help="episodes per group (same seed)")
    ap.add_argument("--max-ep-steps", type=int, default=600)
    ap.add_argument("--baseline", choices=["loo", "mean"], default="loo")
    ap.add_argument("--adv-norm", choices=["std", "none"], default="std")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.05, help="KL(pi||pi_ref) coefficient")
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-conn", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_student(a.ckpt, a.connectome, dev)
    ref = deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    legal = legal_mask_table(dev)
    conn_p, homeo_p, rest_p = model.param_groups()
    opt = torch.optim.AdamW([{"params": conn_p, "lr": a.lr_conn, "weight_decay": 0.0},
                             {"params": homeo_p, "lr": a.lr, "weight_decay": 0.0},
                             {"params": rest_p, "lr": a.lr, "weight_decay": 0.0}], betas=(0.9, 0.95))
    (a.out / "meta.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "model": model.meta()}, indent=1))

    names = a.scenarios.split(",")
    n_env = a.groups * a.group_size
    envs = {nm: [DoomEnv(nm) for _ in range(n_env)] for nm in names}
    rng = np.random.default_rng(a.seed)
    tb = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(a.tb))
    log = open(a.out / "metrics.jsonl", "a")

    def save(it, tag):
        d = a.out / tag
        d.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, d / "model.safetensors")
        (d / "config.json").write_text(json.dumps({"flynet": asdict(cfg), "connectome": str(a.connectome.name), "iter": it}))

    t0 = time.time()
    frames = 0
    for it in range(1, a.iters + 1):
        nm = names[(it - 1) % len(names)]
        sc = SCENARIOS[nm]
        seeds = rng.integers(0, 2**31 - 1, size=a.groups)
        model.eval()
        ts = time.time()
        trajs = sample_groups(model, envs[nm], seeds, sc.id, legal[sc.id], dev, a.max_ep_steps, a.temperature)
        t_sample = time.time() - ts
        adv = advantages(trajs, a.baseline, a.adv_norm)
        batch = flat_batch(trajs, adv)
        frames += len(batch[1])
        # Stay in eval mode: FlyNet has BatchNorm1d per recurrent step. train() would switch BN to batch
        # statistics, so pi(a|s) at the update would differ from pi at sampling (ratio != 1 at step 0,
        # every sample clipped) and the running stats would drift. Gradients flow fine in eval mode.
        model.eval()
        ts = time.time()
        st = grpo_step(model, ref, opt, batch, sc.id, legal[sc.id], dev, a.clip, a.beta, a.ent_coef, a.epochs, a.minibatch)
        t_update = time.time() - ts
        R = np.array([t["R"] for t in trajs])
        rec = {"iter": it, "scenario": nm, "return_mean": float(R.mean()), "return_std": float(R.std()),
               "return_max": float(R.max()), "ep_len": float(np.mean([t["T"] for t in trajs])),
               "adv_abs": float(np.abs(adv).mean()), "frames": frames, "t_sample": t_sample, "t_update": t_update,
               "elapsed": time.time() - t0, **st}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if tb:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and k not in ("iter",):
                    tb.add_scalar(f"grpo/{k}" if k in st or k in ("frames", "elapsed", "t_sample", "t_update") else f"grpo_{nm}/{k}", v, it)
        if it % a.ckpt_every == 0 or it == a.iters:
            save(it, "latest")
        if a.eval_every and (it % a.eval_every == 0 or it == a.iters):
            model.eval()
            res = evaluate(model, legal, episodes=a.eval_episodes, gif_dir=a.out / "gifs" / f"iter{it:05d}", device=dev, scenarios=names)
            res["iter"] = it
            (a.out / "eval.jsonl").open("a").write(json.dumps(res) + "\n")
            print("EVAL", json.dumps(res), flush=True)
            if tb:
                for k, v in res.items():
                    if isinstance(v, dict):
                        tb.add_scalar(f"eval/{k}", v["mean_return"], it)
            if a.s3:
                s3_sync(a.out, a.s3)
    save(a.iters, "final")
    if a.s3:
        s3_sync(a.out, a.s3)
    for es in envs.values():
        for e in es:
            e.close()


if __name__ == "__main__":
    main()
