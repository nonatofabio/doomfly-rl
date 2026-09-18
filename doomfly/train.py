"""Distil the Doom teachers into the fly connectome (analogue of chessfly's
supervised training on Lichess games).

Loss = KL(teacher policy || FlyNet policy, over legal actions)
     + CE(value bins, discounted return normalised per scenario)

Usage:
  python -m doomfly.train --connectome data/processed/connectome_783.npz \
      --rollouts data/rollouts --out runs/flywire783 --steps 60000 --batch 256 \
      [--s3 s3://bucket/runs/flywire783]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

from .doom.actions import SCENARIOS, N_ACTIONS, N_SCENARIOS
from .model.flynet import FlyNet, FlyNetConfig, load_connectome, value_to_bin
from .evaluate import evaluate


def legal_mask_table(device) -> torch.Tensor:
    m = torch.zeros(N_SCENARIOS, N_ACTIONS, dtype=torch.bool)
    for s in SCENARIOS.values():
        m[s.id, list(s.actions)] = True
    return m.to(device)


class Rollouts:
    """All shards in RAM as uint8; episode-level train/val split."""

    def __init__(self, root: Path, val_frac: float = 0.05, max_frames_per_scenario: int | None = None):
        frames, scen, teacher, ret, is_val = [], [], [], [], []
        self.norm = {}
        for d in sorted(p for p in root.iterdir() if (p / "meta.json").exists()):
            meta = json.loads((d / "meta.json").read_text())
            sid = SCENARIOS[meta["scenario"]].id
            self.norm[sid] = (meta["ret_lo"], meta["ret_hi"])
            n = 0
            for f in sorted(d.glob("shard_*.npz")):
                z = np.load(f)
                fr = z["frames"]
                if max_frames_per_scenario and n + len(fr) > max_frames_per_scenario:
                    keep = max_frames_per_scenario - n
                    if keep <= 0:
                        break
                    fr = fr[:keep]
                frames.append(fr); scen.append(z["scenario"][:len(fr)])
                teacher.append(z["teacher"][:len(fr)]); ret.append(z["ret"][:len(fr)])
                ep = z["episode"][:len(fr)]
                is_val.append(ep % round(1 / val_frac) == 0)
                n += len(fr)
            print(f"loaded {meta['scenario']}: {n} frames, teacher ep return {meta['ep_return_mean']:.1f}", flush=True)
        self.frames = np.concatenate(frames)
        self.scen = np.concatenate(scen).astype(np.int64)
        self.teacher = np.concatenate(teacher)
        ret = np.concatenate(ret)
        lo = np.array([self.norm[s][0] for s in self.scen], np.float32)
        hi = np.array([self.norm[s][1] for s in self.scen], np.float32)
        self.ret01 = np.clip((ret - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)
        is_val = np.concatenate(is_val)
        self.train_idx = np.flatnonzero(~is_val)
        self.val_idx = np.flatnonzero(is_val)
        print(f"dataset: {len(self.frames)} frames ({len(self.train_idx)} train / {len(self.val_idx)} val), "
              f"{self.frames.nbytes / 1e9:.1f} GB", flush=True)

    def batch(self, idx, device):
        fr = torch.from_numpy(self.frames[idx]).pin_memory().to(device, non_blocking=True)
        sc = torch.from_numpy(self.scen[idx]).to(device)
        te = torch.from_numpy(self.teacher[idx].astype(np.float32)).to(device)
        rt = torch.from_numpy(self.ret01[idx]).to(device)
        return fr, sc, te, rt


def losses(model, batch, legal):
    fr, sc, te, rt = batch
    mask = legal[sc]
    out = model(fr, sc, legal_mask=mask)
    logp = F.log_softmax(out["policy"].float(), -1)
    te = te * mask
    te = te / te.sum(-1, keepdim=True).clamp(min=1e-6)
    pol = -(te * logp).sum(-1).mean()                      # CE with soft teacher targets (== KL + const)
    val = F.cross_entropy(out["value"].float(), value_to_bin(rt))
    with torch.no_grad():
        acc = (logp.argmax(-1) == te.argmax(-1)).float().mean()
    return pol, val, acc


def s3_sync(local: Path, uri: str):
    subprocess.run(["aws", "s3", "sync", str(local), uri, "--only-show-errors"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tb", type=Path, default=None, help="TensorBoard log dir (optional)")
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-conn", type=float, default=3e-3, help="lr for the per-synapse log-gains")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--max-frames-per-scenario", type=int, default=None)
    ap.add_argument("--s3", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-dyn", type=int, default=5)
    ap.add_argument("--resume", type=Path, default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    data = Rollouts(a.rollouts, max_frames_per_scenario=a.max_frames_per_scenario)
    conn = load_connectome(a.connectome)
    cfg = FlyNetConfig(steps=a.steps_dyn)
    model = FlyNet(conn, cfg).to(dev)
    legal = legal_mask_table(dev)
    meta = {"args": {k: str(v) for k, v in vars(a).items()}, "model": model.meta(),
            "connectome": json.loads(Path(str(a.connectome).replace(".npz", ".json")).read_text())
            if Path(str(a.connectome).replace(".npz", ".json")).exists() else {},
            "ret_norm": {int(k): v for k, v in data.norm.items()}}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta["model"]), flush=True)

    conn_p, homeo_p, rest_p = model.param_groups()
    opt = torch.optim.AdamW([
        {"params": conn_p, "lr": a.lr_conn, "weight_decay": 0.0},
        {"params": homeo_p, "lr": a.lr, "weight_decay": 0.0},
        {"params": rest_p, "lr": a.lr, "weight_decay": a.wd},
    ], betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in opt.param_groups]

    def lr_at(step):
        if step < a.warmup:
            return step / a.warmup
        p = (step - a.warmup) / max(1, a.steps - a.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    step0 = 0
    if a.resume:
        model.load_state_dict(load_file(a.resume / "model.safetensors"), strict=True)
        st = torch.load(a.resume / "opt.pt", map_location=dev)
        opt.load_state_dict(st["opt"]); step0 = st["step"]
        print(f"resumed from step {step0}", flush=True)

    log = open(a.out / "metrics.jsonl", "a")
    tb = None
    if a.tb:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(a.tb))
    rng = np.random.default_rng(a.seed)
    use_bf16 = dev.type == "cuda"
    t0, tl = time.time(), time.time()
    ema = None
    eval_thread = None

    def save(step, tag):
        d = a.out / tag
        d.mkdir(exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, d / "model.safetensors")
        torch.save({"opt": opt.state_dict(), "step": step}, d / "opt.pt")
        (d / "config.json").write_text(json.dumps({"flynet": asdict(cfg), "connectome": str(a.connectome.name), "step": step}))

    def run_eval(step):
        res = evaluate(model, legal, episodes=a.eval_episodes, gif_dir=a.out / "gifs" / f"step{step:07d}", device=dev)
        res["step"] = step
        (a.out / "eval.jsonl").open("a").write(json.dumps(res) + "\n")
        print("EVAL", json.dumps(res), flush=True)
        if tb is not None:
            for k, v in res.items():
                if isinstance(v, (int, float)) and k != "step":
                    tb.add_scalar(f"eval/{k}", v, step)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            tb.add_scalar(f"eval_{k}/{kk}", vv, step)
            tb.flush()
        if a.s3:
            s3_sync(a.out, a.s3)

    model.train()
    for step in range(step0, a.steps + 1):
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_at(step)
        idx = np.sort(rng.choice(data.train_idx, a.batch, replace=False))
        batch = data.batch(idx, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            pol, val, acc = losses(model, batch, legal)
            loss = pol + a.value_coef * val
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        cur = {"pol": pol.item(), "val": val.item(), "acc": acc.item()}
        ema = cur if ema is None else {k: 0.98 * ema[k] + 0.02 * cur[k] for k in cur}
        if step % 50 == 0:
            row = {"step": step, **{k: round(v, 4) for k, v in ema.items()}, "gn": round(gn.item(), 3),
                   "lr": opt.param_groups[2]["lr"], "sps": round(50 * a.batch / (time.time() - tl)), "wall": round(time.time() - t0)}
            tl = time.time()
            if step % 500 == 0:
                model.eval()
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    vi = np.sort(rng.choice(data.val_idx, min(2048, len(data.val_idx)), replace=False))
                    vp, vv, va = [], [], []
                    for j in range(0, len(vi), a.batch):
                        p_, v_, a_ = losses(model, data.batch(vi[j:j + a.batch], dev), legal)
                        vp.append(p_.item()); vv.append(v_.item()); va.append(a_.item())
                    row.update(val_pol=round(np.mean(vp), 4), val_val=round(np.mean(vv), 4), val_acc=round(np.mean(va), 4))
                model.train()
            log.write(json.dumps(row) + "\n"); log.flush()
            print(json.dumps(row), flush=True)
            if tb is not None:
                for k, v in row.items():
                    if k not in ("step", "wall"):
                        tb.add_scalar(f"{'val' if k.startswith('val_') else 'train'}/{k}", v, step)
                tb.add_scalar("train/wall_hours", row["wall"] / 3600, step)
        if step > 0 and step % a.ckpt_every == 0:
            save(step, "latest")
        if step > 0 and step % a.eval_every == 0 or step == a.steps:
            model.eval()
            run_eval(step)
            model.train()
    save(a.steps, "final")
    if a.s3:
        s3_sync(a.out, a.s3)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
