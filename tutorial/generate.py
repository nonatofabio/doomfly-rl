"""Build the infographic tutorial from the real connectome files.

    python tutorial/generate.py            # -> tutorial/index.html (+ tutorial/assets/stats.json)

Everything in the page is computed here from data/processed/connectome_*.npz; nothing is hand-typed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from doomfly.doom.actions import ACTIONS, SCENARIOS  # noqa: E402
from doomfly.doom.env import FRAME_H, FRAME_STACK, FRAME_W  # noqa: E402
from doomfly.doom.record import GAMMA  # noqa: E402
from doomfly.model.flynet import FlyNet, FlyNetConfig, load_connectome  # noqa: E402

BACKBONES = {
    "flywire783": {"file": "connectome_783", "label": "FlyWire FAFB v783 (chessfly-faithful)", "short": "FlyWire 783"},
    "malecns49k": {"file": "connectome_malecns49k", "label": "MaleCNS 49k (HF fernandofernandes/fly-connectome-49k)", "short": "MaleCNS 49k"},
}
RNG = np.random.default_rng(0)

# Side-by-side footage (section 10): the two checkpoints, rendered by scripts/compare_clips.sh from the
# eval_all.json of `python -m doomfly.evaluate --episodes 5 --pick all`. Numbers on the page come from those files.
COMPARE = {
    "backbone": "malecns49k", "slot": "malecns49k_student_vs_grpo",
    "left": {"label": "distilled student (GRPO iter 0)", "dir": "assets/videos/malecns49k_student_iter0",
             "ckpt": "l40s-v2/runs/malecns49k/final"},
    "right": {"label": "GRPO <code>base</code>, iter 300", "dir": "assets/videos/malecns49k_grpo_base_iter300",
              "ckpt": "grpo-use2/runs/malecns49k_grpo_base/final"},
}


def compare_stats():
    """Per-scenario mean/std/returns for both sides, or None for a side whose eval has not finished."""
    out = dict(COMPARE)
    for side in ("left", "right"):
        f = ROOT / COMPARE[side]["dir"] / "eval_all.json"
        d = json.loads(f.read_text()) if f.exists() and f.stat().st_size else None  # empty while eval runs
        out[side] = dict(COMPARE[side], eval=d and {k: {"mean": v["mean_return"], "std": v["std_return"],
                                                          "returns": v.get("returns", []), "lens": v.get("lens", [])}
                                                      for k, v in d.items()})
    return out


def log_hist(x, nb=24):
    x = x[x > 0]
    edges = np.geomspace(1, max(2, x.max()), nb + 1)
    c, _ = np.histogram(x, bins=edges)
    return {"edges": [float(e) for e in edges], "counts": [int(v) for v in c]}


def project(pos, group, is_input, is_readout, k=7000):
    ok = np.isfinite(pos).all(1) & (np.abs(pos).sum(1) > 0)
    idx = np.flatnonzero(ok)
    idx = RNG.choice(idx, size=min(k, len(idx)), replace=False)
    p = pos[idx]
    # PCA to 2-D so both brains land in a comparable frame
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    q = p @ vt[:2].T
    q = (q - q.min(0)) / (q.max(0) - q.min(0) + 1e-9)
    role = np.where(is_input[idx], 1, np.where(is_readout[idx], 2, 0))
    return {"x": [round(float(v), 4) for v in q[:, 0]], "y": [round(float(v), 4) for v in q[:, 1]],
            "role": [int(v) for v in role]}


def summarise(key):
    spec = BACKBONES[key]
    meta = json.loads((ROOT / "data/processed" / f"{spec['file']}.json").read_text())
    z = np.load(ROOT / "data/processed" / f"{spec['file']}.npz", allow_pickle=True)
    n = int(z["n"]); src = z["src"]; dst = z["dst"]; sign = z["sign"]; syn = z["syn_count"].astype(np.float64)
    outdeg = np.bincount(src, minlength=n); indeg = np.bincount(dst, minlength=n)
    model = FlyNet(load_connectome(ROOT / "data/processed" / f"{spec['file']}.npz"), FlyNetConfig())
    m = model.meta()
    groups = {}
    for p, name in zip(model.param_groups(), ["synaptic log-gains", "homeostatic scale/shift", "stem + decoder + heads"]):
        groups[name] = int(sum(t.numel() for t in p))
    sc, cnt = np.unique(z["super_class"], return_counts=True)
    order = np.argsort(-cnt)
    return {
        "key": key, "label": spec["label"], "short": spec["short"],
        "neurons": n, "edges": int(len(src)), "synapses": float(syn.sum()),
        "inputs": int(z["is_input"].sum()), "readout": int(z["is_readout"].sum()),
        "excitatory_fraction": float((sign > 0).mean()),
        "mean_out_degree": float(outdeg.mean()), "median_out_degree": float(np.median(outdeg)),
        "max_out_degree": int(outdeg.max()), "max_in_degree": int(indeg.max()),
        "density": float(len(src)) / (float(n) * float(n)),
        "isolated": int(((outdeg + indeg) == 0).sum()),
        "outdeg_hist": log_hist(outdeg), "syn_hist": log_hist(syn),
        "super_class": [[str(sc[i]), int(cnt[i])] for i in order],
        "params": m["params"], "param_groups": groups,
        "map": project(z["pos"], z["group"], z["is_input"], z["is_readout"]),
        "source_meta": {k: meta[k] for k in meta if k in ("dataset", "source", "license", "hf_revision", "references")},
    }


def main():
    stats = {
        "backbones": [summarise(k) for k in BACKBONES],
        "actions": list(ACTIONS),
        "scenarios": {k: {"n_legal": len(v.actions), "buttons": list(v.buttons), "frame_skip": v.frame_skip,
                          "ppo_steps": v.ppo_steps, "reward_scale": v.reward_scale, "shaping": v.shaping,
                          "doom_skill": v.doom_skill if v.doom_skill is not None else 5} for k, v in SCENARIOS.items()},
        "compare": compare_stats(),
        "model_config": FlyNetConfig().__dict__,
        "frame": {"h": FRAME_H, "w": FRAME_W, "stack": FRAME_STACK, "bytes": FRAME_H * FRAME_W * FRAME_STACK},
        "rollout": {"frames_per_scenario": 300_000, "eps": 0.1, "gamma": GAMMA, "shard_frames": 20_000,
                    "schema": [["frames", "u8", "[T,4,72,96]", "4-frame stack, grayscale, area-averaged from 320x240"],
                               ["scenario", "i8", "[T]", "scenario id (0-4), selects legal mask + value normalisation"],
                               ["action", "i8", "[T]", "action actually taken (teacher argmax, or random w.p. eps)"],
                               ["teacher", "f16", "[T,22]", "teacher's full softmax over the 22-action vocabulary (soft target)"],
                               ["ret", "f32", "[T]", "discounted return-to-go from this state, gamma 0.99"],
                               ["episode", "i32", "[T]", "episode id, so shards can be split without leaking episodes"]]},
    }
    assets = ROOT / "tutorial/assets"; assets.mkdir(exist_ok=True)
    (assets / "stats.json").write_text(json.dumps(stats))
    tpl = (ROOT / "tutorial/template.html").read_text()
    html = tpl.replace("/*__DATA__*/", "const DATA = " + json.dumps(stats) + ";")
    (ROOT / "tutorial/index.html").write_text(html)
    print("wrote tutorial/index.html", len(html) // 1024, "KiB")
    for b in stats["backbones"]:
        print(f"  {b['short']}: {b['neurons']:,} neurons, {b['edges']:,} edges, {b['params']/1e6:.2f}M params, "
              f"{b['inputs']:,} inputs, {b['readout']:,} readout, E-frac {b['excitatory_fraction']:.3f}")


if __name__ == "__main__":
    main()
