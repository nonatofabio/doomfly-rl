"""Tables for the free-play GRPO write-up (docs/freeplay.md, "GRPO on MAP01").

Held-out evals: docs/results/freeplay_heldout_<backbone>_<init|final>.json, all on seed stream 1
(the same 30 seeds), so episode i of every file starts from the same engine seed and diffs are
paired by seed. Pairing removes map-RNG variance, NOT training-run variance: each backbone has one
GRPO run (training seed 0), so a cross-backbone gap is "this run vs that run".

Curves: eval.jsonl / metrics.jsonl pulled from s3://<bucket>/freeplay-grpo-<backbone>/runs/<run>/.

    python scripts/freeplay_compare.py [--runs runs]
"""
import argparse, json
from pathlib import Path
import numpy as np

BACKBONES = {"connectome": ("conn", "freeplay-grpo-conn/malecns49k_grpo_freeplay"),
             "shuffled s0": ("shuffled_s0", "freeplay-grpo-shuffled-s0/malecns49k_shuffled_s0_grpo_freeplay"),
             "no-connectome": ("noconn", "freeplay-grpo-noconn/malecns49k_noconn_grpo_freeplay")}
KEYS = ["R", "kill", "item", "damage", "health", "explore", "T", "dead"]  # health = HP lost (shaper total)
LABEL = {k: ("hp lost" if k == "health" else k) for k in KEYS}
B, rng = 20000, np.random.default_rng(0)


def heldout(tag):
    d = json.loads(Path(f"docs/results/freeplay_heldout_{tag}.json").read_text())
    assert d["seed_stream"] == 1 and d["max_steps"] == 1024 and d["temperature"] == 1.0
    by = {e["seed"]: e for e in d["episodes"]}
    return d, {k: np.array([float(by[s][k]) for s in d["seeds"]]) for k in KEYS}


def stats(d):
    """mean diff, bootstrap 95% CI, two-sided sign-flip permutation p (paired)."""
    lo, hi = np.percentile(d[rng.integers(0, len(d), (B, len(d)))].mean(1), [2.5, 97.5])
    null = np.abs((d * rng.choice([-1, 1], (B, len(d)))).mean(1))
    return d.mean(), lo, hi, (1 + (null >= abs(d.mean()) - 1e-12).sum()) / (B + 1)


def fmt(m, lo, hi, p):
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}], p={p:.2g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    a = ap.parse_args()

    H = {b: {st: heldout(f"{t}_{st}") for st in ("init", "final")} for b, (t, _) in BACKBONES.items()}
    seeds = {tuple(H[b][st][0]["seeds"]) for b in H for st in H[b]}
    assert len(seeds) == 1, "held-out files do not share seeds"
    print(f"held-out: n={len(next(iter(seeds)))} seeds, stream 1\n")

    print("| backbone | stage | " + " | ".join(LABEL[k] for k in KEYS) + " |\n|---|---|" + "---|" * len(KEYS))
    for b in H:
        for st in ("init", "final"):
            v = H[b][st][1]
            print(f"| {b} | {st} | " + " | ".join(f"{v[k].mean():.2f}±{v[k].std():.2f}" for k in KEYS) + " |")
    print("\nexit/secret any:", {f"{b}/{st}": any(e["exit"] or e["secret"] for e in H[b][st][0]["episodes"])
                                for b in H for st in H[b]})

    print("\n## final - init, paired by seed\n| metric | " + " | ".join(H) + " |\n|---|" + "---|" * len(H))
    for k in KEYS:
        print(f"| {LABEL[k]} | " + " | ".join(fmt(*stats(H[b]["final"][1][k] - H[b]["init"][1][k])) for b in H) + " |")

    pairs = [("no-connectome", "connectome"), ("no-connectome", "shuffled s0"), ("shuffled s0", "connectome")]
    print("\n## between backbones, paired by seed\n| metric | " + " | ".join(f"{x} - {y}" for x, y in pairs)
          + " |\n|---|" + "---|" * len(pairs))
    for k in KEYS:
        print(f"| final {LABEL[k]} | " + " | ".join(fmt(*stats(H[x]["final"][1][k] - H[y]["final"][1][k])) for x, y in pairs) + " |")
    for lab, f in [("init R", lambda b: H[b]["init"][1]["R"]),
                   ("gain R", lambda b: H[b]["final"][1]["R"] - H[b]["init"][1]["R"])]:
        print(f"| {lab} | " + " | ".join(fmt(*stats(f(x) - f(y))) for x, y in pairs) + " |")

    # The connectome run's start point is l40s-v2/runs/malecns49k/final (run_grpo.sh STUDENT default),
    # NOT checkpoints/malecns49k_v2_final (g6-12xl-use2-v2, the pi_ref of the zero-shot tables).
    g6 = Path("docs/results/freeplay_heldout_conn_g6v2student_init.json")
    if g6.exists():
        _, v6 = heldout("conn_g6v2student_init")
        v4 = H["connectome"]["init"][1]
        print("\n## connectome students at init: l40s-v2 (GRPO start) - g6-12xl-use2-v2 (zero-shot pi_ref)")
        print("| metric | g6 v2 | l40s v2 | diff |\n|---|---|---|---|")
        for k in KEYS:
            print(f"| {LABEL[k]} | {v6[k].mean():.2f} | {v4[k].mean():.2f} | {fmt(*stats(v4[k] - v6[k]))} |")
        # the start-point gap docs/freeplay.md first reported, against the zero-shot pi_ref
        print("\n| init R, minus the g6 v2 student | shuffled s0 | no-connectome |\n|---|---|---|")
        print("| diff | " + " | ".join(
            fmt(*stats(H[b]['init'][1]['R'] - v6['R'])) for b in ("shuffled s0", "no-connectome")) + " |")

    print("\n## in-run eval R (10 episodes, seeds default_rng(0), every 25 iters)")
    for b, (_, run) in BACKBONES.items():
        ev = [json.loads(l) for l in (a.runs / run / "eval.jsonl").read_text().splitlines()]
        r = [e["R"]["mean"] for e in ev]
        last = np.concatenate([[x] for x in r[-4:]])
        auc = float(np.mean(r))
        f90 = next(e["iter"] for e, x in zip(ev, r) if x >= 0.9 * last.mean())
        print(f"{b}: iter0 {r[0]:.2f}, last-4 mean {last.mean():.2f}, mean over 21 evals {auc:.2f}, "
              f"first eval >= 90% of last-4 at iter {f90}\n  " + " ".join(f"{x:.1f}" for x in r))

    print("\n## training blocks (100 iters = 3200 episodes, 512-decision cap)")
    print("| backbone | iters | R | ep_len | dead | ent | kill | item | damage | hp lost | explore |\n|---|---|" + "---|" * 9)
    for b, (_, run) in BACKBONES.items():
        m = [json.loads(l) for l in (a.runs / run / "metrics.jsonl").read_text().splitlines()]
        for i in range(0, 500, 100):
            blk = m[i:i + 100]
            mu = lambda k: np.mean([x[k] for x in blk])
            print(f"| {b} | {i + 1}-{i + 100} | {mu('return_mean'):.2f} | {mu('ep_len'):.0f} | {mu('dead'):.2f} | "
                  f"{mu('ent'):.2f} | {mu('c_kill'):.2f} | {mu('c_item'):.2f} | {mu('c_damage'):.1f} | "
                  f"{mu('c_health'):.1f} | {mu('c_explore'):.1f} |")
        print(f"| {b} | wall | {m[-1]['elapsed'] / 3600:.2f} h, {m[-1]['frames']:,} frames, "
              f"t_sample {np.mean([x['t_sample'] for x in m]):.1f} s, t_update {np.mean([x['t_update'] for x in m]):.1f} s |"
              + " |" * 8)
        print(f"| {b} | max | exit {max(x['c_exit'] for x in m):.0f}, secret {max(x['c_secret'] for x in m):.0f}, "
              f"item {max(x['c_item'] for x in m):.2f} |" + " |" * 8)


if __name__ == "__main__":
    main()
