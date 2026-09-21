"""Live evaluation: let FlyNet play ViZDoom scenarios, report per-scenario mean
episode return, and save footage of one (or every) episode per scenario.

Usage (standalone):
  python -m doomfly.evaluate --ckpt runs/flywire783/final --connectome data/processed/connectome_783.npz \
      --episodes 10 --gif-dir runs/flywire783/gifs/final
  # smooth 35 fps MP4 of the median episode (what the tutorial embeds):
  python -m doomfly.evaluate ... --gif-dir out --tics --pick median --fmt mp4

--pick   best (default; what train.py logs) | median | all (one file per episode)
--tics   record every game tic inside the frame skip instead of one frame per decision
--fmt    gif (default) | mp4 (needs the imageio-ffmpeg plugin: pip install doomfly[video])
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .doom.actions import SCENARIOS, N_ACTIONS
from .doom.env import DoomEnv


@torch.no_grad()
def play_episode(model, env: DoomEnv, sid: int, legal_row: torch.Tensor, device, greedy=False, record=False, max_steps=2100):
    obs, _ = env.reset()
    frames, R, done, n = [], 0.0, False, 0
    record = record and not env.record_tics  # per-tic footage is collected by the env itself
    local = np.asarray(env.sc.actions)
    g2l = {int(g): i for i, g in enumerate(local)}
    while not done and n < max_steps:
        if record:
            frames.append(env.render())
        fr = torch.from_numpy(obs[None]).to(device)
        sc = torch.tensor([sid], device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(fr, sc, legal_mask=legal_row[None])["policy"].float()[0]
        a = int(logits.argmax()) if greedy else int(torch.distributions.Categorical(logits=logits).sample())
        obs, r, done, _, _ = env.step(g2l[a])
        R += r; n += 1
    if env.record_tics:
        frames = list(env.tic_frames)
    return R, n, frames


def save_clip(path: Path, frames, fps: float, fmt: str = "gif"):
    """Write frames as GIF (any imageio) or MP4 (needs imageio-ffmpeg). Returns the path written."""
    path = path.with_suffix(f".{fmt}")
    if fmt == "gif":
        imageio.mimsave(path, frames, duration=1 / fps, loop=0)
    else:
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                               output_params=["-movflags", "+faststart"], macro_block_size=None)
        for f in frames:
            w.append_data(f)
        w.close()
    return path


def evaluate(model, legal, episodes=10, gif_dir: Path | None = None, device=torch.device("cpu"), scenarios=None, greedy=False,
             *, tics=False, pick="best", fmt="gif"):
    """pick: 'best' keeps the highest-return episode (train.py default), 'median' the middle one by
    return, 'all' writes every episode as <scenario>_ep<i>.<fmt>. tics=True records every game tic
    (35 fps) instead of one frame per decision (35/frame_skip fps)."""
    assert pick in ("best", "median", "all"), pick
    was_training = model.training
    model.eval()
    out = {}
    record = gif_dir is not None
    for name in scenarios or list(SCENARIOS):
        sc = SCENARIOS[name]
        env = DoomEnv(name, render=True, seed=12345, record_tics=tics and record)
        rets, lens, clips = [], [], []
        for ep in range(episodes):
            R, n, frames = play_episode(model, env, sc.id, legal[sc.id], device, greedy=greedy, record=record)
            rets.append(R); lens.append(n)
            if record:
                clips.append(frames)
        env.close()
        out[name] = {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
                     "max_return": float(np.max(rets)), "mean_len": float(np.mean(lens))}
        if record and any(clips):
            gif_dir.mkdir(parents=True, exist_ok=True)
            # 35 game tics/s; one frame per decision -> 35/frame_skip fps (8.75 at frame_skip 4)
            fps = 35.0 if tics else 35.0 / env.frame_skip
            order = np.argsort(rets)  # ascending by return
            if pick == "all":
                for i, fr in enumerate(clips):
                    save_clip(gif_dir / f"{name}_ep{i}", fr, fps, fmt)
                out[name]["returns"] = [float(r) for r in rets]
                out[name]["lens"] = [int(n) for n in lens]
            else:
                i = int(order[-1]) if pick == "best" else int(order[len(order) // 2])
                save_clip(gif_dir / name, clips[i], fps, fmt)
                out[name]["clip_episode"] = i
                out[name]["clip_return"] = float(rets[i])
    if was_training:
        model.train()
    return out


if __name__ == "__main__":
    from .surgery import load_flynet
    from .train import legal_mask_table

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--connectome", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--gif-dir", type=Path, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--tics", action="store_true", help="record every game tic (35 fps) instead of one frame per decision")
    ap.add_argument("--pick", choices=["best", "median", "all"], default="best")
    ap.add_argument("--fmt", choices=["gif", "mp4"], default="gif")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, _ = load_flynet(a.ckpt, a.connectome, dev)
    print(json.dumps(evaluate(model, legal_mask_table(dev), a.episodes, a.gif_dir, dev, scenarios=a.scenarios, greedy=a.greedy,
                            tics=a.tics, pick=a.pick, fmt=a.fmt), indent=1))
