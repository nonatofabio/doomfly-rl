# tutorial video slots

`<backbone>_<scenario>.mp4` (960x720, 35 fps) and `doomfly_<backbone>_reel_web.mp4` are read by `../../index.html`.
Not tracked in git (too large); regenerate with

    python -m doomfly.evaluate --ckpt <ckpt> --connectome data/processed/<npz> --episodes 10 \
        --gif-dir assets/videos/<tag>/episodes --tics --pick all --fmt mp4 --device cpu \
        > assets/videos/<tag>/eval_all.json
    REEL_SLOT=1 scripts/make_clips.sh assets/videos/<tag> <backbone>
    scripts/publish_site.sh sync

Status: malecns49k = v2 final (step 60k). flywire783 = interim v2 step 20k (best-of-10 GIF via
scripts/pull_videos.sh); redo from `l40s-v2/runs/flywire783/final` when that run finishes.
