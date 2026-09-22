#!/usr/bin/env bash
# Publish the trained checkpoints and the connectome they need to Hugging Face.
#
#   hf auth login                 # once
#   scripts/publish_hf.sh         # -> https://huggingface.co/fabiononato/doomfly-rl
#
# Uploads checkpoints/<name>/{model.safetensors,config.json} for the three step-60k
# MaleCNS-49k students, data/processed/connectome_malecns49k.{npz,json}, and hf/README.md
# as the model card. Idempotent: re-running uploads only what changed.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=${HF_REPO:-fabiononato/doomfly-rl}
CKPTS="malecns49k_v2_final malecns49k_shuffled_s0 malecns49k_noconn"

for c in $CKPTS; do
  [[ -f checkpoints/$c/model.safetensors && -f checkpoints/$c/config.json ]] \
    || { echo "!! checkpoints/$c incomplete; pull it from S3 first"; exit 1; }
done
[[ -f data/processed/connectome_malecns49k.npz ]] || { echo "!! connectome npz missing"; exit 1; }

hf repo create "$REPO" --type model 2>/dev/null || true   # exists -> fine

hf upload "$REPO" hf/README.md README.md
hf upload "$REPO" docs/img/connectome_rotate.gif connectome_rotate.gif
hf upload "$REPO" data/processed/connectome_malecns49k.npz connectome_malecns49k.npz
hf upload "$REPO" data/processed/connectome_malecns49k.json connectome_malecns49k.json
for c in $CKPTS; do
  hf upload "$REPO" "checkpoints/$c" "$c"
done

echo "✓ https://huggingface.co/$REPO"
