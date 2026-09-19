#!/usr/bin/env bash
# Publish the tutorial + videos + a source tarball to a dev desktop and expose it through a Midway-gated tunnel.
#
#   scripts/publish_site.sh                 # sync + (re)start server and tunnel
#   scripts/publish_site.sh sync            # only rsync files
#   scripts/publish_site.sh status          # tmux + tunnel status
#   scripts/publish_site.sh stop
#
# Knobs: HOST (dev desktop), PORT (local port on the box), NAME (tunnel suffix), ALLOW (tunnel --allow value).
# The tunnel lives while `tunnel create` runs AND the box has a valid Midway cookie (~12-23h): re-run `mwinit`
# on the box, then `scripts/publish_site.sh` again, when the URL starts failing.
set -euo pipefail
HOST=${HOST:-dev-dsk-fnp-2b-0822a465.us-west-2.amazon.com}
PORT=${PORT:-8090}
NAME=${NAME:-doomfly}
# default audience: aws-velocity-labs POSIX group + members of the #stallion-stan Slack channel (C0BMSC4D3TM, 2026-09-18)
ALLOW=${ALLOW:-posix:aws-velocity-labs,opieter,murmeral,rycolez,pgrayy,lizrad,fjonatse,gauravaz,gsird,eshkuma,arielnab,zhalbert,okapl,tynoble,maxrat,traklord,tmoreton,alrichey,willismt,jonabuck,ncclegg,akhtrma,vivdalal,arron,maczas}
REPO=/home/fnp/wd/doomfly
SITE=/home/fnp/doomfly-site
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cmd=${1:-all}

sync() {
  ssh "$HOST" "mkdir -p $REPO"
  rsync -az --delete --stats \
    --exclude .venv --exclude data --exclude runs --exclude checkpoints --exclude .agent --exclude __pycache__ \
    --exclude .DS_Store --exclude 'assets/videos/*/gif' --exclude 'assets/videos/*/episodes' \
    "$ROOT/" "$HOST:$REPO/"
  ssh "$HOST" bash -s "$REPO" "$SITE" <<'EOF'
set -e
REPO=$1; SITE=$2
mkdir -p "$SITE"
ln -sfn "$REPO/tutorial/index.html" "$SITE/index.html"
ln -sfn "$REPO/tutorial/assets" "$SITE/assets"
ln -sfn "$REPO/assets/videos" "$SITE/videos"
tar -C "$(dirname "$REPO")" -czf "$SITE/doomfly-src.tar.gz" \
  --exclude .git --exclude 'assets/videos' --exclude 'tutorial/assets/videos' doomfly
ls -la "$SITE"
EOF
}

start() {
  ssh "$HOST" bash -s "$REPO" "$SITE" "$PORT" "$NAME" "$ALLOW" <<'EOF'
set -e
REPO=$1; SITE=$2; PORT=$3; NAME=$4; ALLOW=$5
export PATH=$HOME/.toolbox/bin:$PATH
tmux kill-session -t doomfly-site 2>/dev/null || true
tmux new-session -d -s doomfly-site -n serve \
  "python3 $REPO/scripts/serve_site.py $SITE $PORT 2>&1 | tee -a $HOME/doomfly-site.serve.log"
sleep 1
tmux new-window -t doomfly-site -n tunnel \
  "tunnel create $PORT --name $NAME --allow $ALLOW 2>&1 | tee -a $HOME/doomfly-site.tunnel.log"
sleep 6
echo "--- serve:";  tail -3 $HOME/doomfly-site.serve.log
echo "--- tunnel:"; tail -15 $HOME/doomfly-site.tunnel.log
EOF
}

status() {
  ssh "$HOST" 'export PATH=$HOME/.toolbox/bin:$PATH; tmux ls 2>&1; tunnel list 2>&1 | head; echo; tail -5 ~/doomfly-site.tunnel.log 2>/dev/null'
}

case "$cmd" in
  sync) sync ;;
  start) start ;;
  status) status ;;
  stop) ssh "$HOST" 'tmux kill-session -t doomfly-site 2>/dev/null; echo stopped' ;;
  all) sync; start ;;
  *) echo "usage: $0 [sync|start|status|stop]" >&2; exit 2 ;;
esac
