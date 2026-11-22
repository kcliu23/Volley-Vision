#!/usr/bin/env bash
# Make a throwaway VolleyVision workspace for one clip: the tools run there with the repo's code,
# read your videos / models through symlinks, and write only to COPIES of the clip's labels and
# outputs. Your real data/verified, data/player_ids and output/ are never touched.
#
#   scripts/sandbox.sh <clip> [<sandbox dir>]      prints the sandbox path on the last line
#
# Then:  cd <sandbox> && PYTHONPATH=<repo> python -m tools.review holdout/<clip>.mov ...
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
CLIP="${1:?usage: sandbox.sh <clip> [<sandbox dir>]}"
SB="${2:-${TMPDIR:-/tmp}/vv-verify-$CLIP-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$SB/data" "$SB/output"
echo "$REPO" > "$SB/.vv-repo"                              # marks it as ours (cleanup.sh checks this)
for d in videos holdout models training; do               # read-only inputs: symlinks
  [ -e "$REPO/$d" ] && ln -sfn "$REPO/$d" "$SB/$d"
done
for d in proposals verified cameras player_ids; do        # the clip's labels: copies
  mkdir -p "$SB/data/$d"
  [ -f "$REPO/data/$d/$CLIP.json" ] && cp "$REPO/data/$d/$CLIP.json" "$SB/data/$d/"
done
if [ -d "$REPO/output/$CLIP" ]; then                      # the clip's outputs: copies, no videos (big)
  rsync -a --exclude '*.mp4' "$REPO/output/$CLIP/" "$SB/output/$CLIP/"
fi
echo "$SB"
