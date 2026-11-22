#!/usr/bin/env bash
# Read-only: is this machine / sandbox fit to drive VolleyVision?   scripts/doctor.sh [<sandbox dir>]
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
ok=1
say() { printf '%-34s %s\n' "$1" "$2"; }
cd "$REPO"
py=$(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null) && say "python" "$py" || { say "python" "MISSING"; ok=0; }
for m in cv2 numpy sklearn; do
  python -c "import $m" 2>/dev/null && say "import $m" ok || { say "import $m" MISSING; ok=0; }
done
command -v ffprobe >/dev/null && say "ffprobe" ok || say "ffprobe" "missing (review can't seek: slow, still correct)"
say "git branch" "$(git branch --show-current) ($(git status --porcelain | grep -v '^??' | wc -l | tr -d ' ') modified files)"
if pgrep -f "tools.review" >/dev/null; then
  say "user's review running" "YES (pid $(pgrep -f tools.review | tr '\n' ' ')): never drive it, never write their data/"
else
  say "user's review running" "no"
fi
if [ -n "${1:-}" ]; then
  SB="$1"
  if [ "$(cat "$SB/.vv-repo" 2>/dev/null)" = "$REPO" ]; then say "sandbox" "$SB (ours)"; else say "sandbox" "NOT a sandbox.sh dir"; ok=0; fi
  real=$(cd "$SB/data/verified" 2>/dev/null && pwd -P)
  [ "$real" = "$(cd "$REPO/data/verified" && pwd -P)" ] && { say "sandbox labels" "POINT AT THE REAL ONES: stop"; ok=0; } || say "sandbox labels" "copies ($(ls "$SB/data/verified" 2>/dev/null | wc -l | tr -d ' ') files)"
fi
[ $ok = 1 ] && echo "doctor: OK" || { echo "doctor: NOT OK"; exit 1; }
