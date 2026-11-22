#!/usr/bin/env bash
# Remove a sandbox made by sandbox.sh (only if it is one). Evidence lives elsewhere and stays.
#   scripts/cleanup.sh <sandbox dir>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
SB="${1:?usage: cleanup.sh <sandbox dir>}"
[ "$(cat "$SB/.vv-repo" 2>/dev/null)" = "$REPO" ] || { echo "not a sandbox.sh dir: $SB (left alone)"; exit 1; }
rm -rf "$SB"
echo "removed $SB"
