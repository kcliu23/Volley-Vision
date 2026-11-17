"""Rally detection vs your marks: the serve (start) and "rally over" (end) you marked in tools.review.

    python -m evaluation.rallies_eval holdout1 holdout2 ... [--output output] [--labels]

For each serve you marked, the detected rally whose start is nearest: how far off the start is,
and how far its end (the ball's first floor contact) is before your "rally over" mark (the ball
lands, then you call it dead: the end should be at or a little before your mark).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths


def runs(frames: list[int]) -> list[int]:
    out = []
    for f in sorted(frames):
        if not out or f - out[-1][-1] > 1:
            out.append([f])
        else:
            out[-1].append(f)
    return [r[0] for r in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    sfx = "_labels" if a.labels else ""
    starts, ends, n_mark, n_found, extra = [], [], 0, 0, 0
    for s in a.stems:
        p = Path(a.output) / s / f"rallies{sfx}.json"
        if not p.exists():
            continue
        R = json.load(open(p))["rallies"]
        C = {int(k): v for k, v in (json.load(open(paths.verified(s))).get("contacts") or {}).items()}
        serves = runs([f for f, v in C.items() if v == "serve"])
        deads = runs([f for f, v in C.items() if v == "dead"])
        used = set()
        for sv in serves:
            n_mark += 1
            if not R:
                continue
            i = min(range(len(R)), key=lambda i: abs(R[i]["f0"] - sv))
            if abs(R[i]["f0"] - sv) > 60:
                print(f"  {s}: serve {sv}: no rally starts near it (nearest {R[i]['f0']})")
                continue
            n_found += 1
            used.add(i)
            starts.append(R[i]["f0"] - sv)
            d = [x for x in deads if x >= sv]
            if d:
                ends.append(R[i]["f1"] - d[0])
            print(f"  {s}: serve {sv} -> rally {R[i]['f0']}-{R[i]['f1']} ({R[i]['end']}), your rally over at "
                  f"{d[0] if d else '-'}; won by {({-1: 'near', 1: 'far'}).get(R[i]['winner'], '?')} ({R[i]['why']})")
        extra += len(R) - len(used)
    st, en = np.abs(starts), np.array(ends)
    print(f"serves marked {n_mark}, rallies found for {n_found}, rallies with no serve mark {extra}")
    if len(st):
        print(f"start: within 3 frames of your serve {np.mean(st <= 3):.0%} (median {np.median(st):.0f} frames off)")
    if len(en):
        print(f"end: before your rally-over mark by median {-np.median(en):.0f} frames; after it {np.mean(en > 3):.0%}")


if __name__ == "__main__":
    main()
