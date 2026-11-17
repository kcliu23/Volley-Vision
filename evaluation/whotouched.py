"""How often the pipeline names the right toucher: output/<clip>/touches.json (tools.ball3d_run)
against your who-touched-it (tools.review: touched_by + touched_box).

    python -m evaluation.whotouched holdout1 holdout2 ... [--output output]

Each touch you named is matched to the pipeline's touch within MATCH frames:
  right       the same number (or, for a player you didn't number, the same box: overlap > 0.5)
  wrong       another player
  no toucher  the pipeline has the touch but names nobody (no player at the ball)
  no touch    the pipeline has no touch there
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.common import paths

MATCH = 4


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def score(stem: str, output: str) -> dict:
    out = {"named": 0, "right": 0, "wrong": 0, "no toucher": 0, "no touch": 0}
    vp, tp = Path(paths.verified(stem)), Path(output) / stem / "touches.json"
    if not vp.exists() or not tp.exists():
        return out
    V = json.load(open(vp))
    who = {int(k): int(v) for k, v in (V.get("touched_by") or {}).items()}
    boxes = {int(k): v for k, v in (V.get("touched_box") or {}).items()}
    mine = json.load(open(tp))["touches"]
    for f, n in who.items():
        out["named"] += 1
        near = [t for t in mine if abs(t["frame"] - f) <= MATCH]
        if not near:
            out["no touch"] += 1
            continue
        t = min(near, key=lambda t: abs(t["frame"] - f))
        w = t["toucher"]
        if not w:
            out["no toucher"] += 1
        elif w["number"] == n or (f in boxes and w.get("box") and _iou(boxes[f], w["box"]) > 0.5):
            out["right"] += 1
        else:
            out["wrong"] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    keys = ["named", "right", "wrong", "no toucher", "no touch"]
    tot = dict.fromkeys(keys, 0)
    print(f"{'clip':12s} " + " ".join(f"{k:>11s}" for k in keys))
    for s in a.stems:
        r = score(s, a.output)
        if r["named"]:
            print(f"{s:12s} " + " ".join(f"{r[k]:11d}" for k in keys))
        for k in keys:
            tot[k] += r[k]
    print(f"{'ALL':12s} " + " ".join(f"{tot[k]:11d}" for k in keys)
          + (f"   right {tot['right'] / tot['named']:.0%} of the touches you named" if tot["named"] else
             "   (no who-touched-it labels yet: tools.review)"))


if __name__ == "__main__":
    main()
