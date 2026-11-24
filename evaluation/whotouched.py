"""How often the pipeline names the right toucher: output/<clip>/touches.json (tools.ball3d_run)
against your who-touched-it (tools.review: touched_by + touched_box).

    python -m evaluation.whotouched holdout1 holdout2 ... [--output output] [--player-ids]

Each touch you named (consecutive marked frames = one touch) is matched to the pipeline's touch
within MATCH frames of it:
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
from core.people.player_labels import _iou

MATCH = 4


def score(stem: str, output: str, sfx: str = "") -> dict:
    out = {"named": 0, "right": 0, "wrong": 0, "no toucher": 0, "no touch": 0}
    vp, tp = Path(paths.verified(stem)), Path(output) / stem / f"touches{sfx}.json"
    if not vp.exists() or not tp.exists():
        return out
    V = json.load(open(vp))
    who = {int(k): int(v) for k, v in (V.get("touched_by") or {}).items()}
    boxes = {int(k): v for k, v in (V.get("touched_box") or {}).items()}
    mine = json.load(open(tp))["touches"]
    runs = []                                          # one touch = consecutive frames you marked
    for f in sorted(who):
        if runs and f - runs[-1][-1] <= 1 and who[f] == who[runs[-1][-1]]:
            runs[-1].append(f)
        else:
            runs.append([f])
    for run in runs:
        f, n = run[len(run) // 2], who[run[0]]
        out["named"] += 1
        near = [t for t in mine if run[0] - MATCH <= t["frame"] <= run[-1] + MATCH]
        if not near:
            out["no touch"] += 1
            continue
        t = min(near, key=lambda t: abs(t["frame"] - f))
        w = t["toucher"]
        if not w:
            out["no toucher"] += 1
        elif w["number"] == n or any(g in boxes and w.get("box") and _iou(boxes[g], w["box"]) > 0.5 for g in run):
            out["right"] += 1
        else:
            out["wrong"] += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--player-ids", action="store_true", help="score touches_ids.json (tools.ball3d_run --player-ids)")
    a = ap.parse_args()
    keys = ["named", "right", "wrong", "no toucher", "no touch"]
    tot = dict.fromkeys(keys, 0)
    print(f"{'clip':12s} " + " ".join(f"{k:>11s}" for k in keys))
    for s in a.stems:
        r = score(s, a.output, "_ids" if a.player_ids else "")
        if r["named"]:
            print(f"{s:12s} " + " ".join(f"{r[k]:11d}" for k in keys))
        for k in keys:
            tot[k] += r[k]
    print(f"{'ALL':12s} " + " ".join(f"{tot[k]:11d}" for k in keys)
          + (f"   right {tot['right'] / tot['named']:.0%} of the touches you named" if tot["named"] else
             "   (no who-touched-it labels yet: tools.review)"))


if __name__ == "__main__":
    main()
