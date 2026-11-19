"""Are the 3D fit's touches real? Its touches against your touch marks (tools.review, m + type).

    python -m evaluation.touches3d holdout1 holdout2 ...                 # output/<clip>/ball3d.json
    python -m evaluation.touches3d holdout1 ... --output outA --output outB --name now --name timing

The 3D fit splits a rally into flights; every join between two flights is a touch it chose.
Your marks say where the real touches are. A touch you marked usually spans a few frames (the ball
is on the hands for a while: 208-212 = one set), so consecutive marks are one touch.

Counted only while the ball is in play (from a serve to the rally's end, evaluation.inplay); the
serve itself is left out (a rally's first flight starts there, it isn't a join), and so is the
end of play. A join with the ball at the floor (below FLOOR_M) is a bounce, not a touch, and is
neither found nor invented.

    found     your touch has a 3D touch within TOL frames of it
    missed    your touch has none (the fit flew straight through it)
    invented  a 3D touch with no touch of yours within TOL frames (a fake touch: the fit bought a
              better match to the picture with a touch that didn't happen)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths
from evaluation.inplay import in_play

TOL = 3                  # frames either side of your marked touch
FLOOR_M = 0.35           # a join this low is a bounce on the floor


def your_touches(V: dict) -> list[tuple[int, int, str]]:
    """Your marks as touches: runs of consecutive frames with a mark, (first, last, type);
    serves, rally ends and touches you didn't see (v) left out."""
    c = sorted((int(k), v) for k, v in (V.get("contacts") or {}).items() if v not in ("dead",))
    out = []
    for f, v in c:
        if out and f - out[-1][1] <= 1:
            out[-1] = (out[-1][0], f, out[-1][2])
        else:
            out.append((f, f, v))
    unseen = set(V.get("unseen") or [])                 # marked not seen (tools.review v): frame is a guess
    return [t for t in out if t[2] != "serve" and not any(g in unseen for g in range(t[0], t[1] + 1))]


def fit_touches(B: dict) -> list[tuple[int, float]]:
    """The 3D fit's touches: (frame, ball height m) at each join between two flights of a rally."""
    fr = {int(k): v["xyz"] for k, v in B["frames"].items()}
    out = []
    for a, b in zip(B["flights"][:-1], B["flights"][1:]):
        if b["f0"] - a["f1"] > 10:                       # a new rally
            continue
        f = (a["f1"] + b["f0"]) / 2
        z = [fr[g][2] for g in (a["f1"], b["f0"]) if g in fr]
        out.append((f, float(np.mean(z)) if z else np.nan))
    return out


def score(stem: str, output: str) -> dict:
    V = json.load(open(paths.verified(stem)))
    play = in_play(V) or set()
    mine = [t for t in your_touches(V) if t[0] in play or t[1] in play]
    B = json.load(open(Path(output) / stem / "ball3d.json"))
    fits = [(f, z) for f, z in fit_touches(B) if round(f) in play and not (np.isfinite(z) and z < FLOOR_M)]
    found = [t for t in mine if any(t[0] - TOL <= f <= t[1] + TOL for f, _ in fits)]
    invented = [f for f, _ in fits if not any(t[0] - TOL <= f <= t[1] + TOL for t in mine)]
    err = []
    for t in found:
        f = min((f for f, _ in fits if t[0] - TOL <= f <= t[1] + TOL), key=lambda f: abs(f - (t[0] + t[1]) / 2))
        err.append(0.0 if t[0] <= f <= t[1] else min(abs(f - t[0]), abs(f - t[1])))
    return {"marks": len(mine), "found": len(found), "missed": [t for t in mine if t not in found],
            "invented": invented, "outside_mark_frames": float(np.mean(err)) if err else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", action="append", help="one per version to compare (default: output/)")
    ap.add_argument("--name", action="append", help="a short name per --output")
    ap.add_argument("--detail", action="store_true", help="list missed and invented touches per clip")
    a = ap.parse_args()
    outs = a.output or [str(paths.OUTPUT)]
    names = a.name or outs
    print(f"{'version':12s} {'clip':10s} {'touches':>7s} {'found':>6s} {'missed':>6s} {'invented':>8s}")
    for out, name in zip(outs, names):
        T = {"marks": 0, "found": 0, "missed": 0, "invented": 0}
        for s in a.stems:
            if not (Path(out) / s / "ball3d.json").exists():
                print(f"{name:12s} {s:10s}  (no ball3d.json)")
                continue
            r = score(s, out)
            T["marks"] += r["marks"]; T["found"] += r["found"]
            T["missed"] += len(r["missed"]); T["invented"] += len(r["invented"])
            print(f"{name:12s} {s:10s} {r['marks']:7d} {r['found']:6d} {len(r['missed']):6d} {len(r['invented']):8d}")
            if a.detail:
                print(f"{'':24s} missed {[(t[0], t[2]) for t in r['missed']]}  invented {[round(f) for f in r['invented']]}")
        rec = T["found"] / max(T["marks"], 1)
        print(f"{name:12s} {'ALL':10s} {T['marks']:7d} {T['found']:6d} {T['missed']:6d} {T['invented']:8d}"
              f"   found {rec:.0%}, invented per 10 real touches {10 * T['invented'] / max(T['marks'], 1):.1f}")
        print()


if __name__ == "__main__":
    main()
