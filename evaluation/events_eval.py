"""Touch types in events.json (core.events: from the order inside the rally) vs the types you
marked in tools.review.

    python -m evaluation.events_eval holdout1 holdout2 ... [--output output]

Each of your touches is matched to the event within TOL frames; your "pass" counts as right when
it's a reception or a dig (your key 2 is pass/receive), "hit" as attack.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core.common import paths
from core.events import build
from evaluation.reach3d import touches_all

TOL = 3
SAME = {"serve": {"serve"}, "pass": {"reception", "dig"}, "dig": {"dig", "reception"}, "set": {"set"},
        "hit": {"attack"}, "block": {"block"}}


def z_model(stems: list[str], output: str) -> dict:
    """Ball height per 1st / 2nd / 3rd touch (mean, sd) from your marked types on these clips."""
    import numpy as np
    zs = {"P1": [], "P2": [], "P3": []}
    for s in stems:
        T = json.load(open(Path(output) / s / "touches.json"))["touches"]
        for t0, t1, k in touches_all(json.load(open(paths.verified(s)))):
            near = [t for t in T if t0 - TOL <= t["frame"] <= t1 + TOL and t.get("xyz")]
            key = {"pass": "P1", "dig": "P1", "set": "P2", "hit": "P3"}.get(k)
            if near and key:
                zs[key].append(near[0]["xyz"][2])
    return {k: (float(np.mean(v)), max(0.15, float(np.std(v)))) for k, v in zs.items() if len(v) >= 3}


def compare(stem: str, output: str, zm: dict | None = None):
    d = Path(output) / stem
    T = json.load(open(d / "touches.json"))["touches"]
    R = json.load(open(d / "rallies.json"))["rallies"]
    E = build(T, R, stem, z_model=zm)["events"]
    V = json.load(open(paths.verified(stem)))
    unseen = set(V.get("unseen") or [])
    rows = []
    for t0, t1, kind in touches_all(V):
        if kind not in SAME or any(g in unseen for g in range(t0, t1 + 1)):
            continue
        near = [e for e in E if t0 - TOL <= e["frame"] <= t1 + TOL]
        e = min(near, key=lambda e: abs(e["frame"] - (t0 + t1) / 2)) if near else None
        rows.append((t0, kind, e["type"] if e else None))
    return rows, E


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--folds", action="store_true", help="heights fit on the other half of the clips (honest)")
    a = ap.parse_args()
    conf, n, ok, unmatched = Counter(), 0, 0, 0
    half = len(a.stems) // 2
    for i, s in enumerate(a.stems):
        zm = None
        if a.folds:
            other = a.stems[half:] if i < half else a.stems[:half]
            zm = z_model(other, a.output)
        rows, _ = compare(s, a.output, zm)
        for f, mine, got in rows:
            if got is None:
                unmatched += 1
                continue
            n += 1
            right = got in SAME[mine]
            ok += right
            conf[(mine, got)] += 1
            if a.detail and not right:
                print(f"  {s} {f}: you {mine}, events {got}")
    print(f"touches matched {n} (no event within {TOL} frames: {unmatched}); type right {ok}/{n} = {ok / max(n, 1):.0%}")
    kinds = sorted({k for k, _ in conf}); got = sorted({g for _, g in conf})
    print("you \\ events " + " ".join(f"{g:>9s}" for g in got))
    for k in kinds:
        print(f"{k:13s} " + " ".join(f"{conf[(k, g)]:9d}" for g in got))


if __name__ == "__main__":
    main()
