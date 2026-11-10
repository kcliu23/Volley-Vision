"""Does the 3D ball path hold together at touches? Breaks between flights, and the 2D fit.

    python -m evaluation.joins3d test20                       # output/test20/ball3d.json
    python -m evaluation.joins3d test20 --before old.json     # side by side with an older run

At each join between two flights of a rally, the ball's step across the touch (per frame) is
compared with its normal step just before and after. A BREAK is a step more than 2.5x the
larger of those (+ 0.1 m): the two flights don't meet, usually because one sits at the wrong
distance from the camera (a shift along the camera's line of sight is invisible in 2D).
Also: the 3D ball projected into the picture against the tracker's 2D ball (px).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths
from core.geometry.camera import Camera


def report(stem: str, B: dict, cam, track: dict) -> dict:
    fr = {int(k): np.array(v["xyz"]) for k, v in B["frames"].items()}
    fl = B["flights"]
    breaks, joins = [], 0
    for a, b in zip(fl[:-1], fl[1:]):
        f1, f0 = a["f1"], b["f0"]
        if f0 - f1 > 10 or not all(x in fr for x in (f1 - 1, f1, f0, f0 + 1)):
            continue                                       # a new rally, not a touch
        joins += 1
        before, after = np.linalg.norm(fr[f1] - fr[f1 - 1]), np.linalg.norm(fr[f0 + 1] - fr[f0])
        d = fr[f0] - fr[f1]
        across = np.linalg.norm(d) / (f0 - f1)
        if across > 2.5 * max(before, after) + 0.1:
            ray = (fr[f1] - cam.position) / np.linalg.norm(fr[f1] - cam.position)
            breaks.append((f1, f0, float(np.linalg.norm(d)), float(d @ ray)))
    e = []
    for f, X in fr.items():
        p = track.get(f)
        if p is not None:
            e.append(float(np.hypot(*(cam.project([X])[0] - p))))
    e = np.array(e)
    return {"flights": len(fl), "joins": joins, "breaks": breaks, "px_median": float(np.median(e)),
            "px_p90": float(np.percentile(e, 90)), "frames": len(fr)}


def show(name, r):
    print(f"  {name:7s} {r['flights']:3d} flights, {r['joins']:3d} joins, {len(r['breaks'])} breaks; "
          f"3D vs 2D ball: median {r['px_median']:.1f} px, 90% within {r['px_p90']:.1f} px; {r['frames']} frames")
    for f1, f0, d, along in r["breaks"]:
        print(f"           break {f1}->{f0}: {d:.2f} m apart ({along:+.2f} m along the camera's line of sight)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stem")
    ap.add_argument("--before", help="an older ball3d.json to compare with")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    cam = Camera.load(Path("data/cameras") / f"{a.stem}.json")
    track = {r["f"]: r["pos"] for r in json.load(open(paths.track(a.stem, a.output)))["frames"] if r.get("pos")}
    print(a.stem)
    if a.before:
        show("before", report(a.stem, json.load(open(a.before)), cam, track))
    show("now", report(a.stem, json.load(open(Path(a.output) / a.stem / "ball3d.json")), cam, track))


if __name__ == "__main__":
    main()
