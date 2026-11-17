"""Impossible ball paths in a 3D result (output/<clip>/ball3d.json): what volleyball's rules and
physics forbid, counted per clip.

    python -m evaluation.legality holdout1 holdout2 ... --output output --name now [--output B --name B]

    through net   a flight that changes side below the tape (the ball went through the net)
    outside       a flight that changes side outside the antennas (|y| > 4.5 m)
    4+ touches    a team touching the ball 4+ times in a row (a block at the net doesn't count)
    underground   a flight going below the floor (lowest point < -0.1 m)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.scene3d.ball3d import BALL_D

ANTENNA_Y = 4.5
BLOCK_NEAR_M, BLOCK_BELOW_TAPE_M = 1.0, 0.3


def check(path: Path) -> dict:
    B = json.load(open(path))
    F = B["flights"]
    fr = B["frames"]
    net_h = None
    for f in F:
        nc = f.get("net_crossing")
        if nc:
            net_h = nc["height_m"] - BALL_D / 2 - nc["clears_net_by_m"]
            break
    out = {"flights": len(F), "through net": 0, "outside": 0, "4+ touches": 0, "underground": 0}
    for f in F:
        nc = f.get("net_crossing")
        if nc and nc["clears_net_by_m"] < -0.05:
            out["through net"] += 1
        if nc and abs(nc["y_m"]) > ANTENNA_Y:
            out["outside"] += 1
        if f["min_height_m"] < -0.1:
            out["underground"] += 1
    # touches: where one flight ends and the next starts, in the same chain
    team, n = None, 0
    for a, b in zip(F[:-1], F[1:]):
        if b["chain"] != a["chain"] or b["f0"] != a["f1"] + 1:
            team, n = None, 0
            continue
        p = fr.get(str(b["f0"]))
        if not p:
            continue
        x, z = p["xyz"][0], p["xyz"][2]
        side = 1 if x > 0 else -1
        block = net_h is not None and abs(x) < BLOCK_NEAR_M and z > net_h - BLOCK_BELOW_TAPE_M and side != team
        if block:
            team, n = side, 0
            continue
        if side == team:
            n += 1
            if n == 4:
                out["4+ touches"] += 1
        else:
            team, n = side, 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", action="append", default=None)
    ap.add_argument("--name", action="append", default=None)
    a = ap.parse_args()
    outs = a.output or ["output"]
    names = a.name or [Path(o).name for o in outs]
    keys = ["flights", "through net", "outside", "4+ touches", "underground"]
    print(f"{'version':12s} " + " ".join(f"{k:>12s}" for k in keys) + f" {'impossible':>11s}")
    for o, nm in zip(outs, names):
        tot = dict.fromkeys(keys, 0)
        for s in a.stems:
            p = Path(o) / s / "ball3d.json"
            if p.exists():
                for k, v in check(p).items():
                    tot[k] += v
        bad = sum(tot[k] for k in keys[1:])
        print(f"{nm:12s} " + " ".join(f"{tot[k]:12d}" for k in keys) + f" {bad:11d}")


if __name__ == "__main__":
    main()
