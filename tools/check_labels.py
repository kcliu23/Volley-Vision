"""Everything to re-check in one clip's labels, in one list (the last step of tools.annotate).

    python -m tools.check_labels holdout1            # data/verified, data/player_ids, output/<clip>
    python -m tools.check_labels holdout1 holdout2 --summary

    ball / touches   core.common.label_checks (tools.review: b jumps to each)
    players          the same number on two people in one frame; a number that changes side of
                     the net and stays there (a swap?); numbered people that look nothing alike
                     (needs the re-ID model, core.people.reid_model: skipped without it)
    3D               a touch where the 3D ball is more than 3 m from the player you said touched
                     it, or, if you didn't say, from everyone the pose saw on court, numbered or
                     not (evaluation.reach3d, mix): an unnumbered far server isn't flagged

Nothing is changed: it prints frames and reasons, and the tool to fix each in.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.common import paths
from core.common.label_checks import all_flags
from core.people.player_labels import _iou

SIDE_HOLD = 30            # frames a number must stay on the other side to count as a switch
COURT_X, COURT_Y = 12.0, 8.0   # feet beyond this (court + free zone) aren't on the floor: skipped


def _runs(flags: dict[int, str]) -> dict[int, str]:
    """The same reason on consecutive frames: one item, 'frames a-b'."""
    out, prev = {}, None
    for f in sorted(flags):
        why = flags[f]
        if prev and f - prev[1] <= 2 and why == prev[2]:
            prev[1] = f
            continue
        if prev:
            out[prev[0]] = prev[2] + (f" (frames {prev[0]}-{prev[1]})" if prev[1] > prev[0] else "")
        prev = [f, f, why]
    if prev:
        out[prev[0]] = prev[2] + (f" (frames {prev[0]}-{prev[1]})" if prev[1] > prev[0] else "")
    return out


def player_flags(ids: dict, cam) -> dict[int, str]:
    out: dict[int, str] = {}
    fr = {int(k): v for k, v in (ids.get("frames") or {}).items()}
    side_runs = defaultdict(list)                       # number -> [(frame, side)]
    for f in sorted(fr):
        seen = defaultdict(list)
        for b in fr[f]:
            n = b.get("player")
            if not isinstance(n, int):
                continue
            seen[n].append(b["box"])
            if cam is not None:
                x0, y0, x1, y1 = b["box"]
                g = cam.floor_point([((x0 + x1) / 2, y1)], 0.0)[0]
                if abs(g[0]) > COURT_X or abs(g[1]) > COURT_Y:   # off the court area: feet in the air
                    continue                                    # (a jump) or not in the picture
                if abs(g[0]) > 1.0:                     # near the net: no side
                    side_runs[n].append((f, int(np.sign(g[0]))))
        for n, bs in seen.items():
            apart = [i for i in range(1, len(bs)) if _iou(bs[0], bs[i]) < 0.3]   # two boxes on ONE person: not yours
            if apart:
                out.setdefault(f, f"#{n} on {len(apart) + 1} people")
    for n, L in side_runs.items():
        cur, run = None, []
        for f, s in L:
            if cur is None:
                cur = s
                continue
            if s != cur:
                run.append((f, s))
                if len(run) >= SIDE_HOLD and all(x[1] == s for x in run):
                    out.setdefault(run[0][0], f"#{n} moves to the other side of the net and stays (a swap? or feet hidden)")
                    cur, run = s, []
            else:
                run = []
    return _runs(out)


def report(stem: str, output: str) -> list[tuple[int, str, str]]:
    """[(frame, what, tool to fix it in)]."""
    rows = []
    vp = Path(paths.verified(stem))
    V = json.load(open(vp)) if vp.exists() else {}
    for f, why in all_flags(V).items():
        rows.append((f, why, "review"))
    ip = Path("data/player_ids") / f"{stem}.json"
    cam = None
    cp = Path("data/cameras") / f"{stem}.json"
    if cp.exists():
        from core.geometry.camera import Camera
        cam = Camera.load(cp)
    if ip.exists():
        for f, why in player_flags(json.load(open(ip)), cam).items():
            rows.append((f, why, "label_players"))
    bp = Path(output) / stem / "ball3d_labels.json"          # the 3D from your labels (annotate step 5) ...
    if not bp.exists():
        bp = bp.with_name("ball3d.json")                     # ... else the automatic one
    if bp.exists() and ip.exists() and cam is not None and V:
        from evaluation.reach3d import FAR_M, gaps
        g, _ = gaps(stem, output, bp.name, mix=True)
        for f, kind, d, n, _z in g:
            if d > FAR_M:
                who = "anyone on court" if n == -1 else f"player #{n} (you said they touched it)"
                rows.append((f, f"3D: ball {d:.2f} m from {who} at this {kind}",
                             "review (ball / touch) or label_players"))
    return sorted(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--summary", action="store_true", help="counts only")
    a = ap.parse_args()
    for s in a.stems:
        rows = report(s, a.output)
        kinds = defaultdict(int)
        for _, why, _ in rows:
            kinds["players" if why.startswith("#") else why.split(":")[0]] += 1
        print(f"{s}: {len(rows)} to re-check  " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
        if not a.summary:
            for f, why, tool in rows:
                print(f"  {f:6d}  {why:78s}  [{tool}]")


if __name__ == "__main__":
    main()
