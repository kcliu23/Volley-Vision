"""Rally pass A (core.rally.pass_a) vs your serve / rally-over marks: does it cover all in-play
frames, and how much of the video does it keep (what pose then runs on)?

    python -m evaluation.pass_a_eval full_rally_test holdout1 ... [--output output]

Only the part of a clip up to your last rally-over mark is judged (past it nothing is marked).
Compared with the old filter (tools.pose.ball_seen: the ball seen at all).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.common import paths
from core.geometry.camera import Camera
from core.rally.pass_a import track_positions, windows
from evaluation.inplay import in_play


def judge(stem: str, output: str):
    V = json.load(open(paths.verified(stem)))
    truth = in_play(V)
    deads = [int(k) for k, v in (V.get("contacts") or {}).items() if v == "dead"]
    if not truth or not deads:
        return None
    last = max(deads)
    truth = {f for f in truth if f <= last}
    T = json.load(open(Path(output) / stem / "track.json"))
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    W = windows(track_positions(T), float(T.get("fps") or 60.0), cam)
    A = {f for a, b in W for f in range(a, min(b, last) + 1)}
    from tools.pose import ball_seen as seen_play
    S = {f for f in (seen_play(stem, output) or set()) if f <= last}
    lost = sorted(truth - A)
    runs = []
    for f in lost:
        if runs and f - runs[-1][1] <= 1:
            runs[-1][1] = f
        else:
            runs.append([f, f])
    return {"frames": last + 1, "in_play": len(truth), "cover": len(truth & A) / len(truth),
            "kept": len(A) / (last + 1), "old_cover": len(truth & S) / len(truth), "old_kept": len(S) / (last + 1),
            "lost": runs, "windows": len(W)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    print(f"{'clip':16s} {'judged':>7s} {'in play':>8s}   {'pass A: cover':>13s} {'kept':>6s}   {'old: cover':>10s} {'kept':>6s}")
    for s in a.stems:
        r = judge(s, a.output)
        if r is None:
            print(f"{s:16s} (no serve + rally-over marks)")
            continue
        print(f"{s:16s} {r['frames']:7d} {r['in_play'] / r['frames']:8.0%}   {r['cover']:13.1%} {r['kept']:6.0%}"
              f"   {r['old_cover']:10.1%} {r['old_kept']:6.0%}")
        if r["lost"]:
            print(f"{'':16s} in play but not kept: " + ", ".join(f"{x}-{y}" for x, y in r["lost"][:8]))


if __name__ == "__main__":
    main()
