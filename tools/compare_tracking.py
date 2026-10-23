"""Compare the current online tracker with the offline linker on saved detections.

    python -m tools.compare_tracking output/detections/*_dets.json

The old tracker is run exactly as main.py runs it (BallTracker, conf >= 0.40 from
YOLO), fed from the saved candidates instead of re-running YOLO.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.linker import link, LinkParams  # noqa: E402


class _Box:
    def __init__(self, c):
        cx, cy, w, h, conf = c
        self.xyxy = [np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])]
        self.conf = [conf]


class _Res:
    def __init__(self, cands):
        self.boxes = [_Box(c) for c in cands]


def run_old(frames, model="models/best4.pt", conf=0.40, with_pos=False):
    from core.ball_tracker import BallTracker
    BallTracker.__init__.__defaults__  # noqa
    tr = BallTracker.__new__(BallTracker)
    # replicate __init__ without loading YOLO
    from collections import deque
    tr.conf, tr.ignore_regions = conf, []
    tr.trail, tr.position = deque(maxlen=40), None
    tr.last_det, tr.last_conf = None, None
    tr._kf, tr._kf_ready, tr._lost, tr._prev_pos = BallTracker._build_kf(), False, 0, None
    tr._weak_streak = tr._decoy_streak = 0
    tr._pending_pos = None
    out, poss = [], []
    for f in frames:
        cands = [c for c in f if c[4] >= conf]
        pos = tr.update_from_result(_Res(cands))
        out.append("det" if tr.last_det is not None else ("coast" if pos is not None else "none"))
        poss.append(pos)
    return (out, poss) if with_pos else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dets", nargs="+")
    ap.add_argument("--save", action="store_true", help="write output/<stem>_link.json")
    args = ap.parse_args()
    tot = {"old": 0, "new": 0, "n": 0}
    print(f"{'clip':10s} {'frames':>6s} | {'old det':>7s} {'coast':>5s} {'cover':>6s} | "
          f"{'det':>5s} {'weak':>5s} {'fill':>5s} {'cover':>6s} | {'gain':>5s}")
    for p in args.dets:
        d = json.load(open(p))
        frames = d["frames"]
        old = run_old(frames)
        o_cov = 1 - old.count("none") / len(old)
        t = link(frames, d["fps"], d["width"], d["height"], LinkParams())
        s = t.summary()
        stem = Path(p).name.replace("_dets.json", "")
        print(f"{stem:10s} {len(frames):6d} | {old.count('det'):7d} {old.count('coast'):5d} {o_cov:6.1%} | "
              f"{s['detected']:5d} {s['weak']:5d} {s['filled']:5d} {s['coverage']:6.1%} | {s['coverage'] - o_cov:+5.1%}")
        tot["old"] += o_cov * len(old)
        tot["new"] += s["coverage"] * len(old)
        tot["n"] += len(old)
        if args.save:
            out = Path("output") / f"{stem}_link.json"
            json.dump({"schema": "vv-link/1", "fps": d["fps"], "width": d["width"], "height": d["height"],
                       "frames": t.to_rows()}, open(out, "w"))
    if tot["n"]:
        print(f"\nALL: old coverage {tot['old'] / tot['n']:.1%}   new coverage {tot['new'] / tot['n']:.1%}")


if __name__ == "__main__":
    main()
