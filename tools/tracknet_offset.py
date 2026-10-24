"""Measure the TrackNet checkpoint's position offset against your labels (a model defect),
and save it as a temporary workaround next to the checkpoint.

    python -m tools.tracknet_offset            # measure on checked training clips (data/verified) and save
    python -m tools.tracknet_offset --check    # only report (per clip, incl. the test clips); save nothing

The current fine-tuned TrackNet places the ball ~0.35 ball-widths too high on clips it wasn't
trained on (the off-the-shelf model didn't). Until TrackNet is retrained, every TrackNet
position is shifted by the measured offset (core.tracknet.correct), which makes its
autolabel proposals need fewer fixes. The offset is saved as
models/tracknet/<checkpoint>.offset.json with the checkpoint's fingerprint: a retrained
checkpoint starts uncorrected; run --check on it to see whether the defect is gone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402
from core.tracknet import DEFAULT_CKPT, offset_path, _ckpt_key  # noqa: E402
from tools.calibrate import labels_for, ball_px, train_clips  # noqa: E402


def clip_offsets(stem: str, kind: str):
    """(TrackNet - labelled ball) for confident TrackNet peaks near the labelled ball, in
    ball-widths, from the raw (uncorrected) TrackNet cache."""
    tp = paths.tracknet(stem)
    if not tp.exists():
        return [], []
    d = json.load(open(paths.dets(stem)))
    ball = ball_px(d)
    tn = json.load(open(tp))["frames"]
    dx, dy = [], []
    for f, lab in labels_for(stem, kind).items():
        if not isinstance(lab, list) or f >= len(tn):
            continue
        near = [c for c in tn[f] if c[4] >= 0.3 and math.hypot(c[0] - lab[0], c[1] - lab[1]) <= 1.5 * ball]
        if near:
            c = max(near, key=lambda c: c[4])
            dx.append((c[0] - lab[0]) / ball); dy.append((c[1] - lab[1]) / ball)
    return dx, dy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--check", action="store_true", help="report only, don't save")
    a = ap.parse_args()
    tr = train_clips()
    gt = [(s, "gt") for s in paths.gt_stems() if paths.dets(s).exists()]
    print("TrackNet - your labels (median, ball-widths; + = right / below):")
    allx, ally = [], []
    for stem, kind in tr + gt:
        dx, dy = clip_offsets(stem, kind)
        if dy:
            print(f"  {stem:8s} {'checked clip' if kind == 'verified' else 'test clip   '}  dx {np.median(dx):+.2f}  dy {np.median(dy):+.2f}  (n={len(dy)})")
            if kind == "verified":
                allx += dx; ally += dy
    if not ally:
        print("no checked clips with TrackNet results yet")
        return
    dx, dy = float(np.median(allx)), float(np.median(ally))
    print(f"measured on your checked clips: dx {dx:+.2f}  dy {dy:+.2f}  (n={len(ally)})")
    if a.check:
        return
    if not Path(a.ckpt).exists():
        sys.exit(f"checkpoint not found: {a.ckpt}")
    out = offset_path(a.ckpt)
    json.dump({"schema": "vv-tracknet-offset/1", "checkpoint": str(a.ckpt), "checkpoint_key": _ckpt_key(str(a.ckpt)),
               "offset_bw": [round(dx, 3), round(dy, 3)], "frames": len(ally),
               "measured_on": [s for s, _ in tr], "created": time.strftime("%Y-%m-%d %H:%M"),
               "note": "temporary workaround for a model defect; retraining should remove it"},
              open(out, "w"), indent=1)
    print(f"saved  →  {out}  (applied to every TrackNet position from this checkpoint)")


if __name__ == "__main__":
    main()
