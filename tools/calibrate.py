"""Fit the per-source "is this the ball" probabilities (core.calibration) on labelled frames.

    python -m tools.calibrate                 # fit on checked training clips (data/verified) -> models/score_calibration.json
    python -m tools.calibrate --report        # + leave-one-clip-out check on the test clips (data/gt)

Each candidate in a labelled frame is positive if it lies within one ball-width of the
labelled ball, negative otherwise (and every candidate in a "no ball" / "hidden" frame is
negative). Training uses data/verified clips with role "train"; data/gt test clips are only
used for the leave-one-clip-out report, never for the saved model.

TrackNet positions are taken with the checkpoint's offset workaround applied (see
tools.tracknet_offset), so positives are judged where the pipeline will actually use them.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402
from core.calibration import SOURCES, features, fit  # noqa: E402


def labels_for(stem: str, kind: str) -> dict[int, object]:
    if kind == "gt":
        lab = json.load(open(paths.gt(stem)))["labels"]
    else:
        V = json.load(open(paths.verified(stem)))
        lab = V["labels"]
    return {int(k): v for k, v in lab.items() if v != "skip"}


def candidates(stem: str) -> tuple[dict, list[list[tuple]]]:
    """(dets dict, per-frame list of (source, x, y, size, conf, aspect)); TrackNet positions
    corrected by the offset being fitted (set_offset) or the saved one."""
    d = json.load(open(paths.dets(stem)))
    n = len(d["frames"])
    out = [[] for _ in range(n)]
    for f, fr in enumerate(d["frames"]):
        for c in fr:
            out[f].append(("yolo", c[0], c[1], max(c[2], c[3]), c[4], min(c[2], c[3]) / max(c[2], c[3], 1e-6)))
    for k, v in (d.get("zoom") or {}).items():
        for c in v["cands"]:
            out[int(k)].append(("zoom", c[0], c[1], max(c[2], c[3]), c[4], min(c[2], c[3]) / max(c[2], c[3], 1e-6)))
    tp = paths.tracknet(stem)
    if tp.exists():
        from core.tracknet import correct
        tn = correct(json.load(open(tp))["frames"], ball_px(d))
        for f, fr in enumerate(tn[:n]):
            for c in fr:
                out[f].append(("tracknet", c[0], c[1], c[2], c[4], 1.0))
    return d, out


def ball_px(d: dict) -> float:
    prof = d.get("profile") or {}
    if prof.get("ball_px"):
        return float(prof["ball_px"])
    s = [max(c[2], c[3]) for f in d["frames"] for c in f if c[4] >= 0.4]
    return float(np.median(s)) if s else 30.0


def collect(clips: list[tuple[str, str]]) -> dict:
    """clips: [(stem, "gt" | "verified")] -> {source: (X, y)}"""
    rows = {s: ([], [], [], []) for s in SOURCES}       # conf, size_ratio, aspect, y
    for stem, kind in clips:
        d, cands = candidates(stem)
        ball = ball_px(d)
        dup = d.get("dup") or []
        for f, lab in labels_for(stem, kind).items():
            if f >= len(cands) or (f < len(dup) and dup[f]):
                continue
            for src, x, y, size, conf, asp in cands[f]:
                pos = isinstance(lab, list) and math.hypot(x - lab[0], y - lab[1]) <= ball
                r = rows[src]
                r[0].append(conf); r[1].append(size / ball); r[2].append(asp); r[3].append(int(pos))
    out = {}
    for s, (c, r, a, y) in rows.items():
        y = np.asarray(y, int)
        out[s] = (features(s, c, r, a) if len(y) else np.zeros((0, 1)), y)
    return out


def train_clips() -> list[tuple[str, str]]:
    gt = set(paths.gt_stems())
    out = []
    for p in sorted(paths.VERIFIED.glob("*.json")):
        V = json.load(open(p))
        if V.get("role") != "test" and p.stem not in gt and paths.dets(p.stem).exists():
            out.append((p.stem, "verified"))
    return out


def report(cal, test: dict, label: str):
    from sklearn.metrics import roc_auc_score
    for s in SOURCES:
        X, y = test[s]
        if len(y) == 0 or y.min() == y.max():
            continue
        m = cal.models[s]
        p = 1 / (1 + np.exp(-(X @ np.asarray(m["coef"]) + m["intercept"])))
        raw = 1 / (1 + np.exp(-X[:, 0]))                # the source's own score
        bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
        cells = []
        for lo, hi in zip(bins, bins[1:]):
            k = (p >= lo) & (p < hi)
            if k.sum() >= 10:
                cells.append(f"{lo:.1f}-{min(hi, 1):.1f}: {y[k].mean():.2f} (n={k.sum()})")
        print(f"  {label:10s} {s:8s} AUC learned {roc_auc_score(y, p):.3f} vs raw score {roc_auc_score(y, raw):.3f}"
              f"   | predicted -> actual: " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--out", default="models/score_calibration.json")
    a = ap.parse_args()
    tr = train_clips()
    gt = [(s, "gt") for s in paths.gt_stems() if paths.dets(s).exists()]
    if not tr:
        print("no checked training clips yet (data/verified); fitting on the test clips instead is not allowed")
        return
    samples = collect(tr)
    cal = fit(samples)
    cal.save(a.out, {"trained_on": [s for s, _ in tr]})
    print(f"fitted on {len(tr)} checked clip(s) {[s for s, _ in tr]}: "
          + ", ".join(f"{s} {len(samples[s][1])} candidates ({int(samples[s][1].sum())} on the ball)" for s in SOURCES)
          + f"  →  {a.out}")
    if a.report:
        print("held-out check on the test clips (never used for fitting):")
        report(cal, collect(gt), "test clips")


if __name__ == "__main__":
    main()
