"""Score the joint linker on the test clips, leave-one-clip-out, next to the current pipeline.

    python -m tools.eval_joint                          # default weights
    python -m tools.eval_joint --grid                   # small search over the main weights

For each test clip (data/gt), the probability curves (core.calibration) are fit on all the
OTHER labelled clips (the other test clips + checked training clips), so no clip is scored
by a model that saw its labels. Reported next to "rule+tn" (rule linker + TrackNet stage).
Note: choosing weights by --grid on these same clips is mildly optimistic; confirm on new
test clips before switching the default.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402

_CACHE = {}


def _score(pos, state, labels, ball):
    r = {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0}
    for k, v in labels.items():
        f = int(k)
        if v == "skip":
            continue
        p = pos[f] if f < len(pos) else None
        s = state[f] if f < len(state) else "none"
        if isinstance(v, list):
            if p is None:
                r["miss"] += 1
            elif math.hypot(p[0] - v[0], p[1] - v[1]) <= ball:
                r["hit"] += 1
            else:
                r["wrong"] += 1
        else:
            r["false" if (p is not None and s in ("detected", "weak", "tracknet")) else "ok"] += 1
    return r


def _cal_without(hold: str):
    from core.calibration import fit
    from tools.calibrate import collect, train_clips
    clips = [(s, "gt") for s in paths.gt_stems() if s != hold and paths.dets(s).exists()] + train_clips()
    return fit(collect(clips))


def _run_clip(args):
    stem, params_list = args
    from core.joint_linker import link, JointParams
    from core.linker import link as rule_link
    from core.tracknet_fill import apply as tracknet_fill
    from core.profile import link_args
    from tools.calibrate import ball_px
    d = json.load(open(paths.dets(stem)))
    labels = json.load(open(paths.gt(stem)))["labels"]
    ball = ball_px(d)
    from core.tracknet import correct
    tn = correct(json.load(open(paths.tracknet(stem)))["frames"], ball) if paths.tracknet(stem).exists() else None
    a = link_args(d)
    cal = _cal_without(stem)
    out = {}
    fr = [list(f) for f in d["frames"]]
    for k, v in (d.get("zoom") or {}).items():
        fr[int(k)] = fr[int(k)] + v["cands"]
    t = rule_link(fr, a["fps"], d["width"], d["height"], dup_flags=a["dup_flags"], content=a["content"], times=a["times"])
    if tn is not None:
        tracknet_fill(t, tn, ball)
    out["rule+tn"] = _score(t.pos, t.state, labels, ball)
    for name, kw in params_list:
        P = JointParams(**kw)
        t = link(d, tn, cal, a["fps"], d["width"], d["height"], ball, P,
                 dup_flags=a["dup_flags"], content=a["content"], times=a["times"])
        out[name] = _score(t.pos, t.state, labels, ball)
    return stem, out


def summarize(results: dict, names: list[str]):
    rows = []
    for n in names:
        t = {k: sum(results[c][n][k] for c in results) for k in ("hit", "wrong", "miss", "ok", "false")}
        vis, hid = t["hit"] + t["wrong"] + t["miss"], t["ok"] + t["false"]
        rows.append((n, t["hit"] / vis, t["wrong"] / vis, t["miss"] / vis, t["false"] / max(hid, 1)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    configs = [("joint", {})]
    if a.grid:
        configs = [(f"joint w{w} m{m} t{t} g{g}", {"w_det": w, "miss": m, "touch": t, "agree_bonus": g})
                   for w, m, t, g in itertools.product((0.7, 1.0, 1.5), (0.0, 0.5, 1.0), (3.0, 6.0), (0.5, 1.5))]
    clips = [s for s in paths.gt_stems() if paths.dets(s).exists()]
    with Pool(a.workers) as pool:
        results = dict(pool.map(_run_clip, [(s, configs) for s in clips]))
    names = ["rule+tn"] + [n for n, _ in configs]
    rows = summarize(results, names)
    base = rows[0]
    rows = [base] + sorted(rows[1:], key=lambda r: -(r[1] - 0.5 * max(0, r[4] - base[4])))
    print(f"{'':28s} on-ball  wrong  missed  false-when-hidden   (leave-one-clip-out, {len(clips)} test clips)")
    for n, h, w, m, fa in rows[:12]:
        print(f"{n:28s} {h:7.1%} {w:6.1%} {m:7.1%} {fa:10.1%}")
    if len(configs) == 1:
        print("\nper clip on-ball:")
        for c in clips:
            r = results[c]
            print("  " + c.ljust(8) + "  ".join(f"{n} {r[n]['hit'] / max(1, r[n]['hit'] + r[n]['wrong'] + r[n]['miss']):.0%}" for n in names))


if __name__ == "__main__":
    main()
