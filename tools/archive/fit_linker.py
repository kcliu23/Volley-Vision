"""Fit the tracklet linker's weights (core.tracklet_linker.AssocParams) on your labels.

    python -m tools.fit_linker                 # fit on dev + checked clips, report holdout
    python -m tools.fit_linker --save          # ... and save models/linker_params.json (used by the linker)
    python -m tools.fit_linker --top 2 --q 0.1 # for another crop rule

The tracklet step decides WHICH object is the ball in play (the game ball, not a ball on the
next court, a head or a light). How it weighs "this object is confidently seen" against "this
is a plausible continuation of the ball we were following" used to be hand-set. Here those
weights are learned: for every labelled clip the finder-first inputs (WASB candidates + YOLO
crop finds, calibrated) and the tracklets are built once; then the chain choice is scored
against your labels for many weight settings (coordinate descent), and the best setting on the
training clips (data/gt + checked training clips) is kept. Holdout clips are only scored, never
fitted on.

Score per labelled frame: +1 if the chosen chain covers it within one ball-width of your label,
-1 if it covers it somewhere else (wrong object), -1 if it claims a ball in a no-ball frame.
With --save, writes models/linker_params.json, which the linker then loads automatically.
Measured 2026-10-27: fitting raised the chain's on-ball share on holdout by 0.4-1.4 points but
left the full pipeline (after the joint linker) unchanged, so no weights file is shipped yet.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402

FINDER = "models/wasb/wasb_ft2.pth.tar"
from core.finder_first import CROP_TOP, CROP_P  # noqa: E402  the pipeline's default crop rule
# weight -> values tried (coordinate descent; the current value is always kept as a candidate)
GRID = {
    "logit_cap": [1e9, 4.0, 2.0, 1.0, 0.5],
    "w_det": [0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
    "miss": [0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0],
    "reentry": [3.0, 6.0, 12.0, 25.0, 50.0, 100.0],
    "reentry_gap_s": [0.3, 0.6, 1.0, 1.5],
    "k_size": [3.0, 10.0, 40.0, 100.0, 250.0],
    "size_tol": [0.15, 0.3, 0.5],
    "size_tol_s": [0.0, 0.1, 0.3, 1.0],
    "sig_gap_bw_s": [2.0, 4.0, 8.0],
    "touch": [1.0, 3.0, 6.0],
}


def prepare(stem: str, labels: dict, cal, rule=(CROP_TOP, CROP_P)):
    """Tracklets + precomputed hand-over features for one clip (finder-first inputs, YOLO crops
    by `rule` = (top, q), simulated from the loose crop cache)."""
    from core import finders
    from core.profile import link_args
    from core.joint_linker import JointParams, build_points
    from core.cost_linker import _gravity
    from core.linker import DETECTED
    from core.tracklets import build as build_tracklets
    from core.tracklet_linker import _meet
    from core.finder_first import ball_px, subset_finds
    sp = paths.cache_dir(stem) / "sample_dets.json"
    d = json.load(open(sp if sp.exists() else paths.dets(stem)))
    ball = ball_px(d)
    a = link_args(d)
    cands = finders.correct(finders.load_cached(stem, FINDER), ball, FINDER)
    loose = json.load(open(paths.cache_dir(stem) / f"crops_{finders.tag(FINDER)}.json"))
    found, ncrops = subset_finds(loose["found"], cands, lambda pk: float(cal.prob("tracknet", [pk])[0]), ball, *rule)
    d2 = dict(d); d2["frames"] = [[] for _ in d["frames"]]; d2["zoom"] = found
    n, fps = len(d["frames"]), a["fps"]
    dup = [bool(x) for x in a["dup_flags"]] if a["dup_flags"] is not None and len(a["dup_flags"]) == n else [False] * n
    upre = np.concatenate([[0], np.cumsum([not x for x in dup])])
    pts = build_points(d2, cands, cal, ball, JointParams(tn_only_bias=1.0), fps)
    gy = _gravity({f: [(x, y, s or ball, p, 1.0) for (x, y, s, lg, st, p) in pts[f] if st == DETECTED]
                   for f in range(n)}, [not x for x in dup], ball)
    trs = build_tracklets(pts, fps, ball, gy, dup, None)
    pairs = {}                                    # (i, j, s) -> features (weights applied later)
    for j, b in enumerate(trs):
        for i in range(j):
            A = trs[i]
            if A.f1 >= b.f1:
                continue
            if A.f1 >= b.f0:
                s = next((k for k, f in enumerate(b.frames) if f > A.f1), None)
                if s is None:
                    continue
                ov = A.f1 - b.f0 + 1
            else:
                s, ov = 0, 0
            fa, fb = A.f1, b.frames[s]
            dm, fm = _meet(A, b, gy, fa, fb)
            pairs[(i, j)] = (s, ov, fa, fb, dm, fm < fb - 1, A.size(-1), b.size(0),
                             int(upre[fb] - upre[fa + 1]) if fb > fa else 0)
    # labels -> per tracklet point: +1 on the ball, -1 wrong object / no-ball frame, 0 unlabelled
    val = []
    for t in trs:
        v = []
        for f, x, y in zip(t.frames, t.xs, t.ys):
            L = labels.get(str(f))
            if isinstance(L, list):
                v.append(1 if math.hypot(x - L[0], y - L[1]) <= ball else -1)
            elif L in ("none",):
                v.append(-1)
            else:
                v.append(0)
        val.append(np.array(v))
    vis = sum(1 for L in labels.values() if isinstance(L, list))
    return {"stem": stem, "trs": trs, "pairs": pairs, "upre": upre, "n": n, "fps": fps, "ball": ball,
            "val": val, "logits": [np.asarray(t.logits, float) for t in trs], "vis": vis,
            "crops": ncrops, "frames": n}


def choose(c, P):
    """The tracklet DP of core.tracklet_linker.link, on precomputed features. Returns [(j, start)]."""
    trs, upre, n, fps, ball = c["trs"], c["upre"], c["n"], c["fps"], c["ball"]
    N = len(trs)
    if not N:
        return []
    max_ov = max(0, int(round(P.max_overlap_s * fps)))
    re_gap = int(round(P.reentry_gap_s * fps))
    capped = [np.minimum(lg, P.logit_cap) for lg in c["logits"]]
    suf = [np.concatenate([np.cumsum(x[::-1])[::-1], [0.0]]) for x in capped]   # sum from k on
    INF = float("inf")
    best, back, trim = [INF] * N, [None] * N, [0] * N
    by_j = {}
    for (i, j), ft in c["pairs"].items():
        by_j.setdefault(j, []).append((i, ft))
    for j, b in enumerate(trs):
        cand = (P.miss * int(upre[b.f0]) - P.w_det * suf[j][0], None, 0)
        for i, (s, ov, fa, fb, dm, touch, sa, sb, unc) in by_j.get(j, ()):
            if best[i] == INF or ov > max_ov:
                continue
            gap_s = max(fb - fa, 1) / fps
            sig = ball * (P.sig0_bw + P.sig_gap_bw_s * gap_s)
            h = (dm / sig) ** 2 + (P.touch if touch else 0.0)
            if fb - fa >= re_gap:
                h = min(h, P.reentry)
            if sa and sb:
                ex = abs(math.log(sb / sa)) - (P.size_tol + P.size_tol_s * gap_s)
                h += P.k_size * max(0.0, ex) ** 2
            v = best[i] + P.miss * unc + h - P.w_det * suf[j][s]
            if v < cand[0]:
                cand = (v, i, s)
        best[j], back[j], trim[j] = cand
    end_cost, end_j = P.miss * int(upre[n]), None
    for j, t in enumerate(trs):
        v = best[j] + P.miss * int(upre[n] - upre[t.f1 + 1])
        if v < end_cost:
            end_cost, end_j = v, j
    chain, j = [], end_j
    while j is not None:
        chain.append((j, trim[j]))
        j = back[j]
    return chain[::-1]


def score(c, P) -> tuple[int, int, int]:
    """(hits, wrong-or-false, labelled ball frames) for the chain chosen with weights P."""
    hit = bad = 0
    seen = set()
    for j, s in choose(c, P):
        for f, v in zip(c["trs"][j].frames[s:], c["val"][j][s:]):
            if f in seen:
                continue
            seen.add(f)
            hit += v > 0
            bad += v < 0
    return hit, bad, c["vis"]


def total(clips, P):
    h = b = v = 0
    for c in clips:
        x, y, z = score(c, P)
        h += x; b += y; v += z
    return h - b, h / max(v, 1), b / max(v, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--save", action="store_true", help="write models/linker_params.json")
    ap.add_argument("--top", type=int, default=CROP_TOP, help="crop rule: candidates per frame")
    ap.add_argument("--q", type=float, default=CROP_P, help="crop rule: calibrated P(ball) needed (-1 = none)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--cached", action="store_true", help="reuse the prepared clips from the last run (same labels)")
    a = ap.parse_args()
    from core.calibration import Calibration
    from core import finders
    from core.tracklet_linker import AssocParams
    import tools.calibrate as C
    cal = Calibration.load().for_finder(finders.calib_name(FINDER))
    gt = paths.gt_stems()
    groups = {"train": [(s, paths.gt(s)) for s in gt] + [(s, paths.verified(s)) for s, _ in C.train_clips()],
              "holdout": [(s, paths.verified(s)) for s in paths.holdout_stems() if paths.verified(s).exists()]}
    data = {}
    t0 = time.time()
    import pickle
    cache = paths.dev() / "fit_linker_cache.pkl"
    rule = (a.top, None if a.q < 0 else a.q)
    key = (FINDER, rule, sorted(str(lp) for g in groups.values() for _, lp in g),
           Path("models/score_calibration.json").stat().st_mtime)
    if a.cached and cache.exists():
        k0, data = pickle.load(open(cache, "rb"))
        if k0 != key:
            data = {}
    for g, items in groups.items():
        if g in data:
            continue
        data[g] = []
        for s, lp in items:
            if not (paths.cache_dir(s) / f"crops_{finders.tag(FINDER)}.json").exists() \
                    or finders.load_cached(s, FINDER) is None:
                print(f"  {s}: no loose crop cache (python -m tools.eval_finder_first on it), skipped")
                continue
            data[g].append(prepare(s, json.load(open(lp))["labels"], cal, rule))
        print(f"{g}: {len(data[g])} clips prepared ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump((key, data), open(paths.ensure(cache), "wb"))

    P0 = AssocParams()
    P = P0
    J = total(data["train"], P)[0]
    print(f"start (hand-set): train score {J}")
    for r in range(a.rounds):
        changed = False
        for k, vals in GRID.items():
            best_v, best_J = getattr(P, k), J
            for v in vals:
                Q = replace(P, **{k: v})
                Jq = total(data["train"], Q)[0]
                if Jq > best_J:
                    best_v, best_J = v, Jq
            if best_v != getattr(P, k):
                P, J, changed = replace(P, **{k: best_v}), best_J, True
                print(f"  round {r + 1}: {k} = {best_v:g}  -> train score {J}", flush=True)
        if not changed:
            break
    for g in ("train", "holdout"):
        for name, Q in (("hand-set", P0), ("fitted", P)):
            sc, hr, br = total(data[g], Q)
            print(f"{g:8s} {name:9s} chain on the ball {hr:6.1%}   wrong object / false {br:5.1%}")
    fitted = {k: getattr(P, k) for k in GRID}
    print("fitted:", {k: (round(v, 3) if v < 1e8 else "none") for k, v in fitted.items()})
    if a.save:
        from core.tracklet_linker import PARAMS_PATH
        json.dump({"schema": "vv-linker-params/1", "finder": FINDER, "crop_rule": list(rule),
                   "trained_on": [c["stem"] for c in data["train"]], "assoc": fitted,
                   "hand_set": {k: getattr(P0, k) for k in GRID}}, open(PARAMS_PATH, "w"), indent=1)
        print(f"saved {PARAMS_PATH}")


if __name__ == "__main__":
    main()
