"""Score the finder-first design against the current pipeline on your labelled clips.

    python -m tools.eval_finder_first                                          # WASB as released
    python -m tools.eval_finder_first --finder models/wasb/wasb_ft2.pth.tar    # your fine-tuned WASB
    python -m tools.eval_finder_first --finder ... --clips test45 test46       # a subset

Setups compared (all end with flights placing the final positions):
  current pipeline        full-frame YOLO + zoom (cached dets) + rule linker + TrackNet fill
  YOLO + <finder> hybrid  full-frame YOLO + zoom + the finder, chosen together by the hybrid linker
  finder-first            the finder on every frame; YOLO ONLY on small crops around its candidates
                          (no full-frame YOLO, no zoom rounds); hybrid linker
  finder-first, strict    the same with a tighter crop rule: only the finder's top candidate per
                          frame, and only if its calibrated probability >= --strict-p (fewer crops)

Score calibration is leave-one-clip-out (each clip scored by probabilities fitted on the other
labelled clips). Needs per clip: the YOLO cache (for the current pipeline and ball size), the
video (for the crops), and the finder's candidates (computed if missing; cached).
Crop finds are cached in output/<clip>/cache/crops_<finder>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402
from core.finder_first import crop_finds  # noqa: E402  (shared with the pipeline)

TN = "models/tracknet/FT_TrackNet_best.pt"
CROP_PX = 640 * 640           # YOLO input of one crop ...
FULL_PX = 1280 * 608          # ... vs one full frame at imgsz 1280 (2622x1206 letterboxed): ~0.53 of a frame


def video_of(stem: str, dirs) -> str | None:
    for d in dirs:
        for p in glob.glob(str(Path(d) / f"{stem}.*")):
            return p
    return None


def score(pos, state, labels, ball):
    r = {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0, "off": 0}
    for k, v in labels.items():
        f = int(k)
        if v == "skip":
            continue
        p = pos[f] if f < len(pos) else None
        s = state[f] if f < len(state) else "none"
        if isinstance(v, list):
            if p is None:
                r["miss"] += 1
            else:
                e = math.hypot(p[0] - v[0], p[1] - v[1])
                r["hit" if e <= ball else "wrong"] += 1
                r["off"] += e > 0.25 * ball
        else:
            r["false" if (p is not None and s in ("detected", "weak", "tracknet")) else "ok"] += 1
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--finder", default="models/wasb/wasb_volleyball_best.pth.tar")
    ap.add_argument("--yolo", default="models/best4.pt")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--video-dir", action="append", default=["videos", "holdout"])
    ap.add_argument("--strict-p", type=float, default=0.6,
                    help="strict crop rule: crop only the finder's top candidate with calibrated probability >= this")
    a = ap.parse_args()
    from ultralytics import YOLO
    from core import finders
    from core.calibration import fit
    from core.linker import link as rule_link
    from core.tracknet_fill import apply as fill
    from core.tracklet_linker import link_hybrid
    from core.joint_linker import JointParams
    from core.profile import link_args
    from core.flights import from_track, apply_flights
    import tools.calibrate as C

    gt = paths.gt_stems()
    hold = [s for s in paths.holdout_stems() if paths.verified(s).exists()]
    checked = [s for s, _ in C.train_clips()]
    groups = [("holdout clips", hold), ("dev clips (data/gt)", gt), ("checked training clips", checked)]
    if a.clips:
        groups = [(t, [s for s in g if s in a.clips]) for t, g in groups]
    clips = [s for _, g in groups for s in g if paths.dets(s).exists() and video_of(s, a.video_dir)]
    ftag = finders.tag(a.finder) or "tracknet"
    model = YOLO(a.yolo)

    # finder candidates (computed if missing) + per-clip calibration samples
    C.EXTRA_FINDERS[:] = [a.finder] if finders.kind(a.finder) != "tracknet" else []
    per_clip = {}
    for s in clips:
        d = json.load(open(paths.dets(s)))
        if finders.load_cached(s, a.finder) is None:
            finders.candidates(video_of(s, a.video_dir), d, a.finder)
        if finders.load_cached(s, TN) is None:
            finders.candidates(video_of(s, a.video_dir), d, TN)
    lab_clips = [(s, "gt") for s in gt if paths.dets(s).exists()] + C.train_clips() + [(s, "verified") for s in hold]
    for s, kind in lab_clips:
        if finders.load_cached(s, a.finder) is not None:
            per_clip[s] = C.collect([(s, kind)])
    name = finders.calib_name(a.finder)

    def cal_without(stem):
        merged = {}
        for s, smp in per_clip.items():
            if s == stem:
                continue
            for src, (X, y) in smp.items():
                merged.setdefault(src, ([], []))
                merged[src][0].append(X); merged[src][1].append(y)
        return fit({k: (np.concatenate(X), np.concatenate(y)) for k, (X, y) in merged.items()}).for_finder(name)

    res, frames = {}, 0
    crops = {"finder-first": 0, "finder-first, strict": 0}
    for s in clips:
        d = json.load(open(paths.dets(s))); ball = C.ball_px(d); a_ = link_args(d)
        labels = json.load(open(paths.gt(s) if s in gt else paths.verified(s)))["labels"]
        kw = dict(dup_flags=a_["dup_flags"], content=a_["content"], times=a_["times"])
        out = {}

        def finish(n, t, ckpt=TN):
            fl, ct = from_track(t)
            pos, st = apply_flights(t, fl, ct, place_tracknet=finders.place_on_arc(ckpt))
            out[n] = score(pos, st, labels, ball)

        fr = [list(f) for f in d["frames"]]
        for k, v in (d.get("zoom") or {}).items():
            fr[int(k)] += v["cands"]
        t = rule_link(fr, a_["fps"], d["width"], d["height"], **kw)
        fill(t, finders.correct(finders.load_cached(s, TN), ball, TN), ball)
        finish("current pipeline", t)
        cands = finders.correct(finders.load_cached(s, a.finder), ball, a.finder)
        cal = cal_without(s)
        jp = JointParams(tn_only_bias=1.0)
        # B: full-frame YOLO + zoom + the finder, chosen together by the hybrid linker
        t = link_hybrid(d, cands, cal, a_["fps"], d["width"], d["height"], ball, jparams=jp, **kw)
        finish(f"YOLO + {ftag} hybrid", t, a.finder)
        frames += len(d["frames"])
        # C: finder-first (loose crop rule) and C': strict crop rule
        for label, keep, suffix in (("finder-first", None, ""),
                                    ("finder-first, strict", lambda c: cal.prob("tracknet", [c[4]])[0] >= a.strict_p,
                                     f"_strict{a.strict_p:g}")):
            cf = crop_finds(s, video_of(s, a.video_dir), cands, ball, ftag, model, keep=keep, suffix=suffix)
            crops[label] += cf["crops"]
            d2 = dict(d); d2["frames"] = [[] for _ in d["frames"]]; d2["zoom"] = cf["found"]
            t = link_hybrid(d2, cands, cal, a_["fps"], d["width"], d["height"], ball, jparams=jp, **kw)
            finish(f"{label} ({ftag})", t, a.finder)
        res[s] = out
        print(f"  {s}: done", flush=True)

    print(f"\nYOLO work vs the current pipeline's full frames (one 640 px crop ~ {CROP_PX / FULL_PX:.2f} of a 1280 px frame; "
          f"not counting the current pipeline's zoom rounds):")
    for label, n in crops.items():
        print(f"  {label:22s} {n / max(frames, 1):.2f} crops per frame  ~ {n * CROP_PX / FULL_PX / max(frames, 1):.0%} of full-frame YOLO")
    for title, g in groups:
        g = [s for s in g if s in res]
        if not g:
            continue
        print(f"{title} ({len(g)} clips)")
        print(f"  {'':28s} {'on-ball':>7s} {'wrong':>6s} {'missed':>7s} {'false':>6s} {'>0.25bw':>8s}")
        for n in res[g[0]]:
            t = {k: sum(res[c][n][k] for c in g) for k in ("hit", "wrong", "miss", "ok", "false", "off")}
            vis, hid = t["hit"] + t["wrong"] + t["miss"], t["ok"] + t["false"]
            print(f"  {n:28s} {t['hit'] / vis:7.1%} {t['wrong'] / vis:6.1%} {t['miss'] / vis:7.1%} "
                  f"{t['false'] / max(hid, 1):6.1%} {t['off'] / max(t['hit'] + t['wrong'], 1):8.1%}")


if __name__ == "__main__":
    main()
