"""Compare ball finders (TrackNet, WASB, ...) on your labelled clips, before any linking.

    python -m tools.compare_finders                              # current TrackNet vs WASB
    python -m tools.compare_finders --finders tracknet wasb tn2:models/tracknet/FT2_TrackNet_best.pt
    python -m tools.compare_finders --runs                       # compute missing candidate caches first

A finder is "tracknet" (current checkpoint, offset-corrected), "wasb" (core.wasb; "wasb:path" for a
fine-tuned checkpoint), or "name:checkpoint" for another TrackNet checkpoint. Candidates come from each finder's cache
(output/<clip>/cache/tracknet[_tag].json); --runs computes missing ones (needs the videos).

Per group of clips (holdout / dev / checked), for each finder:
  found       share of visible-ball frames with a candidate within 1 ball-width (what a linker
              could use at best)
  top right   share of visible-ball frames where its strongest candidate (peak >= --thr) is on the ball
  top wrong   ... is somewhere else
  false       share of hidden / no-ball frames where it still reports a ball (peak >= --thr)
  precision   median distance of on-ball candidates to your label, in ball-widths
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


def finder_spec(s: str):
    if s == "tracknet":
        from core.tracknet import DEFAULT_CKPT
        return "tracknet", None, DEFAULT_CKPT
    if s == "wasb":
        from core.wasb import DEFAULT_CKPT, TAG
        return "wasb", TAG, DEFAULT_CKPT
    if s.startswith("wasb:"):                     # a fine-tuned WASB: "wasb:models/wasb/wasb_ft2.pth.tar"
        from core import finders
        ckpt = s.split(":", 1)[1]
        return finders.tag(ckpt), finders.tag(ckpt), ckpt
    name, ckpt = s.split(":", 1)
    return name, name, ckpt


def load_cands(stem, name, tag, ckpt, ball, video, run):
    cp = paths.tracknet(stem, tag=tag)
    dets = json.load(open(paths.dets(stem)))
    if run and video:
        if name.startswith("wasb"):
            from core.wasb import candidates
        else:
            from core.tracknet import candidates
        fr = candidates(video, dets, ckpt, cache_path=cp, log=None)
    elif cp.exists():
        fr = json.load(open(cp))["frames"]
    else:
        return None
    if fr is None:
        return None
    if name == "tracknet" or not name.startswith("wasb"):
        from core.tracknet import correct
        fr = correct(fr, ball, ckpt=ckpt)       # the per-checkpoint offset (0 if none measured)
    return fr


def score(fr, labels, ball, thr):
    r = {"vis": 0, "hid": 0, "found": 0, "top_ok": 0, "top_wrong": 0, "false": 0, "err": []}
    for k, v in labels.items():
        f = int(k)
        if v == "skip" or f >= len(fr):
            continue
        cands = fr[f]
        top = max(cands, key=lambda c: c[4]) if cands else None
        if isinstance(v, list):
            r["vis"] += 1
            d = [math.hypot(c[0] - v[0], c[1] - v[1]) / ball for c in cands]
            if d and min(d) <= 1.0:
                r["found"] += 1
                r["err"].append(min(d))
            if top is not None and top[4] >= thr:
                ok = math.hypot(top[0] - v[0], top[1] - v[1]) <= ball
                r["top_ok" if ok else "top_wrong"] += 1
        else:
            r["hid"] += 1
            if top is not None and top[4] >= thr:
                r["false"] += 1
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--finders", nargs="+", default=["tracknet", "wasb"])
    ap.add_argument("--thr", type=float, default=0.5, help="peak needed to count a finder's own decision")
    ap.add_argument("--runs", action="store_true", help="compute missing candidate caches (needs videos)")
    ap.add_argument("--video-dir", action="append", default=["videos", "holdout"])
    a = ap.parse_args()
    from tools.calibrate import ball_px
    specs = [finder_spec(s) for s in a.finders]
    gt = paths.gt_stems()
    hold = [s for s in paths.holdout_stems() if paths.verified(s).exists()]
    checked = [p.stem for p in sorted(paths.VERIFIED.glob("*.json")) if p.stem not in gt and p.stem not in hold]

    def video_of(stem):
        for d in a.video_dir:
            for p in glob.glob(str(Path(d) / f"{stem}.*")):
                return p
        return None

    for title, group, src in (("holdout clips", hold, "verified"), ("dev clips (data/gt)", gt, "gt"),
                              ("checked training clips", checked, "verified")):
        tot = {s[0]: None for s in specs}
        n_clips = 0
        for stem in group:
            if not paths.dets(stem).exists():
                continue
            labels = json.load(open(paths.gt(stem) if src == "gt" else paths.verified(stem)))["labels"]
            ball = ball_px(json.load(open(paths.dets(stem))))
            rows = {}
            for name, tag, ckpt in specs:
                fr = load_cands(stem, name, tag, ckpt, ball, video_of(stem), a.runs and not paths.tracknet(stem, tag=tag).exists())
                if fr is None:
                    break
                rows[name] = score(fr, labels, ball, a.thr)
            if len(rows) < len(specs):            # compare only clips every finder has
                continue
            n_clips += 1
            for name, r in rows.items():
                if tot[name] is None:
                    tot[name] = r
                else:
                    for k in r:
                        tot[name][k] = tot[name][k] + r[k]
        if not n_clips:
            print(f"{title}: no clips with candidates from every finder (run with --runs)")
            continue
        print(f"{title}: {n_clips} clips (peak >= {a.thr} for its own decision)")
        print(f"  {'':12s} {'found':>7s} {'top right':>10s} {'top wrong':>10s} {'false':>7s} {'precision':>10s}")
        for name, r in tot.items():
            print(f"  {name:12s} {r['found'] / max(r['vis'], 1):7.1%} {r['top_ok'] / max(r['vis'], 1):10.1%} "
                  f"{r['top_wrong'] / max(r['vis'], 1):10.1%} {r['false'] / max(r['hid'], 1):7.1%} "
                  f"{np.median(r['err']) if r['err'] else float('nan'):9.2f}bw")


if __name__ == "__main__":
    main()
