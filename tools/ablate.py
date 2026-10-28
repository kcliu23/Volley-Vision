"""Which parts of the finder-first tracker earn their keep? Score variants on the same cached inputs.

    python -m tools.ablate                         # every variant, every labelled clip
    python -m tools.ablate --variants joint classic --clips holdout1 holdout5

Inputs per clip (all cached, no YOLO or WASB is run): the clip profile (sample_dets.json, else
dets.json), the finder's candidates, and the YOLO crop finds for the pipeline's crop rule (taken
from the loose crop cache). Every variant ends with flights (parabolas between contacts), as in
the pipeline. Variants:

  joint     the pipeline: finder candidates + YOLO crop finds -> joint linker (core.joint_linker.link)
  classic   the classic pipeline on the full-frame YOLO cache: rule linker + zoom finds + TrackNet fill

Add a variant here to test a change (e.g. a stage removed) against the pipeline on the same inputs.
2026-10-28: a tracklet layer + rival/fixture removal before the joint linker scored the same on
holdout/dev and worse on the checked clips, so it was removed (core/archive/tracklet_linker.py).

Scored like tools.eval_finder_first (on-ball = within 1 ball-width of your label), per group
(holdout / dev / checked) and per clip, plus the regression cases.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402

FINDER = "models/wasb/wasb_ft2.pth.tar"


def inputs(stem: str, cal, rule):
    from core import finders
    from core.finder_first import ball_px, subset_finds
    from core.profile import link_args
    sp = paths.cache_dir(stem) / "sample_dets.json"
    d = json.load(open(sp if sp.exists() else paths.dets(stem)))
    ball = ball_px(d)
    cands = finders.correct(finders.load_cached(stem, FINDER), ball, FINDER)
    from core.finder_first import crop_suffix
    lp = paths.cache_dir(stem) / f"crops_{finders.tag(FINDER)}.json"
    if lp.exists():
        found, _ = subset_finds(json.load(open(lp))["found"], cands, lambda pk: float(cal.prob("tracknet", [pk])[0]), ball, *rule)
    else:                                              # no loose cache: this rule's own cache
        found = json.load(open(paths.cache_dir(stem) / f"crops_{finders.tag(FINDER)}{crop_suffix(*rule)}.json"))["found"]
    from core.finder_first import SAMPLE_EVERY
    d2 = dict(d)
    d2["zoom"] = found
    # as in the pipeline: the sizing pass's full-frame boxes (every SAMPLE_EVERY-th frame) are candidates
    d2["frames"] = [list(fr) if f % SAMPLE_EVERY == 0 else [] for f, fr in enumerate(d["frames"])]
    return d2, cands, ball, link_args(d)


def run_variant(name, d2, cands, cal, ball, a, stem=None):
    """name: "joint", "classic", or "joint:key=value,key=value" (JointParams changed)."""
    from dataclasses import fields
    from core.joint_linker import JointParams
    from core.finder_first import FINDER_ONLY_BIAS
    jp = JointParams(tn_only_bias=FINDER_ONLY_BIAS)
    if name.startswith("joint:"):
        types = {f.name: f.type for f in fields(JointParams)}
        kv = dict(x.split("=") for x in name.split(":", 1)[1].split(","))
        conv = {k: (v.lower() in ("1", "true") if types[k] in (bool, "bool") else
                    int(v) if types[k] in (int, "int") else float(v)) for k, v in kv.items()}
        jp = JointParams(**{"tn_only_bias": FINDER_ONLY_BIAS, **conv})
        name = "joint"
    kw = dict(dup_flags=a["dup_flags"], content=a["content"], times=a["times"])
    W, H = d2["width"], d2["height"]
    if name == "joint":
        from core.joint_linker import link
        return link(d2, cands, cal, a["fps"], W, H, ball, jp, **kw)
    if name == "classic":
        from core import finders
        from core.linker import link as rule_link
        from core.tracknet_fill import apply as fill
        from core.profile import link_args
        TN = "models/tracknet/FT_TrackNet_best.pt"
        d = json.load(open(paths.dets(stem)))
        fr = [list(f) for f in d["frames"]]
        for k, v in (d.get("zoom") or {}).items():
            fr[int(k)] += v["cands"]
        la = link_args(d)
        t = rule_link(fr, la["fps"], d["width"], d["height"], dup_flags=la["dup_flags"], content=la["content"],
                      times=la["times"])
        tn = finders.load_cached(stem, TN)
        if tn is not None:
            fill(t, finders.correct(tn, ball, TN), ball)
        return t
    raise ValueError(name)


def finish(t, variant="joint"):
    """Flights, as in the pipeline (WASB points stay where WASB put them; TrackNet's go on the arc).
    A variant may end in "|key=value,..." to change apply_flights' options (e.g. "joint|max_fill_s=0.3",
    "joint|none" = no flights at all, the linker's own track)."""
    from core.flights import from_track, apply_flights
    opts = {}
    if "|" in variant:
        variant, o = variant.split("|", 1)
        if o == "none":
            return list(t.pos), list(t.state)
        opts = {k: (None if v == "none" else v.lower() == "true" if v.lower() in ("true", "false") else float(v))
                for k, v in (x.split("=") for x in o.split(","))}
    fl, ct = from_track(t)
    return apply_flights(t, fl, ct, place_tracknet=(variant == "classic"), **opts)


def score(pos, state, labels, ball):
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
            else:
                r["hit" if math.hypot(p[0] - v[0], p[1] - v[1]) <= ball else "wrong"] += 1
        else:
            r["false" if (p is not None and s in ("detected", "weak", "tracknet")) else "ok"] += 1
    return r


def fmt(t):
    vis, hid = t["hit"] + t["wrong"] + t["miss"], t["ok"] + t["false"]
    return (f"{t['hit'] / max(vis, 1):7.1%} {t['wrong'] / max(vis, 1):6.1%} {t['miss'] / max(vis, 1):7.1%} "
            f"{t['false'] / max(hid, 1):6.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variants", nargs="+", default=["joint", "classic"])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", default=None, help="save per-clip results (json)")
    a = ap.parse_args()
    from core import finders
    from core.calibration import Calibration
    from core.finder_first import CROP_TOP, CROP_P
    from tools.regression import _cases
    import tools.calibrate as C
    cal = Calibration.load().for_finder(finders.calib_name(FINDER))
    gt = paths.gt_stems()
    groups = [("holdout", [s for s in paths.holdout_stems() if paths.verified(s).exists()]),
              ("dev (data/gt)", gt), ("checked", [s for s, _ in C.train_clips()])]
    if a.clips:
        groups = [(g, [s for s in ss if s in a.clips]) for g, ss in groups]
    cases = _cases()
    labelled = {s for _, ss in groups for s in ss}
    reg_only = [c for c in dict.fromkeys(c[0] for c in cases) if c not in labelled and (not a.clips or c in a.clips)]
    if reg_only:
        groups = groups + [("regression only", reg_only)]
    res, reg = {}, {v: [] for v in a.variants}
    from core.finder_first import crop_suffix
    for g, ss in groups:
        for s in ss:
            if not ((paths.cache_dir(s) / "crops_wasb_ft2.json").exists()
                    or (paths.cache_dir(s) / f"crops_wasb_ft2{crop_suffix(CROP_TOP, CROP_P)}.json").exists()):
                print(f"  {s}: no crop cache (run main.py on it), skipped")
                continue
            t0 = time.time()
            d2, cands, ball, args_ = inputs(s, cal, (CROP_TOP, CROP_P))
            labels = {} if g == "regression only" else \
                json.load(open(paths.gt(s) if s in gt else paths.verified(s)))["labels"]
            res[s] = {}
            for v in a.variants:
                t = run_variant(v.split("|")[0], d2, cands, cal, ball, args_, stem=s)
                for stem, desc, check in cases:
                    if stem == s:
                        ok, det = check(t)
                        reg[v].append((s, desc, ok, det))
                pos, st = finish(t, v)
                res[s][v] = score(pos, st, labels, ball)
            print(f"  {s}: done ({time.time() - t0:.0f}s)", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    for g, ss in groups:
        ss = [s for s in ss if s in res]
        if not ss or g == "regression only":
            continue
        print(f"\n{g} ({len(ss)} clips)          on-ball  wrong  missed  false")
        for v in a.variants:
            t = {k: sum(res[s][v][k] for s in ss) for k in ("hit", "wrong", "miss", "ok", "false")}
            print(f"  {v:22s} {fmt(t)}")
        print("  per clip (on-ball): " + "  ".join(
            f"{s}:" + "/".join(f"{res[s][v]['hit'] / max(1, res[s][v]['hit'] + res[s][v]['wrong'] + res[s][v]['miss']):.0%}"
                              for v in a.variants) for s in ss))
    print("\nregression cases")
    for v in a.variants:
        r = reg[v]
        print(f"  {v:10s} {sum(ok for *_, ok, _ in r)}/{len(r)} passed"
              + "".join(f"\n     FAIL {s} {d}: {det}" for s, d, ok, det in r if not ok))


if __name__ == "__main__":
    main()
