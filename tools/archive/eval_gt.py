"""Score trackers against ground-truth labels (from tools.label_gt).

    python -m tools.eval_gt                 # every clip with labels in data/gt/
    python -m tools.eval_gt --tol 1.5       # "on the ball" = within 1.5 ball-widths

For each labelled frame:
  ball visible in GT  -> HIT if the track is within tol ball-widths, WRONG if it's elsewhere,
                         MISS if the track has no position
  ball not visible    -> OK if the track has no position (or only a filled guess),
                         FALSE if it reports a detected/weak ball somewhere
Uses cached detections + zoom finds (no YOLO). Linkers compared: old tracker, rule, cost, and
rule+tn (rule linker + the TrackNet stage, when output/<clip>/cache/tracknet.json exists).
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def tracks_for(stem):
    from core.linker import link as rule_link
    from core.cost_linker import link as cost_link
    from tools.compare_tracking import run_old
    from core import paths
    d = json.load(open(paths.dets(stem)))
    dup = d.get("dup") if len(d.get("dup") or []) == len(d["frames"]) else None

    def merged(zkey):
        fr = [list(f) for f in d["frames"]]
        for k, v in (d.get(zkey) or {}).items():
            fr[int(k)] = fr[int(k)] + v["cands"]
        return fr

    out = {}
    _, old_pos = run_old(d["frames"], with_pos=True)
    out["old"] = (old_pos, ["detected" if p is not None else "none" for p in old_pos])
    from core.profile import link_args
    a = link_args(d)
    t = rule_link(merged("zoom"), a["fps"], d["width"], d["height"], dup_flags=a["dup_flags"],
                  content=a["content"], times=a["times"])
    out["rule"] = (t.pos, t.state)
    t = cost_link(merged("zoom_cost"), a["fps"], d["width"], d["height"], dup_flags=a["dup_flags"],
                  content=a["content"], times=a["times"])
    out["cost"] = (t.pos, t.state)
    tp = paths.tracknet(stem)
    if tp.exists():
        from core.tracknet_fill import apply as tracknet_fill
        t = rule_link(merged("zoom"), a["fps"], d["width"], d["height"], dup_flags=a["dup_flags"],
                      content=a["content"], times=a["times"])
        sizes0 = [max(c[2], c[3]) for f in d["frames"] for c in f if c[4] >= 0.4]
        from core.tracknet import correct
        bpx = float(np.median(sizes0)) if sizes0 else 30.0
        tracknet_fill(t, correct(json.load(open(tp))["frames"], bpx), bpx)
        out["rule+tn"] = (t.pos, t.state)
    sizes = [max(c[2], c[3]) for f in d["frames"] for c in f if c[4] >= 0.4]
    return out, float(np.median(sizes)) if sizes else 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/gt")
    ap.add_argument("--tol", type=float, default=1.0, help="hit radius in ball-widths")
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.gt}/*_gt.json"))
    if not files:
        print("no labels yet: run python -m tools.label_gt --suggested")
        return
    names = ["old", "rule", "cost", "rule+tn"]
    tot = {n: {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0, "err": []} for n in names}
    print(f"{'clip':8s} {'vis':>4s} {'hid':>4s} | " + " | ".join(f"{n:>4s} hit  wrong miss false" for n in names))
    for fp in files:
        gt = json.load(open(fp))
        stem = Path(fp).name.replace("_gt.json", "")
        from core import paths
        if not paths.dets(stem).exists():
            print(f"{stem}: no detections cached, skipped")
            continue
        tracks, ball = tracks_for(stem)
        row = {n: {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0} for n in names}
        vis = hid = 0
        for k, v in gt["labels"].items():
            f = int(k)
            if v == "skip":
                continue
            for n in names:
                pos, state = tracks[n]
                p = pos[f] if f < len(pos) else None
                s = state[f] if f < len(state) else "none"
                if isinstance(v, list):
                    if p is None:
                        row[n]["miss"] += 1
                    else:
                        e = math.hypot(p[0] - v[0], p[1] - v[1])
                        if e <= args.tol * ball:
                            row[n]["hit"] += 1
                            tot[n]["err"].append(e)
                        else:
                            row[n]["wrong"] += 1
                else:
                    if p is not None and s in ("detected", "weak", "tracknet"):
                        row[n]["false"] += 1
                    else:
                        row[n]["ok"] += 1
            vis += isinstance(v, list)
            hid += v == "none"
        for n in names:
            for kk in row[n]:
                tot[n][kk] += row[n][kk]
        print(f"{stem:8s} {vis:4d} {hid:4d} | " + " | ".join(
            f"{row[n]['hit'] / max(vis, 1):8.0%} {row[n]['wrong']:5d} {row[n]['miss']:4d} {row[n]['false']:5d}" for n in names))
    print()
    for n in names:
        t = tot[n]
        vis = t["hit"] + t["wrong"] + t["miss"]
        hid = t["ok"] + t["false"]
        err = np.median(t["err"]) if t["err"] else float("nan")
        print(f"{n:5s}  on-ball {t['hit'] / max(vis, 1):6.1%}   wrong ball/place {t['wrong'] / max(vis, 1):5.1%}   "
              f"missed {t['miss'] / max(vis, 1):5.1%}   false ball when hidden {t['false'] / max(hid, 1):5.1%}   "
              f"median error {err:5.1f}px")


if __name__ == "__main__":
    main()
