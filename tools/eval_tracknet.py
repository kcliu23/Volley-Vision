"""Benchmark TrackNet candidates against GT, alone and combined with YOLO through the rule linker.

    python -m tools.eval_tracknet --tag vb            # needs output/detections/<stem>_tracknet_vb.json
    python -m tools.eval_tracknet --tag vb --scale 0.7

Reports
  candidate recall  share of visible-ball GT frames where a source has ANY candidate within
                    tol ball-widths (YOLO >= 0.05 incl. zoom finds / TrackNet / union):
                    the ceiling for what a linker can do with those candidates
  tracks            on-ball / wrong / missed / false-when-hidden, as in tools.eval_gt, for
                      yolo        rule linker on YOLO + zoom (current pipeline)
                      tn          TrackNet's top blob per frame, peak >= 0.5 (its own decision rule)
                      tn+link     rule linker on TrackNet candidates only
                      yolo+tn     rule linker on YOLO + zoom + TrackNet candidates (conf x scale)
"""
import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def score(pos, state, labels, ball, tol):
    r = {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0, "err": []}
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
                if e <= tol * ball:
                    r["hit"] += 1
                    r["err"].append(e)
                else:
                    r["wrong"] += 1
        else:
            r["false" if (p is not None and s in ("detected", "weak")) else "ok"] += 1
    return r


def any_near(cands, v, r):
    return any(math.hypot(c[0] - v[0], c[1] - v[1]) <= r for c in cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vb")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply TrackNet conf before linking")
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--gt", default="data/gt")
    ap.add_argument("--clips", nargs="*")
    a = ap.parse_args()
    from core.linker import link
    from core.profile import link_args

    names = ["yolo", "tn", "tn+link", "yolo+tn"]
    tot = {n: {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0, "err": []} for n in names}
    rec = {"yolo": 0, "tn": 0, "union": 0, "vis": 0, "tn_only": 0, "yolo_only": 0}
    print(f"{'clip':7s} {'vis':>4s} | recall yolo   tn  union | " + " | ".join(f"{n:>7s}" for n in names))
    for g in sorted(glob.glob(f"{a.gt}/*_gt.json")):
        stem = Path(g).name.replace("_gt.json", "")
        if a.clips and stem not in a.clips:
            continue
        tp = Path(f"output/detections/{stem}_tracknet_{a.tag}.json")
        if not tp.exists():
            print(f"{stem}: no TrackNet cache ({tp}), skipped")
            continue
        labels = json.load(open(g))["labels"]
        d = json.load(open(f"output/detections/{stem}_dets.json"))
        tn = json.load(open(tp))["frames"]
        n = len(d["frames"])
        yo = [list(f) for f in d["frames"]]
        for k, v in (d.get("zoom") or {}).items():
            yo[int(k)] = yo[int(k)] + v["cands"]
        sizes = [max(c[2], c[3]) for f in d["frames"] for c in f if c[4] >= 0.4]
        ball = float(np.median(sizes)) if sizes else 30.0
        la = link_args(d)

        def run_link(fr):
            t = link(fr, la["fps"], d["width"], d["height"], dup_flags=la["dup_flags"],
                     content=la["content"], times=la["times"])
            return t.pos, t.state

        tn_scaled = [[[c[0], c[1], c[2], c[3], min(1.0, c[4] * a.scale)] for c in f] for f in tn]
        tracks = {}
        tracks["yolo"] = run_link(yo)
        top = [(f[0][0], f[0][1]) if f and f[0][4] >= 0.5 else None for f in tn]
        tracks["tn"] = (top, ["detected" if p else "none" for p in top])
        tracks["tn+link"] = run_link(tn_scaled)
        tracks["yolo+tn"] = run_link([yo[i] + tn_scaled[i] for i in range(n)])

        vis = 0; ry = rt = ru = 0
        for k, v in labels.items():
            if not isinstance(v, list):
                continue
            f = int(k); vis += 1
            hy = any_near([c for c in yo[f] if c[4] >= 0.05], v, a.tol * ball)
            ht = any_near(tn[f], v, a.tol * ball)
            ry += hy; rt += ht; ru += hy or ht
            rec["tn_only"] += ht and not hy; rec["yolo_only"] += hy and not ht
        rec["yolo"] += ry; rec["tn"] += rt; rec["union"] += ru; rec["vis"] += vis
        row = {}
        for nme in names:
            row[nme] = score(*tracks[nme], labels, ball, a.tol)
            for kk in tot[nme]:
                tot[nme][kk] += row[nme][kk]
        print(f"{stem:7s} {vis:4d} |      {ry / max(vis, 1):4.0%} {rt / max(vis, 1):4.0%} {ru / max(vis, 1):5.0%} | "
              + " | ".join(f"{row[nme]['hit'] / max(vis, 1):7.0%}" for nme in names), flush=True)

    V = max(rec["vis"], 1)
    print(f"\ncandidate recall (any candidate within {a.tol} ball-width): YOLO {rec['yolo'] / V:.1%}   "
          f"TrackNet {rec['tn'] / V:.1%}   union {rec['union'] / V:.1%}   "
          f"(TrackNet-only {rec['tn_only']}, YOLO-only {rec['yolo_only']} of {rec['vis']} frames)")
    print(f"TrackNet conf scale for linking: {a.scale}")
    for nme in names:
        t = tot[nme]
        vis = t["hit"] + t["wrong"] + t["miss"]; hid = t["ok"] + t["false"]
        err = np.median(t["err"]) if t["err"] else float("nan")
        print(f"{nme:8s} on-ball {t['hit'] / max(vis, 1):6.1%}   wrong {t['wrong'] / max(vis, 1):5.1%}   "
              f"missed {t['miss'] / max(vis, 1):5.1%}   false-when-hidden {t['false'] / max(hid, 1):5.1%}   "
              f"median err {err:5.1f}px")


if __name__ == "__main__":
    main()
