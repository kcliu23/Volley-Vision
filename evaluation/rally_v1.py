"""Rally pass B v1 (core.rally.pass_b) vs the rule (core/scene3d/rallies.py), leave one clip out.

    python -m evaluation.rally_v1 full_rally_test holdout1 ... test20 [--output output] [--rules <dir> ...]

v1+rule: v1's rallies with their edges snapped to the rule's (core.rally.pass_b.snap_edges).
Each clip is predicted by trees trained on the OTHER clips only, inside the part you've checked
(evaluation.rallies_eval.checked_span). Both are scored with evaluation.rallies_eval.judge.
--rules: more folders to look for <clip>/rallies.json (the rule's output) when output/ has none.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths
from core.rally import pass_b
from core.rally.pass_a import track_positions
from evaluation.inplay import in_play
from evaluation.rallies_eval import END_TOL_S, START_TOL_F, checked_span, edge_hits, judge, truth_rallies


def load(stem: str, output: str) -> dict | None:
    V = json.load(open(paths.verified(stem)))
    truth = truth_rallies(V.get("contacts") or {})
    rej = [int(k) for k in (V.get("not_rally") or {})]
    fps, n = float(V.get("fps") or 60.0), int(V["n"])
    span = checked_span(truth, rej, fps, n)
    d = Path(output) / stem
    if span is None or not (d / "track.json").exists():
        return None
    T = json.load(open(d / "track.json"))
    pos = track_positions(T)
    if len(pos) < n:
        pos = np.vstack([pos, np.full((n - len(pos), 2), np.nan)])
    cp = Path("data/cameras") / f"{stem}.json"
    cam = None
    if cp.exists():
        from core.geometry.camera import Camera
        cam = Camera.load(cp)
    pose = json.load(open(d / "pose.json"))["frames"] if (d / "pose.json").exists() else None
    X, names = pass_b.features(pos[:n], fps, cam, float(V["height"]), pose)
    y = pass_b.window_labels(in_play(V) or set(), n, fps)
    w = max(1, int(round(pass_b.WINDOW_S * fps)))
    a, b = span
    rows = np.arange(len(X))
    keep = (rows * w >= a) & ((rows + 1) * w - 1 <= b)                 # windows inside the checked part
    return {"stem": stem, "X": X, "y": y, "keep": keep, "names": names, "fps": fps, "n": n,
            "truth": truth, "rej": rej, "span": span}


def rule_rallies(stem: str, dirs: list[str]) -> list[tuple[int, int]] | None:
    for d in dirs:
        p = Path(d) / stem / "rallies.json"
        if p.exists():
            return [(r["f0"], r["f1"]) for r in json.load(open(p))["rallies"]]
    return None


def _pass(ok, n):
    return f"{ok}/{n}" if n else "-"


def _line(name, J):
    return (f"{name:7s} found {J['found']}/{J['n_true']} fake {len(J['fake'])}  play kept "
            f"{J['play_kept'] / max(J['play'], 1):.1%}  dead cut {J['dead_cut'] / max(J['dead'], 1):.1%}  "
            f"start <={START_TOL_F} frames {_pass(J['start_ok'], J['start_n'])}  "
            f"end <={END_TOL_S:g} s {_pass(J['end_ok'], J['end_n'])}")


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--rules", nargs="*", default=[])
    a = ap.parse_args()
    C = [c for c in (load(s, a.output) for s in a.stems) if c is not None]
    tot = {"v1": [], "rule": [], "v1+rule": []}
    for c in C:
        tr = [o for o in C if o is not c]
        Xtr = np.vstack([o["X"][o["keep"]] for o in tr])
        ytr = np.concatenate([o["y"][o["keep"]] for o in tr])
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
        p = m.predict_proba(c["X"])[:, 1]
        acc = np.mean((p[c["keep"]] >= 0.5) == c["y"][c["keep"]])
        v1 = pass_b.to_rallies(p, c["fps"], c["n"])
        J1 = judge(c["truth"], v1, c["span"], c["rej"])
        J1.update(edge_hits(J1, c["fps"]))
        tot["v1"].append(J1)
        print(f"{c['stem']:16s} windows {int(c['keep'].sum()):5d} ({c['y'][c['keep']].mean():.0%} in play), "
              f"v1 window accuracy {acc:.1%}")
        print("   " + _line("v1", J1))
        R = rule_rallies(c["stem"], [a.output] + a.rules)
        if R is not None:
            J0 = judge(c["truth"], R, c["span"], c["rej"])
            J0.update(edge_hits(J0, c["fps"]))
            tot["rule"].append(J0)
            print("   " + _line("rule", J0))
            J2 = judge(c["truth"], pass_b.snap_edges(v1, R, c["fps"]), c["span"], c["rej"])
            J2.update(edge_hits(J2, c["fps"]))
            tot["v1+rule"].append(J2)
            print("   " + _line("v1+rule", J2))
    for k, Js in tot.items():
        if not Js:
            continue
        S = {key: (sum((J[key] for J in Js), []) if isinstance(Js[0][key], list) else sum(J[key] for J in Js))
             for key in ("starts", "ends", "accepted_start", "accepted_end", "found", "n_true", "fake",
                         "play_kept", "play", "dead_cut", "dead", "start_ok", "start_n", "end_ok", "end_n")}
        print(f"\nALL {_line(k, S)}  ({len(Js)} clips)")


if __name__ == "__main__":
    main()
