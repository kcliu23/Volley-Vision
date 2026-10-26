"""Regression cases: known-tricky moments, with what a correct track must do there.

    python -m tools.regression                      # rule-based linker (core.linker)
    python -m tools.regression --linker cost        # cost-based linker (core.cost_linker)
    python -m tools.regression --tracknet           # rule linker + TrackNet stage

Uses the cached detections + zoom finds in output/<clip>/cache (no YOLO), so it runs
in seconds. Each case came from a real bug; a linker change that re-breaks one fails here.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(stem, zkey="zoom"):
    from core import paths
    d = json.load(open(paths.dets(stem)))
    fr = [list(f) for f in d["frames"]]
    zoom = d.get(zkey) or d.get("zoom", {})            # each linker's own zoom finds, if cached
    for k, v in zoom.items():
        fr[int(k)] = fr[int(k)] + v["cands"]
    dup = d.get("dup") if len(d.get("dup") or []) == len(fr) else None
    return d, fr, dup


def _near(p, xy, r):
    return p is not None and math.hypot(p[0] - xy[0], p[1] - xy[1]) <= r


# (clip, description, check(track) -> (ok, detail))
def _cases():
    def no_far_court(t):
        bad = [f for f in range(382, 420) if t.pos[f] is not None and t.pos[f][0] < 900 and t.pos[f][1] < 720]
        return not bad, f"far-court frames used: {bad[:6]}"

    def t34_408(t):
        # the ball at 408 is near (1460, 795) — never the sliver 140 px above it
        p = t.pos[408]
        return (p is None or p[1] > 740), f"pos[408]={None if p is None else (round(p[0]), round(p[1]))}"

    def t17_exit(t):
        bad = [f for f in range(323, 342) if t.pos[f] is not None]
        return not bad, f"frames tracked after the ball left the picture: {bad[:6]}"

    def t12_fast(t):
        dets = {382: (1287, 439), 383: (1326, 452), 384: (1396, 477), 385: (1427, 488)}
        bad = [f for f, xy in dets.items() if not _near(t.pos[f], xy, 15)]
        return not bad, f"missed fast-ball detections at {bad}"

    def t21_after_dig(t):
        cov = sum(t.pos[f] is not None for f in range(520, 600)) / 80
        return cov >= 0.75, f"coverage 520-599 = {cov:.0%}"

    def t26_fixture(t):
        bad = [f for f in range(399, 480) if _near(t.pos[f], (1654, 303), 40)]
        return not bad, f"frames on the wall fixture: {len(bad)}"

    def t18_fixture(t):
        bad = [f for f in range(793, 858) if _near(t.pos[f], (1317, 120), 40)]
        return not bad, f"frames on the ceiling object: {len(bad)}"

    def t42_blur(t):
        ok = sum(t.pos[f] is not None and t.state[f] != "filled" for f in range(155, 163))
        return ok >= 6, f"real (non-filled) positions in 155-162: {ok}/8"

    def t53_after_gap(t):
        cov = sum(t.pos[f] is not None for f in range(740, 757)) / 17
        return cov >= 0.8, f"coverage 740-756 = {cov:.0%} (lost when one bad point chained the rest of the clip)"

    def no_teleport(t):
        jumps, pv = [], None
        for f, p in enumerate(t.pos):
            if p is not None and pv is not None:
                if math.hypot(p[0] - pv[1][0], p[1] - pv[1][1]) / (f - pv[0]) > 150:
                    jumps.append(pv[0])
            if p is not None:
                pv = (f, p)
        return not jumps, f">150 px/frame steps at {jumps[:6]}"

    return [
        ("test34", "stays off the far-court ball (382-419)", no_far_court),
        ("test34", "no sliver glitch at 408", t34_408),
        ("test34", "no teleports", no_teleport),
        ("test17", "stops when the ball leaves the picture (323-341)", t17_exit),
        ("test17", "no teleports", no_teleport),
        ("test12", "keeps fast-ball detections after the spike (382-385)", t12_fast),
        ("test21", "follows the ball after the dig (520-599)", t21_after_dig),
        ("test26", "ignores the wall fixture (399-479)", t26_fixture),
        ("test18", "ignores the ceiling object (793-857)", t18_fixture),
        ("test42", "uses the blurred-ball zoom finds (155-162)", t42_blur),
        ("test42", "no teleports", no_teleport),
        ("test53", "keeps the ball after a long gap (740-756)", t53_after_gap),
    ]


def _hybrid(stem, d, a):
    """Tracklets + joint linker on cached candidates (saved score calibration, offset-corrected TrackNet)."""
    from core.tracklet_linker import link_hybrid
    from core.calibration import Calibration
    from core.tracknet import correct
    from tools.calibrate import ball_px
    from core import paths
    ball = ball_px(d)
    tp = paths.tracknet(stem)
    tn = correct(json.load(open(tp))["frames"], ball) if tp.exists() else None
    return link_hybrid(d, tn, Calibration.load(), a["fps"], d["width"], d["height"], ball,
                       dup_flags=a["dup_flags"], content=a["content"], times=a["times"])


def run(linker_name="rule", verbose=True, tracknet=False):
    if linker_name == "cost":
        from core.cost_linker import link as linkf
    else:
        from core.linker import link as linkf
    cache, passed, results = {}, 0, []
    for stem, desc, check in _cases():
        if stem not in cache:
            d, fr, dup = _load(stem, "zoom" if linker_name == "rule" else f"zoom_{linker_name}")
            from core.profile import link_args
            a = link_args(d)
            if linker_name == "hybrid":                 # TrackNet is already part of the hybrid
                cache[stem] = _hybrid(stem, d, a)
                tp = None
            else:
                cache[stem] = linkf(fr, a["fps"], d["width"], d["height"], dup_flags=a["dup_flags"],
                                    content=a["content"], times=a["times"])
                from core import paths
                tp = paths.tracknet(stem)
            if tp is None:
                pass
            elif tracknet and tp.exists():
                from core.tracknet_fill import apply as tracknet_fill
                sz = [max(c[2], c[3]) for f in d["frames"] for c in f if c[4] >= 0.4]
                from core.tracknet import correct
                bpx = sorted(sz)[len(sz) // 2] if sz else 30.0
                tracknet_fill(cache[stem], correct(json.load(open(tp))["frames"], bpx), bpx)
            elif tracknet:
                print(f"  (no TrackNet cache for {stem}: run main.py on it first)")
        ok, detail = check(cache[stem])
        passed += ok
        results.append((stem, desc, ok, detail))
        if verbose:
            print(f"  {'PASS' if ok else 'FAIL'}  {stem:7s} {desc:55s} {'' if ok else detail}")
    if verbose:
        print(f"{passed}/{len(results)} passed ({linker_name} linker{' + TrackNet' if tracknet else ''})")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--linker", choices=["rule", "cost", "hybrid"], default="rule")
    ap.add_argument("--tracknet", action="store_true", help="apply the TrackNet stage (needs <clip>_tracknet.json)")
    a = ap.parse_args()
    run(a.linker, tracknet=a.tracknet)
