"""Rally detection vs your marks: the serve (start) and "rally over" (end) you marked in tools.review.

    python -m evaluation.rallies_eval full_rally_test holdout1 ... [--output output] [--labels] [--detail]

Only the part you've CHECKED is judged: from your first mark (serve, rally over or a rejected
guess, X) to your last closed rally (its rally-over mark) or rejection, 3 s padding each side. A
full game you're halfway through is scored on that half; the rallies past it are "not checked yet".

Per clip and in total:
  rallies   your rallies (serve -> rally over) matched one-to-one to the detected ones by overlap:
            found / missed / fake (a detected rally overlapping none of yours; "you rejected" = X).
  edges     start = detected start - your serve, end = detected end - your rally over, in frames.
            Marks placed with P (accept the guess as is) are exactly the guess: they'd score the
            rule against itself, so they are counted apart ("accepted as is") and left out of
            the median.
  in play   the product number: of your in-play frames, how many a cut video keeps; of the dead
            frames, how many it cuts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths

PAD_S = 3.0               # s: the checked span reaches this far past your first / last mark


def runs(frames: list[int]) -> list[int]:
    """First frame of each run of consecutive frames (a serve marked on 3 frames = one serve)."""
    out = []
    for f in sorted(frames):
        if not out or f - out[-1][-1] > 1:
            out.append([f])
        else:
            out[-1].append(f)
    return [r[0] for r in out]


def truth_rallies(contacts: dict) -> list[tuple[int, int | None]]:
    """Your rallies: (serve, rally over), rally over None when you haven't marked it (yet)."""
    C = {int(k): v for k, v in contacts.items()}
    serves = runs([f for f, v in C.items() if v == "serve"])
    deads = runs([f for f, v in C.items() if v == "dead"])
    out = []
    for i, s in enumerate(serves):
        nxt = serves[i + 1] if i + 1 < len(serves) else float("inf")
        d = [x for x in deads if s < x < nxt]
        out.append((s, d[0] if d else None))
    return out


def checked_span(truth: list, rejected: list[int], fps: float, n: int) -> tuple[int, int] | None:
    """The frames you've checked: first mark .. last closed rally end or rejection, padded."""
    closed = [d for _, d in truth if d is not None]
    lo = [s for s, _ in truth] + list(rejected)
    hi = closed + list(rejected)
    if not lo or not hi:
        return None
    pad = int(round(PAD_S * fps))
    return max(0, min(lo) - pad), min(n - 1, max(hi) + pad)


def judge(truth: list, pred: list[tuple[int, int]], span: tuple[int, int], rejected: list[int] = (),
          near: int = 90) -> dict:
    """Score detected rallies `pred` [(f0, f1)] against your rallies `truth` [(serve, dead|None)]
    inside `span`. Pure function (tests use it)."""
    a, b = span
    T = [(s, d) for s, d in truth if d is not None and a <= s and d <= b]
    P = [(f0, f1) for f0, f1 in pred if f1 >= a and f0 <= b]
    # one-to-one by overlap, biggest first
    pairs = sorted(((min(d, f1) - max(s, f0), i, j) for i, (s, d) in enumerate(T)
                    for j, (f0, f1) in enumerate(P) if min(d, f1) > max(s, f0)), reverse=True)
    mt, mp, match = set(), set(), []
    for _, i, j in pairs:
        if i not in mt and j not in mp:
            mt.add(i); mp.add(j); match.append((i, j))
    starts, ends, acc_s, acc_e = [], [], 0, 0
    for i, j in match:
        (s, d), (f0, f1) = T[i], P[j]
        if f0 == s:
            acc_s += 1
        else:
            starts.append(f0 - s)
        if f1 == d:
            acc_e += 1
        else:
            ends.append(f1 - d)
    fakes = [P[j] for j in range(len(P)) if j not in mp]
    rej = sum(1 for f0, f1 in fakes if any(abs(r - f0) <= near or abs(r - f1) <= near for r in rejected))
    # in play, frame by frame, inside the span
    play = np.zeros(b - a + 1, bool)
    for s, d in T:
        play[s - a:d - a + 1] = True
    kept = np.zeros_like(play)
    for f0, f1 in P:
        kept[max(f0, a) - a:min(f1, b) - a + 1] = True
    return {"n_true": len(T), "found": len(match), "missed": [T[i] for i in range(len(T)) if i not in mt],
            "fake": fakes, "fake_rejected": rej, "starts": starts, "ends": ends,
            "accepted_start": acc_s, "accepted_end": acc_e,
            "play": int(play.sum()), "play_kept": int((play & kept).sum()),
            "dead": int((~play).sum()), "dead_cut": int((~play & ~kept).sum()), "span": span}


def _pct(x, n):
    return f"{x / n:.0%}" if n else "-"


def _edges(name, v, acc):
    v = np.array(v)
    if not len(v):
        return f"{name}: -" + (f" ({acc} accepted as is)" if acc else "")
    return (f"{name}: median {np.median(v):+.0f} frames, within 3 frames {np.mean(np.abs(v) <= 3):.0%}, "
            f"within 0.5 s {np.mean(np.abs(v) <= 30):.0%} (n={len(v)}, {acc} accepted as is left out)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--labels", action="store_true", help="rallies_labels.json (from your 2D ball)")
    ap.add_argument("--detail", action="store_true", help="every missed / fake rally")
    a = ap.parse_args()
    sfx = "_labels" if a.labels else ""
    tot: dict = {}
    print(f"{'clip':16s} {'checked':>13s} {'yours':>5s} {'found':>5s} {'missed':>6s} {'fake':>5s} "
          f"{'play kept':>9s} {'dead cut':>8s}")
    for s in a.stems:
        p = Path(a.output) / s / f"rallies{sfx}.json"
        if not p.exists() or not paths.verified(s).exists():
            print(f"{s:16s} (no rallies{sfx}.json or no labels)")
            continue
        V = json.load(open(paths.verified(s)))
        R = json.load(open(p))["rallies"]
        truth = truth_rallies(V.get("contacts") or {})
        rejected = [int(k) for k in (V.get("not_rally") or {})]
        span = checked_span(truth, rejected, V.get("fps", 30), V["n"])
        if span is None:
            print(f"{s:16s} (no closed rally marked)")
            continue
        J = judge(truth, [(r["f0"], r["f1"]) for r in R], span, rejected)
        later = sum(1 for r in R if r["f0"] > span[1])
        print(f"{s:16s} {span[0]:>6d}-{span[1]:<6d} {J['n_true']:5d} {J['found']:5d} {len(J['missed']):6d} "
              f"{len(J['fake']):5d} {_pct(J['play_kept'], J['play']):>9s} {_pct(J['dead_cut'], J['dead']):>8s}"
              + (f"   ({later} guesses past it: not checked yet)" if later else ""))
        if a.detail:
            for m in J["missed"]:
                print(f"    missed: your rally {m[0]}-{m[1]}")
            for f in J["fake"]:
                print(f"    fake: {f[0]}-{f[1]}")
        for k, v in J.items():
            if isinstance(v, list):
                tot.setdefault(k, []).extend(v)
            elif isinstance(v, int):
                tot[k] = tot.get(k, 0) + v
    if not tot:
        return
    print(f"\nall: {tot['n_true']} rallies of yours, {tot['found']} found, {len(tot['missed'])} missed, "
          f"{len(tot['fake'])} fake ({tot['fake_rejected']} of them you rejected with X)")
    print(f"in play kept {_pct(tot['play_kept'], tot['play'])}, dead time cut {_pct(tot['dead_cut'], tot['dead'])}")
    print(_edges("start - your serve", tot["starts"], tot["accepted_start"]))
    print(_edges("end - your rally over", tot["ends"], tot["accepted_end"]))


if __name__ == "__main__":
    main()
