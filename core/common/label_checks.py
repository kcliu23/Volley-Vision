"""Frames in a clip's labels that deserve a second look (tools.review: b jumps there).

Built from YOUR labels (data/verified/<clip>.json), not from the autolabel proposals, so it works
on a clip you already finished. Each flag is a frame and a short reason. Nothing here changes a
label; it only points.

Ball (labels: frame -> [x, y] | "none" | "hidden" | "skip"):
    jump      the ball moves much farther between two frames than around them
    off arc   the point is far from a smooth curve through its neighbours (not near a touch)
    blink     one frame without a ball between frames with one (a missed click?)
    guessed   a long stretch of filled-in frames (i / I) right at a touch

Touches (contacts: frame -> type):
    gap       the same touch type two frames apart: one touch, or two? (a gap splits a touch)
    mixed     consecutive marked frames with different types
    no serve  a rally (ends with "rally over") with no serve mark before it
    no end    a serve with no "rally over" before the next serve
"""
from __future__ import annotations

import numpy as np

JUMP_X = 3.0            # [guess] a step this many times its neighbours' ...
JUMP_MIN_PX = 25.0      # [guess] ... and at least this long
ARC_PX = 10.0           # [guess] farther than this from the curve through 4 points on each side
TOUCH_NEAR = 4          # frames around a touch: the ball bends there, no arc check
GUESS_RUN = 8           # [guess] this many filled-in frames in a row at a touch


def _xy(labels: dict, f: int):
    v = labels.get(f)
    return np.asarray(v[:2], float) if isinstance(v, (list, tuple)) and len(v) >= 2 else None


def _times(vis: list[int], pts: dict) -> dict[int, float]:
    """Each visible frame's time in real frames. Screen recordings repeat a picture (the ball
    doesn't move: that frame takes no time) and then skip one to catch up (a double step: two
    frames of time). Without this, every repeat looks like a ball off its arc."""
    t, cur, owed = {}, 0.0, 0
    for i, f in enumerate(vis):
        if i:
            a = vis[i - 1]
            d = float(np.linalg.norm(pts[f] - pts[a]))
            gap = f - a
            near = [float(np.linalg.norm(pts[vis[j]] - pts[vis[j - 1]])) / max(vis[j] - vis[j - 1], 1)
                    for j in range(max(1, i - 3), min(len(vis), i + 4)) if j != i]
            ref = float(np.median(near)) if near else 0.0
            if gap == 1 and d < 0.5 and ref > 3.0:      # a repeated picture
                owed += 1
                cur += 0.0
            elif owed and gap == 1 and ref > 3.0 and d > 1.6 * ref:   # the catch-up skip
                owed -= 1
                cur += 2.0
            else:
                cur += gap
        t[f] = cur
    return t


def ball_flags(labels: dict, contacts: dict | None = None, src: dict | None = None) -> dict[int, str]:
    labels = {int(k): v for k, v in labels.items()}
    touch = sorted(int(k) for k in (contacts or {}))
    near_touch = lambda f: any(abs(f - t) <= TOUCH_NEAR for t in touch)
    out: dict[int, str] = {}
    frames = sorted(labels)
    pts = {f: _xy(labels, f) for f in frames}
    vis = [f for f in frames if pts[f] is not None]
    T = _times(vis, pts)
    repeat = {f for a, f in zip(vis, vis[1:]) if f - a == 1 and T[f] == T[a]}
    step = {}
    for a, b in zip(vis, vis[1:]):
        if b - a == 1:
            step[a] = float(np.linalg.norm(pts[b] - pts[a]))
    for a, s in step.items():
        nb = [step[g] for g in (a - 2, a - 1, a + 1, a + 2) if g in step]
        dt = max(T[a + 1] - T[a], 1.0)
        if len(nb) >= 2 and s / dt > JUMP_MIN_PX and s / dt > JUMP_X * max(np.median(nb), 1.0) and not near_touch(a):
            out[a + 1] = f"jump: {s:.0f} px from frame {a} (around: {np.median(nb):.0f} px)"
    for f in vis:
        if f in out or near_touch(f) or f in repeat:
            continue
        nb = [g for g in range(f - 4, f + 5) if g != f and pts.get(g) is not None and g not in repeat]
        if len(nb) < 6 or not (any(g < f for g in nb) and any(g > f for g in nb)):
            continue
        t = np.array([T[g] for g in nb]) - T[f]
        P = np.array([pts[g] for g in nb])
        A = np.c_[np.ones_like(t), t, t * t]
        coef, *_ = np.linalg.lstsq(A, P, rcond=None)
        r = float(np.linalg.norm(coef[0] - pts[f]))
        if r > ARC_PX:
            out[f] = f"off arc: {r:.0f} px from the curve through its neighbours"
    for f in frames:
        if labels[f] in ("none",) and pts.get(f - 1) is not None and pts.get(f + 1) is not None:
            out.setdefault(f, "blink: no ball here, a ball just before and after")
    if src:
        src = {int(k): v for k, v in src.items()}
        run = []
        for f in sorted(src) + [10 ** 9]:
            if src.get(f) == "interp" and (not run or f == run[-1] + 1):
                run.append(f)
                continue
            if len(run) >= GUESS_RUN and any(near_touch(g) for g in run):
                out.setdefault(run[0], f"guessed: {len(run)} filled-in frames ({run[0]}-{run[-1]}) at a touch")
            run = [f] if src.get(f) == "interp" else []
    return out


def touch_flags(contacts: dict) -> dict[int, str]:
    c = sorted((int(k), v) for k, v in contacts.items())
    out: dict[int, str] = {}
    for (f, a), (g, b) in zip(c, c[1:]):
        if g - f == 2 and a == b and a not in ("dead", "serve"):
            out[g] = f"gap: {a} at {f} and {g}, nothing at {f + 1}: one touch (mark {f + 1}) or two?"
        if g - f == 1 and a != b and "dead" not in (a, b):
            out[g] = f"mixed: {a} at {f}, {b} at {g} (one touch, one type)"
    serve_open = None
    for f, v in c:
        if v == "serve":
            if serve_open is not None and f - serve_open > 3:
                out.setdefault(f, f"no end: the serve at {serve_open} has no 'rally over' (9) before this serve")
            if serve_open is None or f - serve_open > 3:
                serve_open = f
        elif v == "dead":
            if serve_open is None and dict(c).get(f - 1) != "dead":   # a run of "rally over" frames is one end
                out.setdefault(f, "no serve: this rally has no serve mark (1) before it")
            serve_open = None
    return out


def all_flags(verified: dict) -> dict[int, str]:
    contacts = verified.get("contacts") or {}
    flags = ball_flags(verified.get("labels") or {}, contacts, verified.get("src"))
    for f, why in touch_flags(contacts).items():
        flags[f] = why if f not in flags else flags[f] + "; " + why
    return dict(sorted(flags.items()))
