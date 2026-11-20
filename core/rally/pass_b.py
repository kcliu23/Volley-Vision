"""Rally pass B, v1 (EXPERIMENTAL): in play or dead, per half second, learned from your marks.

notes/ANALYSIS_DESIGN.md section 2, roadmap 2.2. Features per WINDOW_S window, from the 2D ball
(track.json + the camera's net tape) and the players (pose.json), each also averaged over the
seconds around it (play is a few seconds long; one window alone can't tell a toss from a set):

  ball    seen, above the tape, highest point over the tape, speed, up/down turns (arcs)
  people  how many, how fast they move (box centre / box height per s: size-free)

A gradient-boosted tree gives P(in play) per window; runs above 0.5 become rallies (short gaps
joined, short runs dropped). Scored by evaluation.rally_v1 against the rule (core/scene3d/rallies.py)
with evaluation.rallies_eval.judge. Not wired into main.py: it has to beat the rule on a full game
it never saw first, and only one full game is labeled (Nov 20).

Pose caveat: on full games pose only runs where pass A thought play might be, so "no people"
partly means "pass A said dead". That leaks the pipeline's own guess into the features; the people
features are NaN (unknown) there, not zero, so the trees can't read "no pose" as "dead" directly.
"""
from __future__ import annotations

import numpy as np

from core.rally.pass_a import tape_y

WINDOW_S = 0.5           # s: one row per window
CONTEXT_S = (1.0, 3.0)   # s: each feature also averaged over +- these (centred)
JOIN_GAP_S = 1.0         # [guess] in-play runs this close are one rally (a ball out of the picture)
MIN_RALLY_S = 2.0        # [guess] shorter runs aren't rallies (the shortest marked: 157 frames)
SNAP_S = 2.0             # [guess] a rule edge this close to a v1 edge replaces it (picked once, not tuned)

BALL = ["seen", "high", "over_tape", "speed", "turns"]
PEOPLE = ["people", "move_med", "move_p90"]


def _ball_rows(pos: np.ndarray, fps: float, cam, H: float, w: int) -> np.ndarray:
    n = len(pos)
    ok = np.isfinite(pos[:, 0])
    over = np.full(n, np.nan)
    if cam is not None and ok.any():
        over[ok] = (tape_y(cam, pos[ok, 0]) - pos[ok, 1]) / H        # + above the tape, in picture heights
    sp = np.full(n, np.nan)
    sp[1:] = np.hypot(*(pos[1:] - pos[:-1]).T) * fps / H             # picture heights per s
    vy = np.full(n, np.nan)
    vy[1:] = pos[1:, 1] - pos[:-1, 1]
    rows = []
    for a in range(0, n, w):
        b = min(n, a + w)
        o, s, v = over[a:b], sp[a:b], vy[a:b]
        seen = ok[a:b].mean()
        sv = np.sign(v[np.isfinite(v)])
        rows.append([seen,
                     np.nanmean(o > 0) if np.isfinite(o).any() else np.nan,
                     np.nanmax(o) if np.isfinite(o).any() else np.nan,
                     np.nanmedian(s) if np.isfinite(s).any() else np.nan,
                     float(np.sum(sv[1:] != sv[:-1])) if len(sv) > 1 else np.nan])
    return np.array(rows, float)


def _people_rows(frames: dict | None, n: int, fps: float, w: int) -> np.ndarray:
    rows = np.full((-(-n // w), len(PEOPLE)), np.nan)
    if not frames:
        return rows
    F = {int(k): v for k, v in frames.items()}
    for i, a in enumerate(range(0, n, w)):
        ks = [f for f in range(a, min(n, a + w)) if f in F]
        if not ks:
            continue                                                  # pose didn't run here: unknown
        rows[i, 0] = np.mean([len(F[f]) for f in ks])
        seen: dict = {}
        for f in ks:
            for p in F[f]:
                x0, y0, x1, y1 = p["box"]
                seen.setdefault(p["id"], []).append((f, (x0 + x1) / 2, (y0 + y1) / 2, max(y1 - y0, 1.0)))
        mv = []
        for track in seen.values():
            if len(track) >= 2:
                (fa, xa, ya, ha), (fb, xb, yb, hb) = track[0], track[-1]
                if fb > fa:
                    mv.append(np.hypot(xb - xa, yb - ya) / ((ha + hb) / 2) / ((fb - fa) / fps))
        if mv:
            rows[i, 1], rows[i, 2] = np.median(mv), np.percentile(mv, 90)
    return rows


def _context(X: np.ndarray, k: int) -> np.ndarray:
    """Each column averaged over +-k rows (NaN-aware)."""
    out = np.full_like(X, np.nan)
    for i in range(len(X)):
        blk = X[max(0, i - k):i + k + 1]
        with np.errstate(all="ignore"):
            cnt = np.isfinite(blk).sum(0)
            s = np.nansum(blk, 0)
        out[i] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out


def features(pos: np.ndarray, fps: float, cam, H: float, pose_frames: dict | None) -> tuple[np.ndarray, list[str]]:
    """pos: (n, 2) ball per frame (NaN: not seen). Returns (rows, names): one row per WINDOW_S."""
    w = max(1, int(round(WINDOW_S * fps)))
    base = np.hstack([_ball_rows(pos, fps, cam, H, w), _people_rows(pose_frames, len(pos), fps, w)])
    names = BALL + PEOPLE
    cols, out_names = [base], list(names)
    for c in CONTEXT_S:
        cols.append(_context(base, int(round(c / WINDOW_S))))
        out_names += [f"{x}_pm{c:g}s" for x in names]
    return np.hstack(cols), out_names


def window_labels(play: set[int], n: int, fps: float) -> np.ndarray:
    """1 where at least half the window's frames are in play (your marks)."""
    w = max(1, int(round(WINDOW_S * fps)))
    m = np.zeros(n, bool)
    m[[f for f in play if 0 <= f < n]] = True
    return np.array([m[a:a + w].mean() >= 0.5 for a in range(0, n, w)], int)


def to_rallies(p: np.ndarray, fps: float, n: int, thr: float = 0.5) -> list[tuple[int, int]]:
    """P(in play) per window -> [(f0, f1)] frames: runs above thr, gaps <= JOIN_GAP_S joined,
    runs < MIN_RALLY_S dropped."""
    w = max(1, int(round(WINDOW_S * fps)))
    on = p >= thr
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    joined: list[list[int]] = []
    gap = int(round(JOIN_GAP_S / WINDOW_S))
    for a, b in runs:
        if joined and a - joined[-1][1] - 1 <= gap:
            joined[-1][1] = b
        else:
            joined.append([a, b])
    out = [(a * w, min(n - 1, (b + 1) * w - 1)) for a, b in joined]
    return [(a, b) for a, b in out if (b - a + 1) / fps >= MIN_RALLY_S]


def snap_edges(v1: list[tuple[int, int]], rule: list[tuple[int, int]], fps: float) -> list[tuple[int, int]]:
    """v1 decides WHERE play is, the rule (3D serve touch, floor contact) decides the exact edges:
    each v1 rally's start / end moves to the overlapping rule rallies' first start / last end when
    that is within SNAP_S (design section 2: "edges snapped to the serve touch / ball landing")."""
    snap = int(round(SNAP_S * fps))
    out = []
    for a, b in v1:
        ov = [r for r in rule if min(b, r[1]) > max(a, r[0])]
        if ov:
            r0, r1 = min(r[0] for r in ov), max(r[1] for r in ov)
            a = r0 if abs(r0 - a) <= snap else a
            b = r1 if abs(r1 - b) <= snap else b
        out.append((a, b))
    return out
