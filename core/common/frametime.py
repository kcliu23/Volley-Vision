"""True capture time of each video frame, for screen recordings that repeat and skip frames.

A screen recording at 60 fps of a video that the phone shows a little unevenly has stamps exactly
1/60 s apart, but the pictures are not: now and then a frame is an exact repeat of the one before
(nothing moved), and a few frames later one moment is skipped (everything moved twice as far). In
between, the frames show the scene one frame late. For the 3D fit that is a timing error of
1/60 s on a third of the frames, which gravity can't explain: it shows up as a steady misfit of
several pixels on fast balls and gets "explained" with fake touches.

Read straight from the pictures, no ball needed: the mean change between consecutive frames
(whole picture, downscaled) is ~0 at a repeat and ~2x its local level at a skip. Each frame gets a
step from the one before: 0 (repeat), 2 (skip) or 1. A skip only counts next to a repeat (the
biggest change within 6 frames of it), so fast motion alone doesn't make one. Time of frame k =
(sum of steps up to k) / fps.

    T = frame_times("videos/test75.mov")       # (n,) in frame units; seconds = T / fps
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.scene3d.params3d import REPEAT, SKIP  # noqa: F401


def frame_changes(video: str, size=(480, 270)) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(str(video))
    prev, d = None, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size).astype(np.float32)
        d.append(np.nan if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    return np.array(d)


def _ratios(d: np.ndarray, win: int = 6) -> np.ndarray:
    """Each frame's change over the typical change around it (repeats left out of the typical)."""
    n = len(d)
    r = np.full(n, np.nan)
    for k in range(1, n):
        loc = d[max(1, k - win):min(n, k + win + 1)]
        loc = loc[np.isfinite(loc)]
        if not len(loc):
            continue
        ref = np.median(loc[loc > REPEAT * np.median(loc)])
        if ref > 0:
            r[k] = d[k] / ref
    return r


def steps_from_changes(d: np.ndarray, reach: int = 6) -> np.ndarray:
    """0 / 1 / 2 frame steps: each repeat is paired with the biggest change within `reach` frames
    after it (or before it, when the skip came first); a repeat with no clear skip near it is a
    real pause (time stands still for one frame)."""
    n = len(d)
    r = _ratios(d)
    steps = np.ones(n, int)
    steps[0] = 0
    rep = [k for k in range(1, n) if np.isfinite(r[k]) and r[k] < REPEAT]
    for k in rep:
        steps[k] = 0
    used = set(rep)
    for k in rep:
        best = None
        for side in (range(k + 1, min(n, k + reach + 1)), range(max(1, k - reach), k)):
            c = [(r[j], j) for j in side if j not in used and np.isfinite(r[j])]
            if c:
                m = max(c)
                if m[0] > SKIP and (best is None or m[0] > best[0]):
                    best = m
            if best is not None:
                break                                  # prefer the skip after the repeat
        if best is not None:
            steps[best[1]] = 2
            used.add(best[1])
    return steps


def frame_times(video: str, cache_dir: str | None = None) -> np.ndarray:
    """Frame index -> true time in frame units (float array); cached next to the clip's outputs."""
    cache = Path(cache_dir) / "frame_times.json" if cache_dir else None
    st = Path(video).stat()
    vkey = f"{st.st_size}"                                  # a re-exported clip under the same name
    if cache and cache.exists():
        c = json.load(open(cache))
        if c.get("video_key", vkey) == vkey:
            return np.array(c["t"], float)
    d = frame_changes(video)
    steps = steps_from_changes(d)
    T = np.cumsum(steps).astype(float)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"video_key": vkey, "t": T.tolist(), "repeats": int((steps == 0).sum() - 1), "skips": int((steps == 2).sum())},
                  open(cache, "w"))
    return T


BALL_MIN_STEP = 4.0      # px per frame: below this the ball's motion says nothing about timing
BALL_SKIP = 1.65         # a catch-up skip shows as a ball step this many times its neighbours'
BALL_NOSKIP = 1.35       # ... and a placed skip with a ball step under this shows none


def place_skips_by_ball(T: np.ndarray, uv: np.ndarray, w: np.ndarray, reach: int = 6, log=None) -> np.ndarray:
    """Move each catch-up skip to where the ball says it is. A skipped frame doubles the ball's
    step in the picture; the picture check alone sometimes puts the skip a few frames off
    (test20: repeat at 935/936, skip placed at 939->940, the ball's double step is 936->937; the
    flight 921-943 fitted 11 px off until moved). Only on a ball that moves clearly and steadily
    (detected points, steps on both sides of the new spot within 35% of each other), and only
    when the current spot shows no double step. T: frame -> time (frame units)."""
    T = np.asarray(T, float).copy()
    n = min(len(T), len(uv))
    ok = np.isfinite(uv[:n, 0]) & (w[:n] >= 1.0)
    step = np.full(n, np.nan)                            # step[k]: ball move from frame k to k+1
    both = ok[:-1] & ok[1:]
    step[:-1][both] = np.hypot(*(uv[1:n][both] - uv[:n - 1][both]).T)
    d = np.diff(T[:n])

    def ratio(k):                                        # step k (one frame) vs the nearest steps
        if not (0 <= k < n - 1) or not np.isfinite(step[k]):   # on each side (the ball may be
            return np.nan                                      # speeding up toward the camera)
        side = []
        for rng in (range(k - 1, max(-1, k - 4), -1), range(k + 1, min(n - 1, k + 4))):
            for j in rng:
                if np.isfinite(step[j]) and d[j] > 0:
                    side.append(step[j] / d[j])
                    break
        if not side:                                     # (one side is enough: the ball may
            return np.nan                                # just have left the picture / a hand)
        ref = float(np.mean(side))
        return step[k] / ref if ref >= BALL_MIN_STEP else np.nan

    def near(k, dirn):                                   # the nearest real one-frame step that side
        j = k + dirn
        while 0 <= j < n - 1 and abs(j - k) <= 3:
            if np.isfinite(step[j]) and d[j] > 0:
                return step[j] / d[j]
            j += dirn
        return np.nan

    moved = 0
    for k in np.where(np.isclose(d, 2))[0]:
        rk = ratio(k)
        # the skip's own step should be ~2x; under BALL_NOSKIP x it the skip isn't here
        if not np.isfinite(rk) or rk >= BALL_NOSKIP:
            continue
        best = None
        for g in range(max(0, k - reach), min(n - 1, k + reach + 1)):
            if g == k or not np.isclose(d[g], 1):
                continue
            rg = ratio(g)
            if not np.isfinite(rg) or rg < BALL_SKIP or rg > 2.6:
                continue
            a_, b_ = near(g, -1), near(g, +1)
            if not (np.isfinite(a_) and np.isfinite(b_)) or abs(a_ - b_) > 0.35 * max(a_, b_):
                continue                                 # speed changing here: a touch, not a skip
            if best is None or abs(rg - 2) < abs(best[1] - 2):
                best = (g, rg)
        if best is None:
            continue
        g = best[0]
        if g < k:
            T[g + 1:k + 1] += 1
        else:
            T[k + 1:g + 1] -= 1
        d = np.diff(T[:n])
        moved += 1
        if log:
            log(f"timing: skip moved from {k}->{k + 1} to {g}->{g + 1} (the ball's double step)")
    return T


def summary(T: np.ndarray) -> str:
    st = np.diff(T)
    late = np.arange(len(T)) - T
    return (f"{int((st == 0).sum())} repeated and {int((st == 2).sum())} skipped frames; "
            f"{(np.abs(late - np.median(late)) > 0.5).mean():.0%} of frames are shown off-time")
