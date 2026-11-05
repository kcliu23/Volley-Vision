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

REPEAT = 0.15      # change below this x local level = the same picture again
SKIP = 1.3         # change above this x local level, next to a repeat = the skipped moment


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
    if cache and cache.exists():
        return np.array(json.load(open(cache))["t"], float)
    d = frame_changes(video)
    steps = steps_from_changes(d)
    T = np.cumsum(steps).astype(float)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"t": T.tolist(), "repeats": int((steps == 0).sum() - 1), "skips": int((steps == 2).sum())},
                  open(cache, "w"))
    return T


def summary(T: np.ndarray) -> str:
    st = np.diff(T)
    late = np.arange(len(T)) - T
    return (f"{int((st == 0).sum())} repeated and {int((st == 2).sum())} skipped frames; "
            f"{(np.abs(late - np.median(late)) > 0.5).mean():.0%} of frames are shown off-time")
