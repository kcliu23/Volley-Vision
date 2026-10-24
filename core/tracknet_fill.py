"""Use TrackNet only where the YOLO track has no observation, guided by physics.

Runs after the linker. TrackNet never overrides a YOLO detection; it only fills:

  1. gaps   frames between two observed stretches (at most max_gap_s apart): the local arc
            through the observed points on both sides predicts where the ball is; a TrackNet
            peak >= peak_gap within the corridor around that prediction is taken
  2. ends   outward from the start/end of every observed stretch (the track often starts
            late or stops early while the ball is still visible): step frame by frame,
            predict from the last observed points, take a TrackNet peak >= peak_ext inside
            the corridor, stop after ext_miss_s without one

The corridor radius is r_bw ball-widths, growing with time since the last observation.
Settings are in seconds and ball-widths. Measured on the 10 labelled clips
(tools.eval_gt, rule linker 95.3% on-ball):
    peak_ext 0.4  -> 96.5% on-ball, false ball when hidden 8.0% -> 10.4%
    peak_ext 0.2  -> 97.6% on-ball, false ball when hidden 17.8%
Frames filled here get state "tracknet" (an observation, like "weak").
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

TRACKNET = "tracknet"
OBSERVED = ("detected", "weak", TRACKNET)


@dataclass
class FillParams:
    peak_gap: float = 0.5      # TrackNet peak needed inside a two-sided gap
    peak_ext: float = 0.4      # ... when extending a track end (only one side known)
    r_bw: float = 1.0          # corridor radius, ball-widths, right next to an observation
    gap_grow_s: float = 9.0    # gap corridor grows by this fraction per second from the nearest side
    ext_grow_s: float = 18.0   # end corridor grows by this fraction per second of misses
    max_gap_s: float = 0.5     # longest two-sided gap filled
    ext_miss_s: float = 0.04   # end extension stops after this long without a TrackNet peak
    support_s: float = 0.15    # observed points used for a prediction must be within this
    edge_bw: float = 1.0       # ignore TrackNet peaks this close to the picture edge (blob cut off, centre biased)


def _predict(fs, xs, ys, f):
    fs = np.asarray(fs, float)
    if len(fs) < 2:
        return xs[-1], ys[-1]
    deg = 2 if len(fs) >= 4 and np.ptp(fs) >= 3 else 1
    return float(np.polyval(np.polyfit(fs, xs, 1), f)), float(np.polyval(np.polyfit(fs, ys, deg), f))


def _nearest(cands, px, py, peak_min, r):
    best = None
    for c in cands:
        if c[4] < peak_min:
            continue
        d = math.hypot(c[0] - px, c[1] - py)
        if d <= r and (best is None or d < best[0]):
            best = (d, c)
    return best[1] if best else None


def apply(track, tn_frames: list, ball_px: float, params: FillParams | None = None) -> dict:
    """Fill track in place from TrackNet candidates. Returns counts {"gap": n, "end": n}."""
    P = params or FillParams()
    n = len(track.pos)
    fps = track.fps or 60.0
    if not tn_frames or len(tn_frames) < n:
        return {"gap": 0, "end": 0}
    dup = track.dup if track.dup and len(track.dup) == n else [False] * n
    x0, y0, x1, y1 = track.content or (0, 0, track.width, track.height)
    m = P.edge_bw * ball_px
    inside = lambda c: x0 + m <= c[0] <= x1 - m and y0 + m <= c[1] <= y1 - m
    G = max(2, int(round(P.max_gap_s * fps)))
    ext_miss = max(1, int(round(P.ext_miss_s * fps)))
    sup_span = max(4, int(round(P.support_s * fps)))
    if not track.size or len(track.size) != n:
        track.size = [None] * n
    counts = {"gap": 0, "end": 0}

    def observed(i):
        return track.pos[i] is not None and track.state[i] in OBSERVED

    def take(i, c):
        track.pos[i] = (float(c[0]), float(c[1]))
        track.state[i] = TRACKNET
        track.conf[i] = float(c[4])
        track.size[i] = float(c[2])

    # 1) two-sided gaps
    obs = [i for i in range(n) if observed(i) and not dup[i]]
    for k in range(len(obs) - 1):
        a, b = obs[k], obs[k + 1]
        if b - a <= 1 or b - a > G:
            continue
        L = obs[max(0, k - 2):k + 1]
        R = obs[k + 1:k + 4]
        fs = L + R
        xs = [track.pos[j][0] for j in fs]
        ys = [track.pos[j][1] for j in fs]
        for i in range(a + 1, b):
            if dup[i] or observed(i):
                continue
            px, py = _predict(fs, xs, ys, i)
            r = P.r_bw * ball_px * (1 + P.gap_grow_s * min(i - a, b - i) / fps)
            c = _nearest([c for c in tn_frames[i] if inside(c)], px, py, P.peak_gap, r)
            if c is not None:
                take(i, c)
                counts["gap"] += 1

    # 2) extend outward from the ends of observed stretches
    for direction in (+1, -1):
        ends = [i for i in range(n) if observed(i) and not dup[i]
                and not (0 <= i + direction < n and observed(i + direction))]
        for e in ends:
            f, miss = e + direction, 0
            while 0 <= f < n and miss <= ext_miss and not observed(f):
                if dup[f]:
                    f += direction
                    continue
                sup = [j for j in range(f - direction, f - direction * (sup_span + 1), -direction)
                       if 0 <= j < n and observed(j) and not dup[j]][:6]
                if len(sup) < 2:
                    break
                px, py = _predict(sup, [track.pos[j][0] for j in sup], [track.pos[j][1] for j in sup], f)
                r = P.r_bw * ball_px * (1 + P.ext_grow_s * miss / fps)
                c = _nearest([c for c in tn_frames[f] if inside(c)], px, py, P.peak_ext, r)
                if c is not None:
                    take(f, c)
                    counts["end"] += 1
                    miss = 0
                else:
                    miss += 1
                f += direction

    # repeated frames show the same image: copy the previous frame's TrackNet position
    for i in range(1, n):
        if dup[i] and track.state[i - 1] == TRACKNET and not observed(i):
            track.pos[i], track.state[i], track.conf[i] = track.pos[i - 1], TRACKNET, track.conf[i - 1]
            track.size[i] = track.size[i - 1]
    return counts
