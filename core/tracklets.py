"""Tracklets: short, confident chains of points, one per object, before deciding which object
is the ball in play.

The joint linker decides frame by frame with two points of memory, so when the rally ball's
detections fade it can slide onto another ball (test34: a second ball on the far court) and
never come back. Splitting the problem helps:

  1. tracklets   (this file) link calibrated points (core.joint_linker.build_points) into
                 chains that obey simple physics: each chain predicts its next position from
                 its recent points (with gravity); a point joins only inside a tight gate and
                 with a consistent size. Each chain is one object for a stretch of time.
  2. association (core.tracklet_linker) chooses the sequence of tracklets that is the rally
                 ball, using whole-tracklet evidence and how plausibly one hands over to the next.

Tracklet fields: frames, xs, ys, sizes (None = unknown), logits, states, plus summary features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class TrackletParams:
    gate_bw: float = 0.9          # join radius around the prediction, ball-widths ...
    gate_speed: float = 0.5       # ... + this fraction of the per-frame speed (timing jitter)
    gate_miss: float = 0.5        # ... + this many ball-widths per missed frame
    size_tol: float = 0.35        # |log size ratio| allowed between consecutive known sizes ...
    size_tol_miss: float = 0.05   # ... + this per missed frame
    max_miss_s: float = 0.1       # a tracklet ends after this long without a point
    start_p: float = 0.3          # a new tracklet needs a point at least this likely
    min_len: int = 3              # shorter tracklets are dropped ...
    keep_strong_p: float = 0.8    # ... unless they contain a point this likely
    fit_pts: int = 5


@dataclass
class Tracklet:
    tid: int
    frames: list = field(default_factory=list)
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    logits: list = field(default_factory=list)
    states: list = field(default_factory=list)

    @property
    def f0(self):
        return self.frames[0]

    @property
    def f1(self):
        return self.frames[-1]

    def add(self, f, pt):
        x, y, s, lg, st, p = pt
        self.frames.append(f); self.xs.append(x); self.ys.append(y)
        self.sizes.append(s); self.logits.append(lg); self.states.append(st)

    def size(self, end: int = -1, k: int = 5):
        """Median known size near the start (end=0) or the end (end=-1); None if unknown."""
        s = [v for v in (self.sizes[:k] if end == 0 else self.sizes[-k:]) if v is not None]
        if not s:
            s = [v for v in self.sizes if v is not None]
        return float(np.median(s)) if s else None

    def predict(self, f, gy: float, k: int = 5):
        """Position at frame f from the last (or, for f before the start, first) k points:
        constant horizontal velocity, gravity vertically."""
        if f >= self.f1:
            F, X, Y = self.frames[-k:], self.xs[-k:], self.ys[-k:]
        else:
            F, X, Y = self.frames[:k], self.xs[:k], self.ys[:k]
        if len(F) == 1:
            return X[0], Y[0], 0.0
        F = np.asarray(F, float); X = np.asarray(X, float); Y = np.asarray(Y, float)
        vx = np.polyfit(F, X, 1)
        # y = a + b f + gy/2 f^2  (gravity known per clip): fit a, b on y - gy/2 f^2
        cy = np.polyfit(F, Y - 0.5 * gy * F ** 2, 1)
        px = float(np.polyval(vx, f))
        py = float(np.polyval(cy, f) + 0.5 * gy * f ** 2)
        speed = math.hypot(vx[0], cy[0] + gy * F[-1])
        return px, py, speed

    def summary(self, ball: float, fps: float) -> dict:
        n = len(self.frames)
        F = np.asarray(self.frames, float)
        X = np.asarray(self.xs, float); Y = np.asarray(self.ys, float)
        dur = (F[-1] - F[0]) / fps if n > 1 else 0.0
        path = float(np.sum(np.hypot(np.diff(X), np.diff(Y)))) / ball if n > 1 else 0.0
        return {"n": n, "f0": self.f0, "f1": self.f1, "dur_s": round(dur, 3),
                "evidence": round(float(np.sum(self.logits)), 2),
                "mean_p": round(float(np.mean(1 / (1 + np.exp(-np.asarray(self.logits))))), 3),
                "speed_bw_s": round(path / dur, 1) if dur > 0 else 0.0,
                "size": self.size(), "yolo_frac": round(sum(s != "tracknet" for s in self.states) / n, 2)}


def build(pts_by_frame: list, fps: float, ball: float, gy: float = 0.0, dup=None,
          params: TrackletParams | None = None) -> list[Tracklet]:
    """pts_by_frame[f] = [(x, y, size|None, logit, state, p), ...] (core.joint_linker.build_points)."""
    from scipy.optimize import linear_sum_assignment
    P = params or TrackletParams()
    n = len(pts_by_frame)
    dup = dup if dup is not None and len(dup) == n else [False] * n
    max_miss = max(2, int(round(P.max_miss_s * fps)))
    active: list[Tracklet] = []
    done: list[Tracklet] = []
    nid = 0
    for f in range(n):
        if dup[f]:
            continue
        pts = pts_by_frame[f]
        # assignment cost: distance to prediction in units of the gate; inf outside gate / size mismatch
        if active and pts:
            C = np.full((len(active), len(pts)), 1e6)
            for i, tr in enumerate(active):
                miss = f - tr.f1 - 1
                px, py, sp = tr.predict(f, gy, P.fit_pts)
                ts = tr.size()
                gate = ball * (P.gate_bw + P.gate_miss * miss) + P.gate_speed * sp
                for j, (x, y, s, lg, st, p) in enumerate(pts):
                    d = math.hypot(x - px, y - py)
                    if d > gate:
                        continue
                    if s is not None and ts is not None and abs(math.log(s / ts)) > P.size_tol + P.size_tol_miss * miss:
                        continue
                    C[i, j] = d / gate - 0.1 * lg          # closer and more likely is better
            ri, ci = linear_sum_assignment(C)
            used = set()
            for i, j in zip(ri, ci):
                if C[i, j] < 1e5:
                    active[i].add(f, pts[j]); used.add(j)
        else:
            used = set()
        for j, pt in enumerate(pts):
            if j not in used and pt[5] >= P.start_p:
                tr = Tracklet(nid); nid += 1
                tr.add(f, pt)
                active.append(tr)
        still = []
        for tr in active:
            (still if f - tr.f1 <= max_miss else done).append(tr)
        active = still
    done += active
    keep = [t for t in done if len(t.frames) >= P.min_len or
            max(1 / (1 + math.exp(-lg)) for lg in t.logits) >= P.keep_strong_p]
    keep.sort(key=lambda t: t.f0)
    for i, t in enumerate(keep):
        t.tid = i
    return keep
