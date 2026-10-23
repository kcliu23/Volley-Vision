"""Step 2 of the two-pass tracker: turn raw YOLO candidates into one ball path.

Input: every candidate per frame (from tools.dump_detections, conf >= ~0.05).
Output: one state per frame — detected / weak / filled / none — plus position.

Works on the whole clip at once (offline), in four stages:

  a. clean     Drop repeated video frames (identical candidates to the previous
               frame) and boxes whose size is implausible for this clip's ball.
  b. backbone  Link STRONG candidates (conf >= strong_conf) into tracklets with a
               gravity-aware (quadratic) prediction gate. Overlapping tracklets
               compete — there is one ball — and the better-supported one wins.
  c. extend    Walk outward from every tracklet, forwards and backwards, one
               frame at a time: predict the next position from the last few
               accepted points, accept any candidate (weak or strong) close to
               the prediction. Weak candidates are only ever accepted this way —
               they need the arc to vouch for them.
  d. bridge    Short gaps between accepted points (ball still on screen) are
               filled with a cubic Hermite curve through the two endpoints using
               each side's local velocity. For free flight this reproduces the
               parabola; across a touch it's a smooth best guess.

Positions are pixel coords in the original frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

DETECTED, WEAK, FILLED, NONE = "detected", "weak", "filled", "none"


@dataclass
class LinkParams:
    strong_conf: float = 0.40     # can seed / anchor a track on its own
    min_conf: float = 0.05        # ignore anything below (should match the dump)
    size_lo: float = 0.4          # box size must be within [lo, hi] x clip median
    size_hi: float = 2.5
    tol_size: float = 0.8         # acceptance radius = tol_size * ball size ...
    tol_min_px: float = 14.0      # ... at least this ...
    tol_speed: float = 0.6        # ... + this * per-frame motion (timing jitter)
    gate_growth: float = 0.5      # radius grows by this fraction per missed frame
    max_miss_s: float = 0.15      # extension/linking gives up after this long without a hit
    fit_pts: int = 6              # points used for local prediction
    min_tracklet: int = 3         # strong tracklets shorter than this are dropped
    max_bridge_s: float = 0.5     # longest gap filled by interpolation
    edge_margin: float = 0.0      # bridged path must stay this far inside the frame (px)


@dataclass
class Track:
    fps: float
    width: int
    height: int
    state: list[str]
    pos: list[tuple[float, float] | None]
    conf: list[float | None]
    dup: list[bool] = field(default_factory=list)

    def summary(self) -> dict:
        from collections import Counter
        c = Counter(self.state)
        n = len(self.state)
        return {"frames": n, **{k: c.get(k, 0) for k in (DETECTED, WEAK, FILLED, NONE)},
                "coverage": round(1 - c.get(NONE, 0) / max(n, 1), 3)}

    def to_rows(self) -> list[dict]:
        rows = []
        for i, (s, p, c) in enumerate(zip(self.state, self.pos, self.conf)):
            r = {"f": i, "state": s}
            if p is not None:
                r["pos"] = [round(p[0], 1), round(p[1], 1)]
            if c is not None:
                r["conf"] = round(c, 3)
            if self.dup and self.dup[i]:
                r["dup"] = True
            rows.append(r)
        return rows


# ── helpers ──────────────────────────────────────────────────────────────────

def _predict(ts, xs, ys, t):
    """Quadratic (or lower order with few points) extrapolation to time t."""
    ts = np.asarray(ts, float)
    deg = 2 if len(ts) >= 4 else (1 if len(ts) >= 2 else 0)
    if deg == 0:
        return xs[0], ys[0], 0.0
    span = ts.max() - ts.min()
    if deg == 2 and span < 3:          # too short a baseline for curvature
        deg = 1
    px = np.polyfit(ts - t, xs, deg)
    py = np.polyfit(ts - t, ys, deg)
    vx = px[-2] if deg >= 1 else 0.0
    vy = py[-2] if deg >= 1 else 0.0
    return float(px[-1]), float(py[-1]), float(np.hypot(vx, vy))


def _velocity(ts, xs, ys, t):
    """Local velocity (px/frame) at time t from a quadratic/linear fit."""
    ts = np.asarray(ts, float)
    deg = 2 if len(ts) >= 4 and (ts.max() - ts.min()) >= 3 else 1
    if len(ts) < 2:
        return 0.0, 0.0
    px = np.polyfit(ts - t, xs, deg)
    py = np.polyfit(ts - t, ys, deg)
    return float(px[-2]), float(py[-2])


# ── main entry ───────────────────────────────────────────────────────────────

def link(frames: list, fps: float, width: int, height: int, params: LinkParams | None = None,
         dup_flags: list[bool] | None = None) -> Track:
    """frames: list (one per video frame) of candidate lists [cx, cy, w, h, conf].

    dup_flags: per-frame "this video frame repeats the previous image" (from the
    detection pass, image-based). Without it, repeats are guessed from identical
    candidate positions, which misses repeats where nothing was detected."""
    P = params or LinkParams()
    n = len(frames)
    C = [np.asarray(f, float).reshape(-1, 5) for f in frames]
    C = [c[c[:, 4] >= P.min_conf] for c in C]

    # a. clean ────────────────────────────────────────────────────────────────
    if dup_flags is not None and len(dup_flags) == n:
        dup = [bool(x) for x in dup_flags]
    else:
        dup = [False] * n
        for i in range(1, n):
            a, b = C[i - 1], C[i]
            if len(a) and len(b) and a.shape == b.shape and np.allclose(a[:, :2], b[:, :2]):
                dup[i] = True
    strong_sizes = np.concatenate([np.maximum(c[:, 2], c[:, 3])[c[:, 4] >= P.strong_conf] for c in C] or [[]])
    med = float(np.median(strong_sizes)) if len(strong_sizes) else 30.0
    for i in range(n):
        s = np.maximum(C[i][:, 2], C[i][:, 3])
        C[i] = C[i][(s >= P.size_lo * med) & (s <= P.size_hi * med)]
    usable = [i for i in range(n) if not dup[i]]
    max_miss = max(2, int(round(P.max_miss_s * fps)))

    def tol(size, speed, missed=0):
        r = max(P.tol_size * size, P.tol_min_px) + P.tol_speed * speed
        return r * (1 + P.gate_growth * missed)

    # b. backbone from strong candidates ──────────────────────────────────────
    tracklets: list[dict] = []           # {"pts": {frame: (x, y, size, conf)}}
    alive: list[dict] = []
    for i in usable:
        strong = [c for c in C[i] if c[4] >= P.strong_conf]
        claimed = set()
        # each alive tracklet claims its nearest strong candidate within the gate
        for tr in sorted(alive, key=lambda t: -len(t["pts"])):
            fs = sorted(tr["pts"])[-P.fit_pts:]
            xs = [tr["pts"][f][0] for f in fs]
            ys = [tr["pts"][f][1] for f in fs]
            px, py, sp = _predict(fs, xs, ys, i)
            missed = i - fs[-1] - 1
            best, bd = None, None
            for k, c in enumerate(strong):
                if k in claimed:
                    continue
                d = np.hypot(c[0] - px, c[1] - py)
                if d <= tol(max(c[2], c[3]), sp, missed) * 1.5 and (bd is None or d < bd):
                    best, bd = k, d
            if best is not None:
                c = strong[best]
                tr["pts"][i] = (c[0], c[1], max(c[2], c[3]), c[4])
                claimed.add(best)
        for k, c in enumerate(strong):
            if k not in claimed:
                tr = {"pts": {i: (c[0], c[1], max(c[2], c[3]), c[4])}}
                tracklets.append(tr)
                alive.append(tr)
        alive = [t for t in alive if i - max(t["pts"]) <= max_miss]
    tracklets = [t for t in tracklets if len(t["pts"]) >= P.min_tracklet]

    # resolve overlaps: one ball. Higher support (sum of conf) claims its frames first.
    tracklets.sort(key=lambda t: -sum(p[3] for p in t["pts"].values()))
    owner: dict[int, tuple] = {}         # frame -> (x, y, size, conf, state)
    for t in tracklets:
        span = range(min(t["pts"]), max(t["pts"]) + 1)
        if any(f in owner for f in span):
            # keep only the part that doesn't collide with a stronger tracklet
            free = {f: p for f, p in t["pts"].items() if f not in owner}
            if len(free) < P.min_tracklet:
                continue
            t["pts"] = free
        for f, p in t["pts"].items():
            owner[f] = (*p, DETECTED)

    # c. extend outward, accepting weak candidates that the arc vouches for ──────
    def extend(direction):
        changed = False
        anchors = sorted(owner)
        for f0 in (anchors if direction > 0 else anchors[::-1]):
            # only start from the end of a run
            nxt = f0 + direction
            if nxt in owner:
                continue
            missed = 0
            f = nxt
            while 0 <= f < n and missed <= max_miss and f not in owner:
                if dup[f]:
                    f += direction
                    continue
                near = [g for g in sorted(owner, key=lambda g: abs(g - f)) if abs(g - f) <= 3 * max_miss][:P.fit_pts]
                # only use support on the side we came from (don't predict across the gap)
                near = [g for g in near if (g - f) * direction < 0]
                if len(near) < 2:
                    break
                xs = [owner[g][0] for g in near]
                ys = [owner[g][1] for g in near]
                sz = float(np.median([owner[g][2] for g in near]))
                px, py, sp = _predict(near, xs, ys, f)
                r = tol(sz, sp, missed)
                best, bd = None, None
                for c in C[f]:
                    d = np.hypot(c[0] - px, c[1] - py)
                    if d <= r and (bd is None or d < bd):
                        best, bd = c, d
                if best is not None:
                    st = DETECTED if best[4] >= P.strong_conf else WEAK
                    owner[f] = (best[0], best[1], max(best[2], best[3]), best[4], st)
                    changed = True
                    missed = 0
                else:
                    missed += 1
                f += direction
        return changed

    for _ in range(4):                    # a few rounds: forward/backward feed each other
        a = extend(+1)
        b = extend(-1)
        if not (a or b):
            break

    # d. bridge short on-screen gaps ──────────────────────────────────────────
    state = [NONE] * n
    pos: list = [None] * n
    conf: list = [None] * n
    for f, (x, y, s, c, st) in owner.items():
        state[f], pos[f], conf[f] = st, (x, y), c

    # repeated frames show the same image as the frame before — copy it
    for i in range(1, n):
        if dup[i] and state[i] == NONE and state[i - 1] != NONE:
            state[i], pos[i], conf[i] = state[i - 1], pos[i - 1], conf[i - 1]

    keys = sorted(owner)
    max_gap = int(round(P.max_bridge_s * fps))
    for a, b in zip(keys, keys[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > max_gap:
            continue
        left = [g for g in keys if a - 3 * max_miss <= g <= a][-P.fit_pts:]
        right = [g for g in keys if b <= g <= b + 3 * max_miss][:P.fit_pts]
        if len(left) < 2 or len(right) < 2:
            continue
        # Ball touching the frame edge at either end of the gap may have left the
        # frame (and its box is clipped, so the position is biased): don't bridge.
        def at_edge(g):
            x, y, s = owner[g][0], owner[g][1], owner[g][2]
            return x - s / 2 <= 2 or y - s / 2 <= 2 or x + s / 2 >= width - 2 or y + s / 2 >= height - 2
        if gap > 3 and (at_edge(a) or at_edge(b)):
            continue
        vxa, vya = _velocity(left, [owner[g][0] for g in left], [owner[g][1] for g in left], a)
        vxb, vyb = _velocity(right, [owner[g][0] for g in right], [owner[g][1] for g in right], b)
        xa, ya = owner[a][0], owner[a][1]
        xb, yb = owner[b][0], owner[b][1]
        T = b - a
        lx, ly = [owner[g][0] for g in left], [owner[g][1] for g in left]
        rx, ry = [owner[g][0] for g in right], [owner[g][1] for g in right]
        L = {f: _predict(left, lx, ly, f)[:2] for f in range(a + 1, b + 1)}
        R = {f: _predict(right, rx, ry, f)[:2] for f in range(a, b)}
        sz = float(np.median([owner[g][2] for g in left + right]))
        # Same arc on both sides? (left arc lands on b and right arc lands on a)
        same_arc = (np.hypot(L[b][0] - xb, L[b][1] - yb) <= 1.5 * sz and
                    np.hypot(R[a][0] - xa, R[a][1] - ya) <= 1.5 * sz)
        path = []
        if not same_arc:
            # A touch happened inside the gap: follow the incoming arc forward and
            # the outgoing arc backward, switching where they come closest.
            ks = list(range(a + 1, b))
            dists = [np.hypot(L[k][0] - R[k][0], L[k][1] - R[k][1]) for k in ks]
            k = ks[int(np.argmin(dists))]
            if min(dists) <= 2.0 * sz:
                path = [(f, *(L[f] if f <= k else R[f])) for f in range(a + 1, b)]
        if not path:
            for f in range(a + 1, b):
                u = (f - a) / T
                h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
                h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
                x = h00 * xa + h10 * T * vxa + h01 * xb + h11 * T * vxb
                y = h00 * ya + h10 * T * vya + h01 * yb + h11 * T * vyb
                path.append((f, x, y))
        m = P.edge_margin
        if any(not (m <= x <= width - m and m <= y <= height - m) for _, x, y in path):
            continue                       # ball would leave the frame: don't invent it
        for f, x, y in path:
            if state[f] == NONE:
                state[f], pos[f] = FILLED, (x, y)
    # a repeated frame right after a filled one shows the same image too
    for i in range(1, n):
        if dup[i] and state[i - 1] == FILLED and state[i] in (NONE, FILLED):
            state[i], pos[i] = FILLED, pos[i - 1]

    return Track(fps=fps, width=width, height=height, state=state, pos=pos, conf=conf, dup=dup)
