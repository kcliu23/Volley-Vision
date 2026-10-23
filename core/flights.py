"""Flight segmentation: split the ball track into free flights and find the contacts between them.

Between two contacts (a touch or a floor bounce) the ball is in free flight, so in the
image each coordinate is well described by a parabola in time. We split the track into
the fewest flights that fit well:

    total = Σ over flights [ fit error / σ² ]  +  penalty × (number of flights)

and find the exact best split with dynamic programming over the track's points
(optimal segmentation / change-point detection). Gaps are fine: a flight can span
frames where the ball wasn't seen.

Each boundary between two flights is a contact. Its time and position are where the
incoming and outgoing parabolas meet, so contacts are found even when the ball was
hidden at that moment (e.g. a bounce behind a player's legs).

Contacts are classified:
    bounce   falling in, rising out, keeps its sideways direction, loses speed, and is
             low in the picture (near the floor) — a floor bounce
    touch    anything else (pass, set, attack, block...)
Without camera calibration "low in the picture" is relative to this clip's own lowest
ball positions; with calibration it becomes "height ≈ 0".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import numpy as np


@dataclass
class FlightParams:
    min_pts: int = 4            # a flight needs at least this many observed points
    max_len_s: float = 3.0      # longest flight considered
    penalty: float = 40.0       # cost of an extra flight (in σ² units): higher = fewer contacts
    sigma_px: float = 3.0       # base position noise
    sigma_size: float = 0.08    # + this fraction of the ball size
    sigma_speed: float = 0.35   # + this fraction of the per-frame speed (timing jitter)
    meet_sizes: float = 2.0     # parabolas must meet within this many ball-widths to call it a contact
    # bounce classification
    bounce_keep_dir: float = 0.3    # |vx| must be at least this fraction of the incoming |vx| (same sign)
    bounce_max_gain: float = 1.10   # outgoing speed at most this × incoming (a bounce loses energy)
    bounce_low_frac: float = 0.80   # contact y must be in the lowest (1 - frac) of this clip's ball y range


@dataclass
class Flight:
    f0: int                     # first observed frame
    f1: int                     # last observed frame
    n: int                      # observed points
    cx: list                    # x(t) = cx[0] + cx[1]·t + cx[2]·t², t = frame - f0
    cy: list
    rms: float                  # fit residual, px

    def at(self, f):
        t = f - self.f0
        return (self.cx[0] + self.cx[1] * t + self.cx[2] * t * t,
                self.cy[0] + self.cy[1] * t + self.cy[2] * t * t)

    def vel(self, f):
        t = f - self.f0
        return (self.cx[1] + 2 * self.cx[2] * t, self.cy[1] + 2 * self.cy[2] * t)


@dataclass
class Contact:
    frame: float                # may be fractional / inside a gap
    t: float                    # seconds
    x: float
    y: float
    kind: str                   # "touch" | "bounce"
    vx_in: float                # px/frame
    vy_in: float
    vx_out: float
    vy_out: float
    angle_deg: float
    speed_ratio: float          # |v_out| / |v_in|
    in_gap: bool                # no observation within ±2 frames: located from the two flights
    gap: float                  # distance between the flights at the meeting point, px
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


def _fit(t, v):
    """Quadratic least squares; returns coefficients [a0, a1, a2] and residuals."""
    A = np.stack([np.ones_like(t), t, t * t], 1)
    c, *_ = np.linalg.lstsq(A, v, rcond=None)
    return c, v - A @ c


def segment(frames, xs, ys, sizes, fps, params: FlightParams | None = None):
    """frames/xs/ys/sizes: observed ball points (no repeated frames), sorted by frame.
    Returns (flights, contacts)."""
    P = params or FlightParams()
    F = np.asarray(frames, float)
    X = np.asarray(xs, float)
    Y = np.asarray(ys, float)
    S = np.asarray(sizes, float)
    N = len(F)
    if N < P.min_pts:
        return [], []

    # per-point noise: detection jitter + timing jitter proportional to local speed
    sp = np.zeros(N)
    for i in range(N):
        j0, j1 = max(0, i - 2), min(N - 1, i + 2)
        if j1 > j0:
            sp[i] = np.hypot(X[j1] - X[j0], Y[j1] - Y[j0]) / max(F[j1] - F[j0], 1)
    sig = P.sigma_px + P.sigma_size * S + P.sigma_speed * sp
    w = 1.0 / sig ** 2
    max_len = P.max_len_s * fps

    # cost[i][j]: weighted SSE of one flight over points i..j (inclusive)
    INF = float("inf")
    best = np.full(N + 1, INF)          # best[j] = min cost of segmenting points [0, j)
    best[0] = 0.0
    arg = np.full(N + 1, -1, int)
    fits = {}
    for j in range(P.min_pts, N + 1):
        for i in range(j - P.min_pts, -1, -1):
            if F[j - 1] - F[i] > max_len:
                break
            if best[i] == INF:
                continue
            t = F[i:j] - F[i]
            cx, rx = _fit(t, X[i:j])
            cy, ry = _fit(t, Y[i:j])
            cost = float(np.sum(w[i:j] * (rx * rx + ry * ry)))
            tot = best[i] + cost + P.penalty
            if tot < best[j]:
                best[j], arg[j] = tot, i
                fits[(i, j)] = (cx, cy, float(np.sqrt(np.mean(rx * rx + ry * ry))))
    # points that can't be covered (too few) are left out: walk back from the last reachable end
    end = max(j for j in range(N + 1) if best[j] < INF) if np.isfinite(best).any() else 0
    segs = []
    j = end
    while j > 0 and arg[j] >= 0:
        i = arg[j]
        segs.append((i, j))
        j = i
    segs = segs[::-1]

    flights = []
    for i, j in segs:
        cx, cy, rms = fits[(i, j)]
        flights.append(Flight(int(F[i]), int(F[j - 1]), j - i, list(map(float, cx)), list(map(float, cy)), rms))

    # contacts between consecutive flights
    ylo, yhi = np.percentile(Y, 2), np.percentile(Y, 98)       # image y grows downward
    contacts = []
    for a, b in zip(flights, flights[1:]):
        ball = float(np.median(S[(F >= a.f0) & (F <= b.f1)])) if len(S) else 30.0
        lo, hi = a.f1, b.f0
        ts = np.linspace(lo - 0.5, hi + 0.5, max(4, int((hi - lo + 1) * 4)))
        d = [np.hypot(*(np.subtract(a.at(t), b.at(t)))) for t in ts]
        k = int(np.argmin(d))
        tc, dmin = float(ts[k]), float(d[k])
        if dmin > P.meet_sizes * ball:
            continue                     # flights don't meet: a break in the track, not a contact
        pa, pb = a.at(tc), b.at(tc)
        x, y = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        vin, vout = a.vel(tc), b.vel(tc)
        s_in, s_out = np.hypot(*vin), np.hypot(*vout)
        cosang = (vin[0] * vout[0] + vin[1] * vout[1]) / max(s_in * s_out, 1e-6)
        ratio = s_out / max(s_in, 1e-6)
        # bounce: falling in, rising out, keeps sideways direction, loses energy, low in the picture
        same_dir = vin[0] * vout[0] > 0 and abs(vout[0]) >= P.bounce_keep_dir * abs(vin[0])
        low = y >= ylo + P.bounce_low_frac * (yhi - ylo)
        # sideways direction isn't required: a bounce near a wall or with spin can reverse it
        is_bounce = vin[1] > 0 and vout[1] < 0 and ratio <= P.bounce_max_gain and low
        near_obs = np.any(np.abs(F - tc) <= 2)
        contacts.append(Contact(
            frame=round(tc, 2), t=tc / fps, x=float(x), y=float(y),
            kind="bounce" if is_bounce else "touch",
            vx_in=float(vin[0]), vy_in=float(vin[1]), vx_out=float(vout[0]), vy_out=float(vout[1]),
            angle_deg=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), speed_ratio=float(ratio),
            in_gap=not bool(near_obs), gap=dmin,
        ))
    return flights, contacts


def from_track(track, params: FlightParams | None = None):
    """Segment a linker Track: uses detected + weak points (filled frames are guesses),
    skips repeated frames."""
    fr, xs, ys, ss = [], [], [], []
    for f, (st, p) in enumerate(zip(track.state, track.pos)):
        if p is None or st not in ("detected", "weak"):
            continue
        if track.dup and track.dup[f]:
            continue
        sz = track.size[f] if getattr(track, "size", None) and track.size[f] else 30.0
        if track.content:                  # box cut off at the picture edge: position is biased
            x0, y0, x1, y1 = track.content
            if p[0] - sz / 2 <= x0 + 2 or p[1] - sz / 2 <= y0 + 2 or p[0] + sz / 2 >= x1 - 2 or p[1] + sz / 2 >= y1 - 2:
                continue
        fr.append(f)
        xs.append(p[0])
        ys.append(p[1])
        ss.append(track.size[f] if getattr(track, 'size', None) and track.size[f] else 30.0)
    return segment(fr, xs, ys, ss, track.fps, params)
