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
    penalty: float = 40.0       # cost of an extra flight at 60 fps (in σ² units): higher = fewer contacts.
                                #   Scaled by fps/60: same real-time evidence needed at any frame rate
    sigma_bw: float = 0.1       # base position noise, ball-widths
    sigma_size: float = 0.08    # + this fraction of the ball size
    jitter_s: float = 0.006     # + timing jitter: speed × this many seconds (real timestamps are used,
                                #   so this is just residual jitter; ~0.35 of a frame at 60 fps)
    meet_sizes: float = 2.0     # parabolas must meet within this many ball-widths to call it a contact
    tracknet_sigma_bw: float = 0.35  # extra noise on TrackNet points (measured: median error 0.37 ball-widths
                                     #   vs 0.14 for YOLO detections on the labelled clips)
    # bounce classification
    bounce_keep_dir: float = 0.3    # |vx| must be at least this fraction of the incoming |vx| (same sign)
    bounce_max_gain: float = 1.10   # outgoing speed at most this × incoming (a bounce loses energy)
    bounce_low_frac: float = 0.80   # contact y must be in the lowest (1 - frac) of this clip's ball y range


@dataclass
class Flight:
    f0: int                     # first observed frame
    f1: int                     # last observed frame
    n: int                      # observed points
    cx: list                    # x(t) = cx[0] + cx[1]·t + cx[2]·t², t = real time since f0, in nominal frames
    cy: list
    rms: float                  # fit residual, px
    tau: object = None          # time axis of the clip (frame -> real time in nominal frames)

    def _t(self, f):
        if self.tau is None:
            return f - self.f0
        ax = np.arange(len(self.tau))
        return float(np.interp(f, ax, self.tau) - self.tau[self.f0])

    def at(self, f):
        t = self._t(f)
        return (self.cx[0] + self.cx[1] * t + self.cx[2] * t * t,
                self.cy[0] + self.cy[1] * t + self.cy[2] * t * t)

    def vel(self, f):
        """px per nominal frame"""
        t = self._t(f)
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


def segment(frames, xs, ys, sizes, fps, params: FlightParams | None = None, tau=None, extra_bw=None):
    """frames/xs/ys/sizes: observed ball points (no repeated frames), sorted by frame.
    tau: the clip's time axis (real time per frame, in nominal frames); fits use real time.
    extra_bw: optional extra position noise per point, ball-widths (noisier sources count less).
    Returns (flights, contacts)."""
    P = params or FlightParams()
    F = np.asarray(frames, float)
    TA = np.asarray(tau, float) if tau is not None and len(tau) else None
    T = TA[np.asarray(frames, int)] if TA is not None else F      # real time of each point
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
            sp[i] = np.hypot(X[j1] - X[j0], Y[j1] - Y[j0]) / max(T[j1] - T[j0], 1)
    ball_med = float(np.median(S)) if len(S) else 30.0
    sig = P.sigma_bw * ball_med + P.sigma_size * S + P.jitter_s * sp * fps   # sp: px per nominal frame
    if extra_bw is not None:
        sig = np.sqrt(sig ** 2 + (np.asarray(extra_bw, float) * ball_med) ** 2)
    penalty = P.penalty * fps / 60.0
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
            if T[j - 1] - T[i] > max_len:
                break
            if best[i] == INF:
                continue
            t = T[i:j] - T[i]
            cx, rx = _fit(t, X[i:j])
            cy, ry = _fit(t, Y[i:j])
            cost = float(np.sum(w[i:j] * (rx * rx + ry * ry)))
            tot = best[i] + cost + penalty
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
        flights.append(Flight(int(F[i]), int(F[j - 1]), j - i, list(map(float, cx)), list(map(float, cy)), rms,
                              TA))

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
            frame=round(tc, 2),
            t=(float(np.interp(tc, np.arange(len(TA)), TA)) if TA is not None else tc) / fps,
            x=float(x), y=float(y),
            kind="bounce" if is_bounce else "touch",
            vx_in=float(vin[0]), vy_in=float(vin[1]), vx_out=float(vout[0]), vy_out=float(vout[1]),
            angle_deg=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), speed_ratio=float(ratio),
            in_gap=not bool(near_obs), gap=dmin,
        ))
    return flights, contacts


def from_track(track, params: FlightParams | None = None):
    """Segment a linker Track: uses detected, weak and tracknet points (filled frames are guesses),
    skips repeated frames."""
    P = params or FlightParams()
    fr, xs, ys, ss, ex = [], [], [], [], []
    for f, (st, p) in enumerate(zip(track.state, track.pos)):
        if p is None or st not in ("detected", "weak", "tracknet"):
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
        ex.append(P.tracknet_sigma_bw if st == "tracknet" else 0.0)
    return segment(fr, xs, ys, ss, track.fps, P, getattr(track, "tau", None), extra_bw=ex)


def apply_flights(track, flights, contacts, max_dev_sizes: float = 1.5, smooth_observed: bool = False):
    """Physics gap filling: frames without a detection take their flight's parabola.

    - observed frames (detected/weak) keep their detected position (with
      smooth_observed=True they're replaced by the fit when within max_dev_sizes)
    - unobserved frames inside a flight (filled/none) get the fitted position, state
      "filled" — gaps follow the real arc instead of a smooth guess
    - between two flights that meet at a contact, frames before the contact follow the
      incoming flight and frames after it the outgoing one
    - fitted positions outside the picture area are not drawn (ball left the frame)
    Returns new (pos, state) lists; the track object is not modified.
    """
    n = len(track.pos)
    pos, state = list(track.pos), list(track.state)
    x0, y0, x1, y1 = track.content or (0, 0, track.width, track.height)
    inside = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def put(f, fl):
        p = fl.at(f)
        sz = track.size[f] if getattr(track, "size", None) and track.size[f] else 30.0
        if state[f] in ("detected", "weak", "tracknet"):
            # real detections are more accurate than the fit (measured on the labelled
            # clips: replacing them raised the median error 4.8 -> 5.5 px), so keep them
            if smooth_observed and track.pos[f] is not None and \
                    np.hypot(p[0] - track.pos[f][0], p[1] - track.pos[f][1]) <= max_dev_sizes * sz:
                pos[f] = (float(p[0]), float(p[1]))
        elif inside(p):
            pos[f], state[f] = (float(p[0]), float(p[1])), "filled"
        else:
            pos[f], state[f] = None, "none"

    for fl in flights:
        for f in range(max(0, fl.f0), min(n, fl.f1 + 1)):
            put(f, fl)
    cmap = {}
    for c in contacts:
        cmap.setdefault(int(np.floor(c.frame)), c)
    for a, b in zip(flights, flights[1:]):
        # a contact between them?
        c = next((c for c in contacts if a.f1 - 1 <= c.frame <= b.f0 + 1), None)
        if c is None:
            continue
        for f in range(a.f1 + 1, b.f0):
            put(f, a if f <= c.frame else b)
    # repeated frames show the previous image
    if track.dup:
        for f in range(1, n):
            if track.dup[f] and pos[f - 1] is not None:
                pos[f], state[f] = pos[f - 1], state[f - 1]
    return pos, state
