"""The ball's 3D path through a whole rally, touches chosen by the fit itself (global, one score).

The ball between touches flies under gravity + air drag (core.ball3d). Instead of taking the
2D tracker's flight splits and patching them (merge, re-split, cut-offs: core of tools.ball3d_run's
first version), this module picks the touch moments and the flights TOGETHER:

    minimise   sum over flights of  fit error (2D points + ball size [+ out of the picture])
             + TOUCH_COST per touch
             + continuity at each touch (the two flights meet)  + physical limits
               (touches within reach, no flight faster than a human can hit, and a touch must
               change the ball's velocity: one that barely does is model error, not a touch)

1. candidates   possible touch moments: the tracker's contact hints (+-2 frames), sharp turns of
                the 2D path, the edges of gaps, and a grid every GRID_S (safety net: a touch
                every hint missed is still within a few frames of a candidate). Wrong
                candidates cost nothing: the fit only uses one if it pays for its TOUCH_COST
2. pieces       one arc per candidate pair up to MAX_FLIGHT_S apart, fitted fast (no drag,
                closed-form motion, Gauss-Newton with robust weights: ms each)
3. DP           dynamic programming over (previous flight, flight) chooses the touches with the
                lowest total score: exact, no order effects
4. refine       with drag (slower, exact): try removing each chosen touch and adding the best one
                inside each flight; keep changes that lower the same score
5. joint fit    all flights together, each starting where the previous ended, touch times
                solved continuously (core.ball3d.fit_rally)
6. re-check     a flight that still fits badly gets a candidate at every frame and the rally is
                solved again; if it still fits badly it's flagged, not trusted

Set numbers (TOUCH_COST first) live in core.params3d, each marked with how it was set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.ball3d import BALL_D, G, Link, Segment, fit_rally
from core.params3d import (DV_SOFT, GRID_S, HUBER, LINK_SIGMA_M, MAX_FLIGHT_S, MAX_SPEED, MIN_PTS,  # noqa: F401
                           OUT_CAP, OUT_SIGMA, PLAYER_COST, PLAYER_SOFT_M, REACH_M, SIGMA_PX, SIZE_REL,
                           TOUCH_COST, TURN_DEG)


def huber(r):
    a = np.abs(r)
    return np.where(a <= HUBER, 0.5 * a * a, HUBER * a - 0.5 * HUBER * HUBER)


def capped(r):
    return np.minimum(huber(r), OUT_CAP)


@dataclass
class Obs:
    """2D observations of one rally, in time order."""
    t: np.ndarray
    uv: np.ndarray
    size: np.ndarray
    w: np.ndarray
    frames: np.ndarray
    edge: np.ndarray                     # ball at the picture's edge (off-centre click)
    out_frames: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # ball out of the picture
    box: tuple | None = None
    fps: float = 60.0
    sigma: float = 3.0                   # 2D noise used for scoring (measured per rally in solve())
    T: np.ndarray | None = None          # frame -> true time in frame units (core.frametime); None = f
    net_h: float | None = None           # net height (m): flights can't pass through it
    touchers: object = None              # touchers(frame, ball_uv) -> core.players.touch_evidence list
    near: object = None                  # near(t, xyz) -> metres outside every player's reach, or
                                         # None where there is no player data (core.players.reach)

    def segment(self, i: int, j: int) -> Segment:
        f0, f1 = self.frames[i], self.frames[j - 1]
        o = self.out_frames[(self.out_frames > f0) & (self.out_frames < f1)]
        return Segment(self.t[i:j], self.uv[i:j], self.size[i:j], self.w[i:j], frames=self.frames[i:j],
                       out_t=(self.T[o] if self.T is not None else o) / self.fps, box=self.box)


# --- fast piece fit: no drag, analytic Jacobian, robust Gauss-Newton -----------------------------

class _Cam:
    def __init__(self, cam):
        self.R, self.T, self.f, self.cx, self.cy = cam.R, np.asarray(cam.tvec, float), cam.f, cam.cx, cam.cy
        self.cam = cam


def _fast_fit(o: Obs, i: int, j: int, C: _Cam, iters: int = 12):
    """Best no-drag arc for obs[i:j]: (score, p0, v0 at t[i]) or None."""
    from core.ball3d import _init_segment, _Proj
    t = o.t[i:j] - o.t[i]
    uv, sz, w = o.uv[i:j], o.size[i:j], o.w[i:j]
    seg = o.segment(i, j)
    th = _init_segment(seg, _Proj(C.cam), 12.0)
    hs = np.isfinite(sz) & (sz > 2)
    sw = np.sqrt(w)
    lam = 1e-3
    last = None
    R, T, f = C.R, C.T, C.f
    for _ in range(iters):
        X = th[:3] + np.outer(t, th[3:]) + 0.5 * np.outer(t * t, G)
        Xc = X @ R.T + T
        z = np.maximum(Xc[:, 2], 0.3)
        x, y = Xc[:, 0] / z, Xc[:, 1] / z
        sg = o.sigma
        ru = (f * x + C.cx - uv[:, 0]) / sg * sw
        rv = (f * y + C.cy - uv[:, 1]) / sg * sw
        # d(u,v)/dX
        du = (f / z)[:, None] * (R[0][None, :] - x[:, None] * R[2][None, :]) / sg * sw[:, None]
        dv = (f / z)[:, None] * (R[1][None, :] - y[:, None] * R[2][None, :]) / sg * sw[:, None]
        r = [ru, rv]
        J = [np.hstack([du, du * t[:, None]]), np.hstack([dv, dv * t[:, None]])]
        if hs.any():
            pred = f * BALL_D / z[hs]
            sc = SIZE_REL * sz[hs]
            r.append((pred - sz[hs]) / sc * sw[hs])
            ds = (-f * BALL_D / z[hs] ** 2)[:, None] * R[2][None, :] / sc[:, None] * sw[hs][:, None]
            J.append(np.hstack([ds, ds * t[hs][:, None]]))
        n2 = 2 * len(t)
        r = np.concatenate(r); J = np.vstack(J)
        hr = huber(r)
        hr[:n2] = np.minimum(hr[:n2], OUT_CAP)
        cost = float(np.sum(hr))
        if last is not None and cost > last[0]:
            th = last[1]; lam *= 10
            if lam > 1e4:
                break
            continue
        last = (cost, th.copy())
        a = np.abs(r)
        wt = np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-9))
        wt[:n2][hr[:n2] >= OUT_CAP] = 0.0                # ignored points don't pull the arc
        A = J.T @ (J * wt[:, None]); b = J.T @ (r * wt)
        A += lam * np.diag(np.diag(A) + 1e-9)
        try:
            th = th - np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        lam = max(lam / 3, 1e-6)
    if last is None:
        return None
    cost, th = last
    v = np.linalg.norm(th[3:])
    if v > MAX_SPEED:
        cost += ((v - MAX_SPEED) / 3.0) ** 2
    return cost, th[:3].copy(), th[3:].copy()


def _pos_fast(piece, t):
    """Position of a fast (no-drag) piece at time t."""
    (cost, p0, v0, t0) = piece
    d = t - t0
    return p0 + v0 * d + 0.5 * G * d * d


def dv_penalty(va, vb) -> float:
    """Extra cost of a touch that barely changes the ball's velocity."""
    return 2 * TOUCH_COST * math.exp(-(float(np.linalg.norm(np.asarray(va) - np.asarray(vb))) / DV_SOFT) ** 2)


def _vel_fast(piece, t):
    (cost, p0, v0, t0) = piece
    return v0 + G * (t - t0)


def player_penalty(o: "Obs", tc: float, x) -> float:
    """Extra cost of a touch at time tc, ball at x, from the players: nothing next to a player
    (or on the floor: a bounce), up to PLAYER_COST far from all of them."""
    if o.near is None or x[2] < 0.3:
        return 0.0
    d = o.near(tc, x)
    if d is None:
        return 0.0
    return PLAYER_COST * (1 - math.exp(-(d / PLAYER_SOFT_M) ** 2))


def _transition(A, B, tc, o=None):
    """Continuity + reach + velocity-change (+ player) score of a touch at time tc between
    pieces A and B."""
    a, b = _pos_fast(A, tc), _pos_fast(B, tc)
    s = float(huber(np.linalg.norm(a - b) / LINK_SIGMA_M))
    if o is not None:
        s += player_penalty(o, tc, 0.5 * (a + b))
    s += dv_penalty(_vel_fast(A, tc), _vel_fast(B, tc))
    z = 0.5 * (a[2] + b[2])
    if z > REACH_M:
        s += ((z - REACH_M) / 0.2) ** 2
    if z < -0.1:
        s += ((-0.1 - z) / 0.1) ** 2
    return s


# --- candidates ---------------------------------------------------------------------------------

def candidates(o: Obs, hints: list[float] = ()) -> list[int]:
    """Boundary indices b (a touch between obs[b-1] and obs[b]); always includes 0 and n."""
    n = len(o.t)
    c = {0, n}
    fr = o.frames
    for h in hints:                                   # tracker contact hints, +-2 frames
        b = int(np.searchsorted(fr, h))
        for d in range(-2, 3):
            if 0 < b + d < n:
                c.add(b + d)
    for b in range(1, n):                             # edges of gaps
        if fr[b] - fr[b - 1] > 2:
            c.add(b)
    k = 3                                             # sharp turns of the 2D path
    for b in range(k, n - k):
        v1 = o.uv[b] - o.uv[b - k]; v2 = o.uv[b + k] - o.uv[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 1 and n2 > 1:
            ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))
            if ang > TURN_DEG or max(n1, n2) / max(min(n1, n2), 1e-6) > 1.8:
                c.add(b)
    g = max(1, int(round(GRID_S * o.fps)))           # safety-net grid
    nxt = fr[0] + g
    for b in range(1, n):
        if fr[b] >= nxt:
            c.add(b); nxt = fr[b] + g
    return sorted(c)


# --- the global choice (DP) -------------------------------------------------------------------

def choose(o: Obs, cam, cands: list[int]) -> list[int]:
    """Boundaries [0, b1, ..., n] with the lowest total score (fast pieces)."""
    C = _Cam(cam)
    n = len(o.t)
    pieces = {}
    for a_i, i in enumerate(cands):
        for j in cands[a_i + 1:]:
            if o.t[j - 1] - o.t[i] > MAX_FLIGHT_S:
                break
            if j - i < MIN_PTS:
                continue
            r = _fast_fit(o, i, j, C)
            if r is not None:
                pieces[(i, j)] = (r[0], r[1], r[2], o.t[i])
    # best[(i, j)] = lowest score covering obs[0:j] with last piece (i, j)
    best, back = {}, {}
    by_end = {}
    for (i, j) in pieces:
        by_end.setdefault(j, []).append(i)
    for j in sorted(by_end):
        for i in by_end[j]:
            P = pieces[(i, j)]
            if i == 0:
                best[(i, j)] = P[0]; back[(i, j)] = None
                continue
            tc = 0.5 * (o.t[i - 1] + o.t[i])
            cand = None
            for h in by_end.get(i, []):
                if (h, i) not in best:
                    continue
                s = best[(h, i)] + P[0] + TOUCH_COST + _transition(pieces[(h, i)], P, tc, o)
                if cand is None or s < cand[0]:
                    cand = (s, h)
            if cand is not None:
                best[(i, j)] = cand[0]; back[(i, j)] = cand[1]
    ends = [(best[(i, n)], i) for i in by_end.get(n, []) if (i, n) in best]
    if not ends:
        return [0, n]
    _, i = min(ends)
    out, j = [n], n
    while i is not None:
        out.append(i)
        i, j = back[(i, j)], i
    return sorted(set(out))


# --- refine with drag, joint fit ------------------------------------------------------------------

def _drag_piece_score(o: Obs, i: int, j: int, cam) -> tuple[float, object]:
    seg = o.segment(i, j)
    fit = fit_rally([seg], [], cam, net_h=o.net_h)
    uvp = cam.project(fit.at(0, seg.t))
    r = [((uvp - seg.uv) / o.sigma * np.sqrt(seg.w)[:, None]).ravel()]
    z = cam.depth(fit.at(0, seg.t))
    hs = np.isfinite(seg.size) & (seg.size > 2)
    if hs.any():
        r.append((cam.f * BALL_D / np.maximum(z[hs], 0.3) - seg.size[hs]) / (SIZE_REL * seg.size[hs]) * np.sqrt(seg.w[hs]))
    if len(seg.out_t) and seg.box is not None:
        uo = cam.project(fit.at(0, seg.out_t))
        x0, y0, x1, y1 = seg.box
        ins = np.minimum.reduce([uo[:, 0] - x0, x1 - uo[:, 0], uo[:, 1] - y0, y1 - uo[:, 1]])
        r.append(np.maximum(ins, 0) / o.sigma)
    s = float(np.sum(capped(r[0]))) + float(np.sum(huber(np.concatenate(r[1:])))) if len(r) > 1 else float(np.sum(capped(r[0])))
    v = np.linalg.norm((fit.at(0, seg.t[0] + 1e-3)[0] - fit.at(0, seg.t[0])[0]) / 1e-3)
    if v > MAX_SPEED:
        s += ((v - MAX_SPEED) / 3.0) ** 2
    return s, fit


def refine(o: Obs, cam, bounds: list[int], cands: list[int], rounds: int = 3) -> list[int]:
    """Local search with the drag model: drop a touch / add the best touch inside a flight when
    that lowers the score (pieces only; continuity is settled in the joint fit)."""
    cache = {}

    def S(i, j):
        if (i, j) not in cache:
            cache[(i, j)] = _drag_piece_score(o, i, j, cam) if j - i >= MIN_PTS else (math.inf, None)
        return cache[(i, j)][0]

    def T(a, b, c):
        """Cost of a touch at boundary b between pieces (a, b) and (b, c): TOUCH_COST + small-
        velocity-change penalty (from the drag fits)."""
        S(a, b); S(b, c)
        fa, fb = cache[(a, b)][1], cache[(b, c)][1]
        if fa is None or fb is None:
            return math.inf
        tc = 0.5 * (o.t[b - 1] + o.t[b])
        va = (fa.at(0, tc + 1e-3)[0] - fa.at(0, tc)[0]) / 1e-3
        vb = (fb.at(0, tc + 1e-3)[0] - fb.at(0, tc)[0]) / 1e-3
        return TOUCH_COST + dv_penalty(va, vb) + player_penalty(o, tc, 0.5 * (fa.at(0, tc)[0] + fb.at(0, tc)[0]))

    B = list(bounds)
    for _ in range(rounds):
        changed = False
        k = 1
        while k < len(B) - 1:                          # drop a touch?
            a, b, c = B[k - 1], B[k], B[k + 1]
            if o.t[c - 1] - o.t[a] <= MAX_FLIGHT_S and S(a, c) < S(a, b) + S(b, c) + T(a, b, c):
                B.pop(k); changed = True
            else:
                k += 1
        nb = []
        for a, b in zip(B[:-1], B[1:]):                # add a touch inside a flight?
            inner = [c for c in cands if a + MIN_PTS <= c <= b - MIN_PTS]
            best = None
            for c in inner:
                gain = S(a, b) - (S(a, c) + S(c, b) + T(a, c, b))
                if gain > 0 and (best is None or gain > best[0]):
                    best = (gain, c)
            nb.append(a)
            if best:
                nb.append(best[1]); changed = True
        nb.append(B[-1])
        B = nb
        if not changed:
            break
    return B


def jitter_sigma(o: Obs) -> float:
    """Per-axis 2D noise from frame-to-frame second differences of consecutive points (a smooth
    flight barely bends in one frame; a touch is one frame in many)."""
    fr, uv = o.frames, o.uv
    m = (fr[2:] - fr[1:-1] == 1) & (fr[1:-1] - fr[:-2] == 1) & ~o.edge[1:-1]
    d2 = (uv[2:] - 2 * uv[1:-1] + uv[:-2])[m]
    if len(d2) < 8:
        return SIGMA_PX
    return float(np.median(np.abs(d2)) / 0.6745 / math.sqrt(6))


def solve(o: Obs, cam, hints=(), log=print):
    """Pieces [(segment, (f0, f1), fit, i)] for one rally, plus {"bad": [...], "sigma": px}.
    The 2D noise level is measured, not assumed: first from the points' own jitter, then from
    what the fit leaves over; if the data is noisier than assumed, the rally is solved again at
    the measured level (noise must not be explained away with extra touches)."""
    o.sigma = max(SIGMA_PX, jitter_sigma(o))
    for _ in range(2):
        pieces, info = _solve_once(o, cam, hints, log)
        res = info["resid"]
        meas = float(1.4826 * np.median(np.abs(res))) if len(res) >= 10 else o.sigma
        if meas <= 1.25 * o.sigma:
            break
        log(f"2D noise here is {meas:.1f} px, not {o.sigma:.1f}: solving again at that level")
        o.sigma = meas
    info["sigma"] = o.sigma
    return pieces, info


def _anchors(o: Obs, bounds) -> list:
    """Touches made by a seen player (poses): at each chosen touch, who has a wrist at the ball,
    on which side of the net, standing where. Both flights at the touch get the anchor. Players
    on both sides at the ball (a joust, a block) put it at the net; one player's spot is used
    only when they are the only one there."""
    out = []
    for k in range(1, len(bounds) - 1):
        b = bounds[k]
        tc = 0.5 * (o.t[b - 1] + o.t[b])
        i = b if abs(o.t[b] - tc) < abs(o.t[b - 1] - tc) else b - 1
        ev = o.touchers(int(o.frames[i]), o.uv[i])
        sides = {e["side"] for e in ev if e["side"]}
        if not sides:
            continue
        side = 0 if sides == {-1, 1} else sides.pop()
        spots = [e["spot"] for e in ev if e["spot"] is not None]
        spot = spots[0] if len(ev) == 1 and spots else None
        for kk in (k - 1, k):
            out.append({"k": kk, "t": tc, "side": side, "spot": spot})
    return out


def _solve_once(o: Obs, cam, hints, log):
    cands = candidates(o, hints)
    bounds = choose(o, cam, cands)
    bounds = refine(o, cam, bounds, cands)
    for attempt in range(2):
        segs = [o.segment(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        links = []
        for k in range(1, len(bounds) - 1):
            b = bounds[k]
            tc = 0.5 * (o.t[b - 1] + o.t[b])
            links.append(Link(k - 1, tc, "touch", max(0.5 / o.fps, 0.5 * (o.t[b] - o.t[b - 1]))))
        anchors = _anchors(o, bounds) if o.touchers else []
        fit = fit_rally(segs, links, cam, net_h=o.net_h, anchors=anchors)
        drop = False                                     # wrong points: out of the final fit
        for k, sg_ in enumerate(segs):
            e = np.abs(cam.project(fit.at(k, sg_.t)) - sg_.uv).max(axis=1) / o.sigma
            off = (e > OUT_SIGMA) & (sg_.w > 0)
            if off.any() and (~off).sum() >= MIN_PTS:
                sg_.w = np.where(off, 0.0, sg_.w); drop = True
        if drop:
            fit = fit_rally(segs, links, cam, net_h=o.net_h, anchors=anchors)
        bad = []
        for k, s in enumerate(segs):
            e = np.hypot(*(cam.project(fit.at(k, s.t)) - s.uv).T)
            ok = ~o.edge[bounds[k]:bounds[k + 1]]
            med = float(np.median(e[ok] if ok.sum() >= 3 else e))
            if med > max(8.0, 3 * o.sigma):
                bad.append(k)
        if not bad or attempt == 1:
            break
        dense = set(cands)                               # layer 3: every frame inside bad flights
        for k in bad:
            dense.update(range(bounds[k] + 1, bounds[k + 1]))
        cands = sorted(dense)
        bounds = refine(o, cam, choose(o, cam, cands), cands)
    pieces, resid = [], []
    for k, s in enumerate(segs):
        f0 = int(o.frames[bounds[k]])
        f1 = int(o.frames[bounds[k + 1] - 1])
        pieces.append((s, (f0, f1), fit, k))
        d = (cam.project(fit.at(k, s.t)) - s.uv)[~o.edge[bounds[k]:bounds[k + 1]]]
        resid.extend(d.ravel().tolist())
    return pieces, {"bad": bad, "touch_t": fit.info.get("touch_t", []), "resid": np.array(resid)}


def solve_clip(uv, w, size, edge, box, fps, cam, contacts, log=print, rally_gap_s: float = 1.5,
               players=None, T=None, net_h=None, poses=None, ball_px: float = 25.0):
    """A whole clip: split into rallies (stretches with no ball for > rally_gap_s), solve() each
    with the tracker's contacts as hints. T: true frame times (core.frametime); players: from
    core.players.build; poses: output/<clip>/pose.json frames (tools.pose), to anchor touches to
    the player who made them. Returns [(rally, segment, (f0, f1), fit, i)]."""
    from core.ball3d import out_of_picture
    fr_all = np.where(np.isfinite(uv[:, 0]))[0]
    if not len(fr_all):
        return []
    near = None
    if players:
        from core.players import reach
        near = reach(players, fps)
    rallies = np.split(fr_all, np.where(np.diff(fr_all) > rally_gap_s * fps)[0] + 1)
    pieces = []
    for ri, fr in enumerate(rallies):
        if len(fr) < 8:
            continue
        full = np.arange(fr[0], fr[-1] + 1)
        out_f = out_of_picture(full, np.isfinite(uv[full, 0]), edge).astype(int)
        tf = (T[fr] if T is not None else fr) / fps
        o = Obs(tf, uv[fr], size[fr], w[fr], fr, edge[fr], out_f, box, fps, T=T, net_h=net_h, near=near)
        if poses:
            from core.players import touch_evidence
            o.touchers = lambda f, uv_, P=poses: touch_evidence(P, f, uv_, cam, ball_px, fps=fps)
        hints = [c["frame"] for c in contacts if fr[0] <= c["frame"] <= fr[-1]]
        res, info = solve(o, cam, hints, log=log)
        nt = len(res) - 1
        log(f"rally {fr[0]}-{fr[-1]}: {nt} touch{'es' if nt != 1 else ''} chosen by the fit"
            + (f"; flights still fitting badly: {[res[k][1] for k in info['bad']]}" if info["bad"] else ""))
        for seg, span, fit, i in res:
            pieces.append((ri, seg, span, fit, i))
    return pieces
