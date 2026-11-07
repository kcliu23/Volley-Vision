"""Physics backbone: the ball as a chain of flights (gravity arcs) joined at contacts.

Steps 2-3 of the redesign (docs/DEVELOPMENT_LOG.md). Instead of picking one point per frame
and smoothing afterwards, arcs are found directly in all the candidates, and the game ball is
the chain of arcs that meet at contacts.

  1. points      every frame's candidates (WASB, YOLO crop finds, sizing-pass anchors), merged
                 and calibrated as in the joint linker (core.joint_linker.build_points)
  2. arcs        seed-and-grow robust fitting: a seed of confident points a few frames apart
                 gives an arc; it grows frame by frame, taking the candidate nearest its
                 prediction (within a tolerance that depends on the source: a YOLO box is precise,
                 a WASB-only point less so) and refitting on ALL its points, so one wrong point
                 can't steer it. Physical limits: vertical curvature (gravity in the image) within
                 a range of the clip's gravity, small sideways curvature. Arcs sharing most of their
                 points are merged.
  3. chain       dynamic programming over arcs: evidence (calibrated points), frames not covered,
                 and how plausibly one arc hands over to the next: the arcs must meet (a contact)
                 near in space and time, or the ball re-enters after a long gap.
  4. output      per frame: an observed point on the chosen arc (YOLO boxes keep their own position,
                 finder-only points sit on the arc), or the arc's prediction inside it ("filled").

Time is real time in nominal frames (tau), so variable frame rate is handled.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from core.cost_linker import _gravity
from core.joint_linker import JointParams, build_points
from core.linker import DETECTED, WEAK, LinkParams, finalize


@dataclass
class ArcParams:
    seed_s: float = 0.12        # seed: confident points within this window
    min_pts: int = 5            # an arc needs at least this many points ...
    min_yolo_short: int = 3     # ... or (short arcs, e.g. off a block) this many YOLO-confirmed ones
    max_gap_s: float = 0.25     # an arc stops after this long without a point that fits
    tol_yolo_bw: float = 0.35   # a point fits if within this many ball-widths of the arc: YOLO box ...
    tol_finder_bw: float = 0.7  # ... finder-only point (WASB ~0.17 bw error, plus timing jitter)
    tol_grow_bw_s: float = 2.0  # + this per second since the arc's last point (prediction uncertainty)
    g_lo: float = 0.25          # vertical curvature within [g_lo, g_hi] x the clip's gravity (perspective:
    g_hi: float = 3.0           #   far balls fall fewer px/s^2); free if the clip's gravity is unknown
    x_curv_max: float = 0.6     # |sideways curvature| <= this x the clip's gravity (perspective only)
    seed_p: float = 0.5         # seed points need at least this probability
    merge_share: float = 0.6    # arcs sharing this share of the smaller one's points: keep the better
    max_seeds: int = 4000


@dataclass
class ChainParams:
    w_det: float = 0.7          # evidence weight on each point's logit (as the joint linker)
    logit_cap: float = 3.0      # one point's evidence counts at most this much
    miss: float = 1.0           # per usable frame no chosen arc covers
    sig0_bw: float = 0.8        # arcs must meet within this (ball-widths) ...
    sig_gap_bw_s: float = 6.0   # ... + this per second of gap between them
    touch: float = 2.0          # a contact (direction change between arcs)
    reentry: float = 8.0        # pick the ball up again after a long gap, anywhere
    reentry_gap_s: float = 0.6
    max_overlap_s: float = 0.08


@dataclass
class Arc:
    frames: list = field(default_factory=list)      # inlier frames (sorted)
    pts: dict = field(default_factory=dict)         # frame -> (x, y, size, logit, state, p)
    cx: np.ndarray = None                           # x(t) = cx0 + cx1 t + cx2 t^2, t = tau - t0
    cy: np.ndarray = None
    t0: float = 0.0
    rms: float = 0.0

    @property
    def f0(self):
        return self.frames[0]

    @property
    def f1(self):
        return self.frames[-1]

    def at_t(self, t):
        u = t - self.t0
        return (self.cx[0] + self.cx[1] * u + self.cx[2] * u * u, self.cy[0] + self.cy[1] * u + self.cy[2] * u * u)


def _fit(ts, xs, ys, ws, t0, g, P: ArcParams):
    """Weighted LS fit of x and y as quadratics in (t - t0), with the physical limits."""
    u = np.asarray(ts, float) - t0
    W = np.sqrt(np.asarray(ws, float))
    A = np.stack([np.ones_like(u), u, u * u], 1) * W[:, None]
    X = np.asarray(xs, float) * W
    Y = np.asarray(ys, float) * W
    if len(u) < 3 or np.ptp(u) < 1e-6:
        return None
    cy = np.linalg.lstsq(A, Y, rcond=None)[0]
    if g > 0:                                   # vertical curvature = g/2 within the range
        lo, hi = P.g_lo * g / 2, P.g_hi * g / 2
        if not lo <= cy[2] <= hi:
            c2 = min(max(cy[2], lo), hi)
            cy = np.r_[np.linalg.lstsq(A[:, :2], Y - A[:, 2] * c2, rcond=None)[0], c2]
    cx = np.linalg.lstsq(A, X, rcond=None)[0]
    lim = P.x_curv_max * g / 2 if g > 0 else None
    if lim is not None and abs(cx[2]) > lim:
        c2 = math.copysign(lim, cx[2])
        cx = np.r_[np.linalg.lstsq(A[:, :2], X - A[:, 2] * c2, rcond=None)[0], c2]
    return cx, cy


def _tol(st, ball, P: ArcParams, dt_s: float):
    base = P.tol_yolo_bw if st in (DETECTED, WEAK) else P.tol_finder_bw
    return ball * (base + P.tol_grow_bw_s * dt_s)


def _grow(seed: dict, pts_by_frame, tau, usable, fps, ball, g, P: ArcParams) -> Arc | None:
    """Grow an arc from seed points {frame: point} in both directions; refit on every point."""
    arc = Arc(pts=dict(seed))
    n = len(pts_by_frame)
    max_gap = max(2, int(round(P.max_gap_s * fps)))

    def refit():
        fs = sorted(arc.pts)
        arc.frames = fs
        arc.t0 = float(tau[fs[0]])
        r = _fit([tau[f] for f in fs], [arc.pts[f][0] for f in fs], [arc.pts[f][1] for f in fs],
                 [1.0 / (P.tol_yolo_bw if arc.pts[f][4] in (DETECTED, WEAK) else P.tol_finder_bw) ** 2 for f in fs],
                 arc.t0, g, P)
        if r is None:
            return False
        arc.cx, arc.cy = r
        res = [math.hypot(*(np.subtract(arc.at_t(tau[f]), arc.pts[f][:2]))) for f in fs]
        arc.rms = float(np.sqrt(np.mean(np.square(res))))
        return True

    if not refit():
        return None
    for direction in (+1, -1):
        last = arc.f1 if direction > 0 else arc.f0
        f, missed = last + direction, 0
        while 0 <= f < n and missed <= max_gap:
            if not usable[f]:
                f += direction
                continue
            px, py = arc.at_t(tau[f])
            dt = abs(tau[f] - tau[last]) / fps
            best = None
            for q in pts_by_frame[f]:
                d = math.hypot(q[0] - px, q[1] - py)
                if d <= _tol(q[4], ball, P, dt) and (best is None or d < best[0]):
                    best = (d, q)
            if best is None:
                missed += 1
            else:
                arc.pts[f] = best[1]
                last, missed = f, 0
                if not refit():
                    return None
                # the refit must still explain every earlier point (else this point bent the arc)
                if any(math.hypot(*(np.subtract(arc.at_t(tau[h]), arc.pts[h][:2]))) >
                       _tol(arc.pts[h][4], ball, P, 0.0) * 1.5 for h in arc.frames):
                    del arc.pts[f]
                    refit()
                    missed += 1
            f += direction
    n_yolo = sum(1 for q in arc.pts.values() if q[4] in (DETECTED, WEAK))
    if len(arc.pts) < P.min_pts and n_yolo < P.min_yolo_short:
        return None
    return arc


def find_arcs(pts_by_frame, tau, usable, fps, ball, g, P: ArcParams | None = None) -> list[Arc]:
    """All plausible arcs in the candidates (they may overlap in time: other balls, other objects)."""
    P = P or ArcParams()
    n = len(pts_by_frame)
    w = max(2, int(round(P.seed_s * fps)))
    claimed: dict = {}                       # (frame, x, y) -> arc index, to skip used seeds
    arcs: list[Arc] = []
    seeds = 0
    order = sorted(((q[3], f, i) for f in range(n) if usable[f] for i, q in enumerate(pts_by_frame[f])
                    if q[5] >= P.seed_p), reverse=True)
    for _, f, i in order:
        q = pts_by_frame[f][i]
        if (f, round(q[0]), round(q[1])) in claimed:
            continue
        seeds += 1
        if seeds > P.max_seeds:
            break
        # seed: this point + the nearest consistent point in each nearby frame (simple local motion)
        seed = {f: q}
        for direction in (+1, -1):
            prev = (f, q)
            for g2 in range(f + direction, f + direction * (w + 1), direction):
                if not 0 <= g2 < n or not usable[g2]:
                    continue
                dt = abs(g2 - prev[0])
                cand = [r for r in pts_by_frame[g2] if math.hypot(r[0] - prev[1][0], r[1] - prev[1][1]) <= ball * (0.6 + 3.0 * dt)]
                if cand:
                    r = min(cand, key=lambda r: -r[3])
                    seed[g2] = r
                    prev = (g2, r)
        if len(seed) < 3:
            continue
        arc = _grow(seed, pts_by_frame, tau, usable, fps, ball, g, P)
        if arc is None:
            continue
        k = len(arcs)
        arcs.append(arc)
        for h, r in arc.pts.items():
            claimed[(h, round(r[0]), round(r[1]))] = k
    # merge arcs that share most of their points: keep the better supported one
    keep = []
    for a in sorted(arcs, key=lambda a: -_evidence(a)):
        sa = {(h, round(r[0]), round(r[1])) for h, r in a.pts.items()}
        if any(len(sa & sb) >= 0.6 * min(len(sa), len(sb)) for sb, _ in keep):
            continue
        keep.append((sa, a))
    return sorted([a for _, a in keep], key=lambda a: a.f0)


def arc_gravity(arcs: list[Arc], min_pts: int = 15) -> float:
    """The clip's gravity in the image (px / nominal frame^2): median curvature of long, clean arcs."""
    c = [2 * a.cy[2] for a in arcs if len(a.pts) >= min_pts and a.cy[2] > 0]
    return float(np.median(c)) if len(c) >= 3 else 0.0


def _evidence(a: Arc, cap: float = 3.0) -> float:
    return float(sum(min(q[3], cap) for q in a.pts.values()))


def chain(arcs: list[Arc], n: int, tau, usable, fps, ball, C: ChainParams | None = None) -> list:
    """The sequence of arcs that is the game ball: [(arc index, first frame, last frame)].
    Consecutive arcs may overlap (around a contact both arcs collect nearby points): they are cut
    where they meet, and the overlap's evidence is counted once."""
    C = C or ChainParams()
    if not arcs:
        return []
    upre = np.concatenate([[0], np.cumsum(usable)])
    re_gap = int(round(C.reentry_gap_s * fps))
    ev = []
    for a in arcs:
        lg = [min(a.pts[f][3], C.logit_cap) for f in a.frames]
        ev.append(np.concatenate([np.cumsum(lg[::-1])[::-1], [0.0]]))     # evidence from index k on

    def idx_after(a: Arc, f):                  # first index of a.frames with frame > f
        import bisect
        return bisect.bisect_right(a.frames, f)

    def meet(a: Arc, b: Arc):
        """Where a (forward) and b (backward) come closest, over a's end / b's start (and their
        overlap): (distance px, cut frame). The cut frame is the last frame that belongs to a."""
        lo, hi = min(a.f1, b.f0), max(a.f1, b.f0)
        lo = max(lo, b.f0 - 1, a.f0)
        hi = min(max(hi, lo + 1), b.f1)
        best = None
        for f in range(lo, hi + 1):
            d = math.hypot(*(np.subtract(a.at_t(tau[f]), b.at_t(tau[f]))))
            if best is None or d < best[0]:
                best = (d, f)
        if a.f1 < b.f0:                           # a gap: sample the real time in between too
            for t in np.linspace(tau[a.f1], tau[b.f0], 8):
                d = math.hypot(*(np.subtract(a.at_t(t), b.at_t(t))))
                if d < best[0]:
                    best = (d, best[1])
        return best

    N = len(arcs)
    INF = float("inf")
    best, back, cut = [INF] * N, [None] * N, [None] * N
    for j, b in enumerate(arcs):
        cand = (C.miss * int(upre[b.f0]) - C.w_det * ev[j][0], None, None)
        for i in range(j):
            a = arcs[i]
            if best[i] == INF or a.f1 >= b.f1 or a.f0 >= b.f0:
                continue
            d, fc = meet(a, b)
            s_b = idx_after(b, fc)                # b from the frame after the cut
            if s_b >= len(b.frames):
                continue
            s_a = idx_after(a, fc)                # a's points after the cut are dropped
            fb = b.frames[s_b]
            gap_s = max(tau[fb] - tau[min(a.f1, fc)], 1.0) / fps
            h = (d / (ball * (C.sig0_bw + C.sig_gap_bw_s * gap_s))) ** 2 + C.touch
            if fb - min(a.f1, fc) >= re_gap:
                h = min(h, C.reentry)
            unc = max(0, int(upre[fb] - upre[min(a.f1, fc) + 1]))
            v = best[i] + C.w_det * ev[i][s_a] + C.miss * unc + h - C.w_det * ev[j][s_b]
            if v < cand[0]:
                cand = (v, i, (fc, s_b))
        best[j], back[j], cut[j] = cand
    end, ej = C.miss * int(upre[n]), None
    for j, a in enumerate(arcs):
        v = best[j] + C.miss * int(upre[n] - upre[a.f1 + 1])
        if v < end:
            end, ej = v, j
    seq, j = [], ej
    while j is not None:
        seq.append(j)
        j = back[j]
    seq = seq[::-1]
    out = []
    for k, j in enumerate(seq):
        first = arcs[j].f0 if cut[j] is None or k == 0 else arcs[j].frames[cut[j][1]]
        last = cut[seq[k + 1]][0] if k + 1 < len(seq) else arcs[j].f1
        out.append((j, first, min(last, arcs[j].f1)))
    return out


def link(dets: dict, tn_frames, cal, fps: float, width: int, height: int, ball: float,
         jparams: JointParams | None = None, dup_flags=None, content=None, times=None,
         arc_params: ArcParams | None = None, chain_params: ChainParams | None = None,
         link_params: LinkParams | None = None, return_debug: bool = False):
    """Same inputs and output (a Track) as core.joint_linker.link."""
    JP = jparams or JointParams()
    LP = link_params or LinkParams()
    n = len(dets["frames"])
    content = tuple(content) if content else (0, 0, width, height)
    dup = [bool(x) for x in dup_flags] if dup_flags is not None and len(dup_flags) == n else [False] * n
    usable = [not d for d in dup]
    tau = (np.asarray(times, float) * fps) if times is not None and len(times) == n else np.arange(n, dtype=float)
    pts = build_points(dets, tn_frames, cal, ball, JP, fps)
    g = _gravity({f: [(x, y, s or ball, p, 1.0) for (x, y, s, lg, st, p) in pts[f] if st == DETECTED]
                  for f in range(n)}, usable, ball)
    if g <= 0:                                  # no confirmed triples (finder-first): from free arcs
        g = arc_gravity(find_arcs(pts, tau, usable, fps, ball, 0.0, arc_params))
    arcs = find_arcs(pts, tau, usable, fps, ball, g, arc_params)
    ch = chain(arcs, n, tau, usable, fps, ball, chain_params)
    owner = {}
    for j, lo, hi in ch:
        a = arcs[j]
        for f in range(lo, hi + 1):
            if f in owner and owner[f][4] in (DETECTED, WEAK):
                continue
            q = a.pts.get(f)
            if q is not None:
                x, y, sz, lg, st, p = q
                if st not in (DETECTED, WEAK):          # finder-only: on the arc
                    x, y = a.at_t(tau[f])
                owner[f] = (float(x), float(y), float(sz if sz else ball), float(p), st)
    track = finalize(owner, n, fps, width, height, LP, dup, content, [], TAU=tau)
    if return_debug:
        return track, {"arcs": arcs, "chain": ch, "gravity": g}
    return track
