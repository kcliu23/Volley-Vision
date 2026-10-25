"""Tracklet linker: choose which tracklets are the ball in play, one at a time, over the whole clip.

Step 2 of the joint linker with tracklets (step 1: core.tracklets). The chosen sequence of
tracklets minimizes

    Σ evidence      -w_det · logit(p) for every point kept (sure balls are rewarded)
  + Σ not visible   miss per usable frame no chosen tracklet covers
  + Σ hand-overs    for each consecutive pair (A ends, B starts), how plausibly ONE ball does that:
                      arcs meet  A's arc forward and B's arc backward come within d of each
                                 other somewhere in the gap; cost (d/σ)², σ growing with the gap.
                                 Meeting at B's start = A simply continues its flight; meeting
                                 earlier = a hit in the gap (+ touch cost).
                      size       known sizes must agree (tolerance grows with the gap)
                      re-entry   only after a long gap (ball out of view): fixed cost + size check
  + start / end     uncovered frames before the first / after the last tracklet

Only one tracklet at a time (a small overlap is trimmed), so two balls in view at once can't
both be "the ball": the chain that connects is chosen. Solved exactly by dynamic programming
over tracklets (O(N²), N ≈ tens). Gaps inside the chosen chain are then bridged as before
(core.linker.finalize).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from core.cost_linker import _gravity
from core.joint_linker import JointParams, build_points
from core.linker import LinkParams, finalize, DETECTED, WEAK
from core.tracklets import TrackletParams, build as build_tracklets

TRACKLET_LINKER_VERSION = 1


@dataclass
class AssocParams:
    w_det: float = 0.7
    miss: float = 1.0
    touch: float = 3.0
    sig0_bw: float = 0.6          # hand-over tolerance, ball-widths ...
    sig_gap_bw_s: float = 4.0     # ... + this per second of gap
    size_tol: float = 0.3         # |log size ratio| free ...
    size_tol_s: float = 1.0       # ... + this per second of gap
    k_size: float = 40.0
    reentry: float = 6.0          # pick the ball up again after a long gap, anywhere
    reentry_gap_s: float = 0.6    # ... only after at least this long
    max_overlap_s: float = 0.1    # consecutive tracklets may overlap this much (trimmed)


def _meet(A, B, gy, f_lo, f_hi, k=5):
    """Smallest distance between A's arc (forward) and B's arc (backward) over frames
    f_lo..f_hi, and the frame where it happens."""
    best, bf = float("inf"), f_hi
    for f in range(f_lo, f_hi + 1):
        ax, ay, _ = A.predict(f, gy, k)
        bx, by, _ = B.predict(f, gy, k)
        d = math.hypot(ax - bx, ay - by)
        if d < best:
            best, bf = d, f
    return best, bf


def link(dets: dict, tn_frames, cal, fps: float, width: int, height: int, ball: float,
         params: AssocParams | None = None, jparams: JointParams | None = None,
         tparams: TrackletParams | None = None, dup_flags=None, content=None, times=None,
         link_params: LinkParams | None = None, return_debug: bool = False):
    A_ = params or AssocParams()
    JP = jparams or JointParams()
    LP = link_params or LinkParams()
    n = len(dets["frames"])
    content = tuple(content) if content else (0, 0, width, height)
    dup = [bool(x) for x in dup_flags] if dup_flags is not None and len(dup_flags) == n else [False] * n
    usable = [not d for d in dup]
    upre = np.concatenate([[0], np.cumsum(usable)])

    def uncovered(fa, fb):                    # usable frames strictly between fa and fb
        return max(0, int(upre[fb] - upre[fa + 1])) if fb > fa else 0

    pts = build_points(dets, tn_frames, cal, ball, JP, fps)
    gy = _gravity({f: [(x, y, s or ball, p, 1.0) for (x, y, s, lg, st, p) in pts[f] if st == DETECTED]
                   for f in range(n)}, usable, ball)
    trs = build_tracklets(pts, fps, ball, gy, dup, tparams)
    if not trs:
        return finalize({}, n, fps, width, height, LP, dup, content, [])
    N = len(trs)
    max_ov = max(0, int(round(A_.max_overlap_s * fps)))
    re_gap = int(round(A_.reentry_gap_s * fps))

    def cost_from(t, start_idx):
        return float(-A_.w_det * np.sum(t.logits[start_idx:]))

    def handover(a, b, start_b):
        """Cost of one ball going from tracklet a's end to tracklet b's (trimmed) start."""
        fa, fb = a.f1, b.frames[start_b]
        gap_s = max(fb - fa, 1) / fps
        sig = ball * (A_.sig0_bw + A_.sig_gap_bw_s * gap_s)
        d_meet, f_meet = _meet(a, b, gy, fa, fb)
        cont = (d_meet / sig) ** 2 + (A_.touch if f_meet < fb - 1 else 0.0)
        sa, sb = a.size(-1), b.size(0)
        szc = 0.0
        if sa and sb:
            ex = abs(math.log(sb / sa)) - (A_.size_tol + A_.size_tol_s * gap_s)
            szc = A_.k_size * max(0.0, ex) ** 2
        best = cont
        if fb - fa >= re_gap:
            best = min(best, A_.reentry)
        return best + szc

    # DP over tracklets sorted by start
    INF = float("inf")
    best = [INF] * N
    back = [None] * N
    trim = [0] * N
    for j, b in enumerate(trs):
        cand = (A_.miss * int(upre[b.f0]) + cost_from(b, 0), None, 0)     # j is the first tracklet
        for i in range(j):
            a = trs[i]
            if best[i] == INF or a.f1 >= b.f1:
                continue
            if a.f1 >= b.f0:                                            # overlap: trim b's start
                if a.f1 - b.f0 + 1 > max_ov:
                    continue
                s = next(k for k, f in enumerate(b.frames) if f > a.f1)
            else:
                s = 0
            c = best[i] + A_.miss * uncovered(a.f1, b.frames[s]) + handover(a, b, s) + cost_from(b, s)
            if c < cand[0]:
                cand = (c, i, s)
        best[j], back[j], trim[j] = cand
    total_none = A_.miss * int(upre[n])
    end_cost, end_j = total_none, None
    for j, t in enumerate(trs):
        c = best[j] + A_.miss * int(upre[n] - upre[t.f1 + 1])
        if c < end_cost:
            end_cost, end_j = c, j
    chain = []
    j = end_j
    while j is not None:
        chain.append((j, trim[j]))
        j = back[j]
    chain = chain[::-1]
    if return_debug == "rivals":
        return trs, [j for j, _ in chain], pts, gy
    owner = {}
    for j, s in chain:
        t = trs[j]
        for f, x, y, sz, lg, st in list(zip(t.frames, t.xs, t.ys, t.sizes, t.logits, t.states))[s:]:
            owner[int(f)] = (float(x), float(y), float(sz if sz is not None else ball),
                             float(1 / (1 + math.exp(-lg))), st)
    track = finalize(owner, n, fps, width, height, LP, dup, content, [])
    if return_debug:
        return track, {"tracklets": trs, "chain": [j for j, _ in chain], "gravity": gy}
    return track


def rivals(trs, chain_ids, ball: float, min_overlap: int = 3, far_bw: float = 2.0) -> dict:
    """Points of tracklets that run at the same time as a chosen tracklet but somewhere else:
    another object (a second ball, a head, a light), not the ball in play.
    Returns {frame: [(x, y), ...]}."""
    chosen = [trs[j] for j in chain_ids]
    pos = {}
    for t in chosen:
        for f, x, y in zip(t.frames, t.xs, t.ys):
            pos[f] = (x, y)
    out = {}
    for k, t in enumerate(trs):
        if k in chain_ids:
            continue
        both = [(f, x, y) for f, x, y in zip(t.frames, t.xs, t.ys) if f in pos]
        if len(both) < min_overlap:
            continue
        far = [math.hypot(x - pos[f][0], y - pos[f][1]) > far_bw * ball for f, x, y in both]
        if sum(far) >= 0.8 * len(both):
            # only where the chosen ball is somewhere else at that moment; elsewhere the
            # tracklet stays available (it may be the ball when nothing else is chosen)
            for (f, x, y), is_far in zip(both, far):
                if is_far:
                    out.setdefault(f, []).append((x, y))
    return out


def still_objects(pts_by_frame: list, ball: float, fps: float, still_s: float = 0.5,
                  radius_bw: float = 0.5, moving: set | None = None, vlim_bw_s: float = 180.0) -> dict:
    """Points of objects that stay put: something seen within radius_bw of one spot across
    at least still_s seconds (in at least min_frac of those frames) is a fixture, a sign or a
    resting ball, not the ball in play (a ball in the air can't hover). Works on all points,
    so a flickering fixture is caught even when it never forms one long tracklet (seen at the
    same spot at two moments >= still_s apart is enough). Points in `moving` (they belong to a
    tracklet that is moving) are never flagged, so a ball passing the same spot twice is safe.
    Returns {frame: [(x, y), ...]}."""
    span = max(2, int(round(still_s * fps)))
    allp = [(f, q[0], q[1]) for f, fr in enumerate(pts_by_frame) for q in fr]
    if not allp:
        return {}
    A = np.array(allp)
    r = radius_bw * ball
    out = {}
    moving = moving or set()
    by_f = {}
    for f, x, y in allp:
        by_f.setdefault(int(f), []).append((x, y))

    step = vlim_bw_s / fps * ball                         # farthest a ball can move per frame

    def moves(f, x, y):
        """A point is part of something moving if the adjacent frames show a displaced point
        within ball speed, and it isn't itself sitting at the same spot in most of them."""
        same = disp = 0
        for g in (f - 2, f - 1, f + 1, f + 2):
            near = [math.hypot(qx - x, qy - y) for qx, qy in by_f.get(g, ())]
            same += any(d <= 0.5 * ball for d in near)
            disp += any(0.5 * ball * abs(g - f) < d <= step * abs(g - f) for d in near)
        return disp > 0 and same < 2

    for f, x, y in allp:
        if (f, round(x, 1), round(y, 1)) in moving or moves(int(f), x, y):
            continue
        m = np.hypot(A[:, 1] - x, A[:, 2] - y) <= r
        fr = np.unique(A[m, 0])
        if len(fr) >= 2 and fr.max() - fr.min() >= span:
            out.setdefault(int(f), []).append((x, y))
    return out


def link_hybrid(dets: dict, tn_frames, cal, fps: float, width: int, height: int, ball: float,
                params: AssocParams | None = None, jparams: JointParams | None = None,
                tparams: TrackletParams | None = None, dup_flags=None, content=None, times=None,
                link_params: LinkParams | None = None):
    """Tracklets decide WHICH object is the ball: rival objects running at the same time as
    the chosen chain, and objects that stay put (fixtures, a resting ball), are removed; the
    frame-by-frame joint linker then picks the path from what's left."""
    from core.joint_linker import link as joint_link
    trs, chain_ids, pts, gy = link(dets, tn_frames, cal, fps, width, height, ball, params, jparams, tparams,
                                   dup_flags, content, times, link_params, return_debug="rivals")
    ex = rivals(trs, chain_ids, ball)
    moving = {(f, round(x, 1), round(y, 1)) for t in trs if len(t.frames) >= 3
              and t.summary(ball, fps)["speed_bw_s"] >= 3.0 for f, x, y in zip(t.frames, t.xs, t.ys)}
    for f, xy in still_objects(pts, ball, fps, moving=moving).items():
        ex.setdefault(f, []).extend(xy)
    return joint_link(dets, tn_frames, cal, fps, width, height, ball, jparams, dup_flags=dup_flags,
                      content=content, times=times, link_params=link_params, exclude=ex, pts_by_frame=pts)
