"""Joint linker: one global choice of the ball's path from YOLO, zoom finds and TrackNet together.

Detection first, then one optimization (instead of rule stages plus a TrackNet add-on):

  1. candidates   every YOLO box, zoom find and TrackNet peak, each turned into a learned
                  probability of being the ball (core.calibration). Candidates from
                  different sources at the same spot merge into one point; when YOLO and
                  TrackNet agree, the point gets an agreement bonus.
  2. path         the cost linker's exact search (Viterbi over (previous, current) points):
                    detection   -w_det · logit(p)      (sure balls are rewarded, junk costs)
                    not visible  miss per frame with no point chosen (hidden / out of view)
                    motion       distance from the gravity-aware prediction, in σ
                    hit          velocity restart (touch cost), still subject to the speed limit
                    speed/size   a teleport or a sudden size change never pays off
                    still        a near-motionless step costs (ball in play doesn't sit still)
  3. gaps         bridged as before (core.linker.finalize)

States in the output: detected / weak (from YOLO or zoom, by probability) and tracknet
(TrackNet only). Weights are fit on labelled clips (tools/archive/eval_joint.py).

Status (experimental, not the default): leave-one-clip-out on the 10 test clips it scores
96.6% on-ball vs 96.4% for rule linker + TrackNet, misses fewer frames (1.3% vs 2.8%) but
lands on the wrong spot more (2.1% vs 0.8%), mostly test34 where a second ball on the far
court is in view: local physics can't tell which ball is in play (needs both balls tracked
and the rally ball chosen globally).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from core.cost_linker import CostParams, _gravity
from core.linker import LinkParams, finalize, DETECTED, WEAK

JOINT_LINKER_VERSION = 1


@dataclass
class JointParams(CostParams):
    # defaults = best leave-one-clip-out setting on the 10 test clips (tools/archive/eval_joint.py --grid)
    w_det: float = 0.7          # weight on the detection evidence (logit of p)
    miss: float = 1.0           # per usable frame with no point chosen
    touch: float = 3.0          # a hit (velocity restart)
    agree_bonus: float = 1.5    # YOLO/zoom and TrackNet on the same spot
    vlim_bw_s: float = 180.0    # speed prior free up to this (a hard spike reaches ~34 m/s ≈ 160 bw/s)
    merge_bw: float = 1.0       # candidates closer than this (ball-widths) are one point (TrackNet's own
                                #   position error is up to ~0.7 ball-widths)
    p_min: float = 0.03         # candidates less likely than this are dropped (speed)
    top_k: int = 4              # at most this many points per frame
    logit_clip: float = 6.0
    tn_only_bias: float = 0.0   # logit shift for finder-only points (no YOLO/zoom support). The
                                # finder-first pipeline uses 0.5 (core.finder_first.FINDER_ONLY_BIAS)
    size_near_bw: float = 1.5   # TrackNet point size = YOLO boxes within this distance ...
    size_near_s: float = 0.25   # ... and this time (TrackNet itself gives no size)
    size_from_zoom: bool = False  # ... counting zoom / crop finds too (finder-first has no full-frame boxes)


def _local_sizes(dets: dict, ball: float, P: JointParams, fps: float):
    """For a TrackNet point at (f, x, y): median size of YOLO boxes near that spot and time,
    or None. TrackNet's own 'size' is borrowed from the clip, which would hide a switch to
    a smaller far-court ball from the size-continuity check."""
    boxes = [(f, c[0], c[1], max(c[2], c[3])) for f, fr in enumerate(dets["frames"]) for c in fr if c[4] >= 0.1]
    if P.size_from_zoom:
        boxes += [(int(k), c[0], c[1], max(c[2], c[3])) for k, v in (dets.get("zoom") or {}).items()
                  for c in v["cands"] if c[4] >= 0.1]
    if not boxes:
        return lambda f, x, y: None
    B = np.array(boxes)
    R, T = P.size_near_bw * ball, max(1, int(round(P.size_near_s * fps)))

    def size_at(f, x, y):
        m = (np.abs(B[:, 0] - f) <= T) & (np.hypot(B[:, 1] - x, B[:, 2] - y) <= R)
        return float(np.median(B[m, 3])) if m.sum() >= 2 else None
    return size_at


def build_points(dets: dict, tn_frames, cal, ball: float, P: JointParams, fps: float = 60.0):
    """Per frame: list of (x, y, size, logit, state, p) after calibration and merging.
    size is None where unknown (TrackNet point with no YOLO box nearby)."""
    n = len(dets["frames"])
    size_at = _local_sizes(dets, ball, P, fps)
    raw = [[] for _ in range(n)]
    for f, fr in enumerate(dets["frames"]):
        for c in fr:
            raw[f].append(("yolo", c[0], c[1], max(c[2], c[3]), c[4]))
    for k, v in (dets.get("zoom") or {}).items():
        for c in v["cands"]:
            raw[int(k)].append(("zoom", c[0], c[1], max(c[2], c[3]), c[4]))
    if tn_frames is not None:
        for f, fr in enumerate(tn_frames[:n]):
            for c in fr:
                raw[f].append(("tracknet", c[0], c[1], None, c[4]))
    out = []
    for f in range(n):
        pts = []
        for src in ("yolo", "zoom", "tracknet"):
            sel = [r for r in raw[f] if r[0] == src]
            if not sel:
                continue
            lg = cal.logit(src, [r[4] for r in sel])
            for r, l in zip(sel, lg):
                pts.append([src, r[1], r[2], r[3], float(np.clip(l, -P.logit_clip, P.logit_clip))])
        pts = [p for p in pts if 1 / (1 + math.exp(-p[4])) >= P.p_min]
        pts.sort(key=lambda p: -p[4])
        merged = []                                   # [x, y, size, logit, has_yolo, has_tn, state_src]
        for src, x, y, s, l in pts:
            for m in merged:
                if math.hypot(m[0] - x, m[1] - y) <= P.merge_bw * ball:
                    if src == "tracknet":
                        m[5] = True
                    else:
                        if not m[4]:                  # a YOLO/zoom box gives the precise position and size
                            m[0], m[1], m[2] = x, y, s
                        m[4] = True
                    break
            else:
                merged.append([x, y, s if src != "tracknet" else size_at(f, x, y), l,
                               src != "tracknet", src == "tracknet", src])
        res = []
        for x, y, s, l, hy, ht, src in merged[:P.top_k]:
            lg = l + (P.agree_bonus if (hy and ht) else 0.0) - (P.tn_only_bias if (ht and not hy) else 0.0)
            p = 1 / (1 + math.exp(-lg))
            state = (DETECTED if p >= 0.5 else WEAK) if hy else "tracknet"
            res.append((x, y, s, lg, state, p))
        out.append(res)
    return out


def link(dets: dict, tn_frames, cal, fps: float, width: int, height: int, ball: float,
         params: JointParams | None = None, dup_flags=None, content=None, times=None,
         link_params: LinkParams | None = None, exclude: dict | None = None, pts_by_frame=None):
    """exclude: {frame: [(x, y), ...]} points to drop (e.g. rival objects found by the
    tracklet step); pts_by_frame: precomputed build_points output (saves recomputing)."""
    C = params or JointParams()
    LP = link_params or LinkParams()
    n = len(dets["frames"])
    content = tuple(content) if content else (0, 0, width, height)
    dup = [bool(x) for x in dup_flags] if dup_flags is not None and len(dup_flags) == n else [False] * n
    usable = [not d for d in dup]
    upre = np.concatenate([[0], np.cumsum(usable)])
    vlim, still_speed, size_rate_gap = C.vlim_bw_s / fps, C.still_speed_bw_s / fps, C.size_rate_gap_s / fps

    def u_between(fa, fb):
        return max(0, int(upre[fb] - upre[fa + 1]))

    if pts_by_frame is None:
        pts_by_frame = build_points(dets, tn_frames, cal, ball, C, fps)
    if exclude:
        r = 0.5 * ball
        pts_by_frame = [[q for q in pts_by_frame[f] if not any(math.hypot(q[0] - x, q[1] - y) <= r
                                                                for x, y in exclude.get(f, ()))]
                        for f in range(len(pts_by_frame))]
    pts = [(x, y, s, lg, st, p, f) for f in range(n) if not dup[f] for (x, y, s, lg, st, p) in pts_by_frame[f]]
    if not pts:
        return finalize({}, n, fps, width, height, LP, dup, content, [])
    X = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
    KNOWN = np.array([p[2] is not None for p in pts])
    S = np.array([p[2] if p[2] is not None else ball for p in pts], float)   # unknown size: clip ball for speed units
    LG = np.array([p[3] for p in pts]); F = np.array([p[6] for p in pts])
    # real time in nominal frames (uneven phone timing): motion, speed and size use TAU,
    # counting frames (miss costs, search window) uses F
    TAUF = (np.asarray(times, float) * fps) if times is not None and len(times) == n else np.arange(n, dtype=float)
    TA = TAUF[F]
    det = -C.w_det * LG
    med = ball
    gy = _gravity({f: [(x, y, s, 1 / (1 + math.exp(-lg)), 1.0) for (x, y, s, lg, st, p) in pts_by_frame[f] if st == DETECTED]
                   for f in range(n)}, usable, med)
    P = len(pts)
    G = max(2, int(round(C.short_gap_s * fps)))

    def speed_cost(dist, gap, size):
        v = dist / max(gap, 1) / size
        return C.k_speed * max(0.0, v / vlim - 1.0) ** 2

    INF = float("inf")
    D, back = {}, {}
    bend = np.full(P, INF)
    bend_state = [None] * P
    preds_of = [[] for _ in range(P)]
    for c in range(P):
        fc = int(F[c])
        best, arg = C.start + C.miss * int(upre[fc]), ("start",)
        earlier = np.nonzero(F < fc - G)[0]
        if len(earlier):
            gaps = np.maximum(TA[c] - TA[earlier], 1e-3)
            dist = np.hypot(X[c] - X[earlier], Y[c] - Y[earlier])
            sz = np.maximum(S[earlier], S[c])
            v = dist / np.maximum(gaps, 1) / sz
            sp = C.k_speed * np.maximum(0.0, v / vlim - 1.0) ** 2
            dlog = np.abs(np.log(S[c] / S[earlier])) * (KNOWN[c] & KNOWN[earlier])
            sp = sp + C.k_size * np.maximum(0.0, dlog - (C.size_rate + size_rate_gap * gaps)) ** 2
            between = upre[fc] - upre[F[earlier] + 1]
            tot = bend[earlier] + C.miss * between + C.reentry + sp
            k = int(np.argmin(tot))
            if tot[k] < best:
                best, arg = float(tot[k]), ("reentry", int(earlier[k]))
        st = (-1, c)
        D[st] = best + det[c]
        back[st] = arg
        preds_of[c].append(st)
        lo = fc - G
        for b in range(c - 1, -1, -1):
            fb = int(F[b])
            if fb >= fc:
                continue
            if fb < lo:
                break
            gap = max(float(TA[c] - TA[b]), 1e-3)
            dist = float(np.hypot(X[c] - X[b], Y[c] - Y[b]))
            size = max(S[b], S[c])
            sp = speed_cost(dist, gap, size)
            still = C.still if dist / gap / size < still_speed else 0.0
            dlog = abs(math.log(S[c] / S[b])) if (KNOWN[c] and KNOWN[b]) else 0.0
            szc = C.k_size * max(0.0, dlog - (C.size_rate + size_rate_gap * gap)) ** 2
            base = C.miss * u_between(fb, fc) + det[c] + still + szc
            best, arg = (bend[b] + C.touch + sp, ("touch", bend_state[b])) if bend[b] < INF else (INF, None)
            for sa in preds_of[b]:
                prev = D[sa]
                if prev >= best:
                    continue
                a = sa[0]
                if a == -1:
                    cost = prev + sp
                else:
                    dt = max(float(TA[b] - TA[a]), 1e-3)
                    vx = (X[b] - X[a]) / dt
                    vy = (Y[b] - Y[a]) / dt + 0.5 * gy * dt
                    px = X[b] + vx * gap
                    py = Y[b] + vy * gap + 0.5 * gy * gap * gap
                    sigma = size * (C.sigma0 + C.sigma_gap * gap) + C.sigma_speed * math.hypot(vx, vy)
                    err = math.hypot(X[c] - px, Y[c] - py) / sigma
                    cost = prev + err * err + sp
                if cost < best:
                    best, arg = cost, ("motion", sa)
            if best < INF:
                st = (b, c)
                D[st] = best + base
                back[st] = arg
                preds_of[c].append(st)
        for st in preds_of[c]:
            if D[st] < bend[c]:
                bend[c], bend_state[c] = D[st], st

    best_end, end_pt = C.miss * int(upre[n]), None
    for b in range(P):
        tot = bend[b] + C.miss * int(upre[n] - upre[int(F[b]) + 1])
        if tot < best_end:
            best_end, end_pt = tot, b
    chosen, touches = [], []
    st = bend_state[end_pt] if end_pt is not None else None
    while st is not None:
        chosen.append(st[1])
        kind = back[st]
        if kind[0] == "motion":
            st = kind[1]
        elif kind[0] == "touch":
            touches.append(int(F[st[0]]))
            st = kind[1]
        elif kind[0] == "reentry":
            st = bend_state[kind[1]]
        else:
            st = None
    owner = {}
    for i in chosen[::-1]:
        x, y, s, lg, state, p, f = pts[i]
        owner[int(f)] = (float(x), float(y), float(s if s is not None else ball), float(p), state)
    track = finalize(owner, n, fps, width, height, LP, dup, content, [])
    track.touch_frames = sorted(set(touches))
    if getattr(C, "_debug", False):
        track._debug = {"D": D, "back": back, "F": F, "X": X, "Y": Y, "LG": LG, "det": det, "bend": bend,
                        "bend_state": bend_state, "chosen": chosen[::-1]}
    return track
