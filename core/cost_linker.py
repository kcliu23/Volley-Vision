"""Cost-based linker: pick the ball's path through the whole clip as one optimization.

Every frame offers its candidates (YOLO + zoom finds) or "no ball". A path picks at
most one per frame; its cost is the sum of

  detection   how ball-like a chosen candidate is   λ·(-ln conf) + size/shape penalties
  miss        a usable frame with no ball chosen    β per frame
  motion      distance from the gravity-aware prediction made from the previous two
              chosen points, in units of σ (which grows with gap length and speed)
  touch       instead of following the prediction, pay T and restart the velocity at the
              previous point (a hit); still subject to the speed prior
  speed       prior on implied speed, in ball-widths/frame: free up to vlim, then rising
  still       the ball barely moving between two chosen points (a sign, a knee, a ball
              resting): a ball in play doesn't do that
  re-entry    picking the ball up again after a long gap (> G frames): fixed cost + speed prior

The cheapest path is found exactly with dynamic programming (Viterbi) over states
(previous point, current point), so a two-point velocity is always available.
One path = one ball: switching to another ball costs a huge motion/speed penalty.

Gap filling, repeated frames and output format are shared with core.linker.finalize.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from core.linker import LinkParams, finalize, DETECTED, WEAK

COST_LINKER_VERSION = 1   # bump when behaviour changes (invalidates cached zoom searches)


@dataclass
class CostParams:
    lam: float = 0.6          # weight on -ln(conf)
    miss: float = 2.2         # β: cost of a usable frame without the ball
    start: float = 1.0        # S: cost of starting the path
    reentry: float = 3.0      # R: picking the ball up again after a long gap
    touch: float = 4.0        # T: a hit (velocity restart)
    sigma0: float = 0.35      # motion tolerance, ball-widths, at 1 frame
    sigma_gap: float = 0.10   # + per frame of gap
    sigma_speed: float = 0.40 # + this fraction of the per-frame speed (timing jitter, uneven phone frame times)
    vlim: float = 2.5         # speed prior: free up to this many ball-widths/frame
    k_speed: float = 100.0    # ... then k·(v/vlim - 1)^2 (steep: a teleport must never pay off)
    size_rate: float = 0.12   # size continuity: box size may change by this (log) ...
    size_rate_gap: float = 0.03   # ... + this per frame of gap, before it costs
    k_size: float = 60.0      # cost k·(excess log-ratio)^2
    still_speed: float = 0.1  # ball-widths/frame below which a step counts as "still"
    still: float = 3.2        # cost of a still step (> miss: a still object is never worth following)
    short_gap_s: float = 0.30 # G: gaps up to this keep the velocity (motion model applies)
    size_tol: float = 1.8     # no size penalty within ×/÷ this of the clip's ball size
    min_aspect: float = 0.6   # shape penalty below this short/long side ratio
    strong_conf: float = 0.40 # only for labelling output states detected / weak
    min_conf: float = 0.05


def _gravity(pts_by_frame, usable, med):
    """Per-clip vertical acceleration in px/frame², from confident consecutive triples."""
    ays = []
    for f in range(len(usable) - 2):
        if not (usable[f] and usable[f + 1] and usable[f + 2]):
            continue
        tri = []
        for g in (f, f + 1, f + 2):
            strong = [p for p in pts_by_frame.get(g, []) if p[4] >= 0.5]
            if len(strong) != 1:
                break
            tri.append(strong[0])
        if len(tri) != 3:
            continue
        if max(np.hypot(tri[1][0] - tri[0][0], tri[1][1] - tri[0][1]),
               np.hypot(tri[2][0] - tri[1][0], tri[2][1] - tri[1][1])) > 2 * med:
            continue
        ays.append(tri[0][1] - 2 * tri[1][1] + tri[2][1])
    if len(ays) < 10:
        return 0.0
    return float(np.clip(np.median(ays), 0.0, 3.0))


def link(frames: list, fps: float, width: int, height: int, params: CostParams | None = None,
         dup_flags: list[bool] | None = None, content: list | None = None,
         link_params: LinkParams | None = None, return_debug: bool = False):
    C = params or CostParams()
    LP = link_params or LinkParams()
    n = len(frames)
    content = tuple(content) if content else (0, 0, width, height)

    # repeated frames carry no new information (and no time passes on screen)
    if dup_flags is not None and len(dup_flags) == n:
        dup = [bool(x) for x in dup_flags]
    else:
        dup = [False] * n
        for i in range(1, n):
            a, b = np.asarray(frames[i - 1]).reshape(-1, 5), np.asarray(frames[i]).reshape(-1, 5)
            if len(a) and len(b) and a.shape == b.shape and np.allclose(a[:, :2], b[:, :2]):
                dup[i] = True
    usable = [not d for d in dup]
    upre = np.concatenate([[0], np.cumsum(usable)])          # usable frames in [0, f) = upre[f]

    def u_between(fa, fb):                                    # usable frames strictly between
        return max(0, int(upre[fb] - upre[fa + 1]))

    # points: (x, y, size, conf, aspect, frame)
    raw = []
    for f in range(n):
        if dup[f]:
            continue
        for c in np.asarray(frames[f], float).reshape(-1, 5):
            if c[4] < C.min_conf:
                continue
            size = max(c[2], c[3])
            raw.append((c[0], c[1], size, c[4], min(c[2], c[3]) / max(size, 1e-6), f))
    if not raw:
        return finalize({}, n, fps, width, height, LP, dup, content, [])
    strong_sizes = [p[2] for p in raw if p[3] >= C.strong_conf]
    med = float(np.median(strong_sizes)) if strong_sizes else float(np.median([p[2] for p in raw]))
    by_frame: dict[int, list] = {}
    for p in raw:
        by_frame.setdefault(p[5], []).append(p)
    gy = _gravity(by_frame, usable, med)

    pts = sorted(raw, key=lambda p: p[5])
    P = len(pts)
    X = np.array([p[0] for p in pts]); Y = np.array([p[1] for p in pts])
    S = np.array([p[2] for p in pts]); CF = np.array([p[3] for p in pts])
    AR = np.array([p[4] for p in pts]); F = np.array([p[5] for p in pts])

    det = (C.lam * -np.log(np.clip(CF, 1e-4, 1.0))
           + 2.0 * np.maximum(0.0, np.abs(np.log(S / med)) - np.log(C.size_tol))
           + 3.0 * np.maximum(0.0, C.min_aspect - AR))
    G = max(2, int(round(C.short_gap_s * fps)))

    def speed_cost(dist, gap, size):
        v = dist / max(gap, 1) / size
        return C.k_speed * max(0.0, v / C.vlim - 1.0) ** 2

    INF = float("inf")
    # states: dict keyed by (a, b) with a = -1 for "velocity unknown" (start / re-entry)
    D: dict[tuple, float] = {}
    back: dict[tuple, tuple] = {}         # state -> (previous state, kind)
    bend = np.full(P, INF)                # best cost of any path ending at point b
    bend_state: list = [None] * P
    preds_of: list[list[int]] = [[] for _ in range(P)]   # states (a,b) that exist for b

    first_idx_of_frame = {}
    for i, f in enumerate(F):
        first_idx_of_frame.setdefault(int(f), i)

    for c in range(P):
        fc = int(F[c])
        # ── state (-1, c): start fresh, or re-enter after any earlier point ──
        best, arg = C.start + C.miss * int(upre[fc]), ("start",)
        earlier = np.nonzero(F < fc - G)[0]            # re-entry only after a long gap
        if len(earlier):
            gaps = fc - F[earlier]
            dist = np.hypot(X[c] - X[earlier], Y[c] - Y[earlier])
            sz = np.maximum(S[earlier], S[c])
            v = dist / np.maximum(gaps, 1) / sz
            sp = C.k_speed * np.maximum(0.0, v / C.vlim - 1.0) ** 2
            dlog = np.abs(np.log(S[c] / S[earlier]))
            sp = sp + C.k_size * np.maximum(0.0, dlog - (C.size_rate + C.size_rate_gap * gaps)) ** 2
            between = np.array([u_between(int(F[b]), fc) for b in earlier])
            tot = bend[earlier] + C.miss * between + C.reentry + sp
            k = int(np.argmin(tot))
            if tot[k] < best:
                best, arg = float(tot[k]), ("reentry", int(earlier[k]))
        st = (-1, c)
        D[st] = best + det[c]
        back[st] = arg
        preds_of[c].append(st)

        # ── states (b, c): b within the short-gap window ──
        lo = fc - G
        for b in range(c - 1, -1, -1):
            fb = int(F[b])
            if fb >= fc:
                continue
            if fb < lo:
                break
            gap = fc - fb
            dist = float(np.hypot(X[c] - X[b], Y[c] - Y[b]))
            size = max(S[b], S[c])
            sp = speed_cost(dist, gap, size)
            still = C.still if dist / gap / size < C.still_speed else 0.0
            dlog = abs(np.log(S[c] / S[b]))
            szc = C.k_size * max(0.0, dlog - (C.size_rate + C.size_rate_gap * gap)) ** 2
            base = C.miss * u_between(fb, fc) + det[c] + still + szc
            # option 1: a touch at b (velocity restarts), from the best path ending at b
            best, arg = (bend[b] + C.touch + sp, ("touch", bend_state[b])) if bend[b] < INF else (INF, None)
            # option 2: follow the motion model from each state (a, b)
            for sa in preds_of[b]:
                prev = D[sa]
                if prev >= best:
                    continue
                a = sa[0]
                if a == -1:
                    cost = prev + sp
                else:
                    fa = int(F[a])
                    dt = fb - fa
                    vx = (X[b] - X[a]) / dt
                    vy = (Y[b] - Y[a]) / dt + 0.5 * gy * dt          # velocity at b (const. accel.)
                    px = X[b] + vx * gap
                    py = Y[b] + vy * gap + 0.5 * gy * gap * gap
                    sigma = size * (C.sigma0 + C.sigma_gap * gap) + C.sigma_speed * np.hypot(vx, vy)
                    err = np.hypot(X[c] - px, Y[c] - py) / sigma
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

    # ── end: best last point (or nothing at all) ──
    total_usable = int(upre[n])
    best_end, end_pt = C.miss * total_usable, None
    for b in range(P):
        tot = bend[b] + C.miss * int(upre[n] - upre[int(F[b]) + 1])
        if tot < best_end:
            best_end, end_pt = tot, b

    # ── backtrack ──
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
    chosen = chosen[::-1]

    owner = {}
    for i in chosen:
        owner[int(F[i])] = (float(X[i]), float(Y[i]), float(S[i]), float(CF[i]),
                            DETECTED if CF[i] >= C.strong_conf else WEAK)
    track = finalize(owner, n, fps, width, height, LP, dup, content, [])
    track.touch_frames = sorted(set(touches))
    if return_debug:
        return track, {"cost": best_end, "gravity_px": gy, "ball_size": med, "points": P}
    return track
