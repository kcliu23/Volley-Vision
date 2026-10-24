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

# Bump whenever linking behaviour changes: cached zoom searches were steered by the
# old track, so they are discarded when this changes.
LINKER_VERSION = 8


@dataclass
class LinkParams:
    strong_conf: float = 0.40     # can seed / anchor a track on its own
    min_conf: float = 0.05        # ignore anything below (should match the dump)
    size_lo: float = 0.25         # box size must be within [lo, hi] x clip median (far balls are small)
    size_hi: float = 2.5
    tol_size: float = 0.8         # acceptance radius = tol_size * ball size ...
    tol_min_bw: float = 0.4       # ... at least this many ball-widths ...
    tol_speed: float = 0.6        # ... + this * per-frame motion (timing jitter)
    gate_growth: float = 0.5      # radius grows by this fraction per missed frame
    max_miss_s: float = 0.15      # extension/linking gives up after this long without a hit
    fit_pts: int = 6              # points used for local prediction
    min_tracklet: int = 3         # strong tracklets shorter than this are dropped
    static_s: float = 0.5         # same spot for this long = stationary object, not the ball in play
    static_min_s: float = 0.13    # ... and seen for at least this long in total (>= 4 frames)
    vmax_bw_s: float = 180.0      # fastest plausible ball motion, ball-widths per second (~38 m/s: a hard spike)
    jump_margin: float = 1.0      # + this many ball-widths of slack (detection jitter)
    size_jump: float = 0.4        # box size may change by up to e^0.4 = 1.5x between nearby points ...
    size_jump_rate_s: float = 0.6 # ... plus this (log) per second apart
    min_aspect: float = 0.4       # box short side / long side (blur elongates; boxes cut off at the picture edge are exempt)
    weak_chain_min: int = 5     # weak-only chain: min points at 60 fps (repeated frames don't count) ...
    weak_chain_min_s: float = 0.07  # ... or this long at low frame rates (never fewer than 3 points)
    weak_chain_conf: float = 0.6  # ... and min summed confidence
    max_bridge_s: float = 0.5     # longest gap filled by interpolation
    edge_margin: float = 0.0      # bridged path must stay this far inside the frame (px)
    min_fit_span_s: float = 0.05  # local arc fits need at least this much time for curvature
    # derived per clip from the frame rate (not set by hand): per-frame equivalents
    vmax_sizes: float = 0.0       # vmax_bw_s / fps
    size_jump_rate: float = 0.0   # size_jump_rate_s / fps


@dataclass
class Track:
    fps: float
    width: int
    height: int
    state: list[str]
    pos: list[tuple[float, float] | None]
    conf: list[float | None]
    dup: list[bool] = field(default_factory=list)
    content: tuple = ()                             # picture area (x0, y0, x1, y1) inside the frame
    tau: list = field(default_factory=list)         # real time per frame, in nominal frames (times * fps)
    size: list = field(default_factory=list)        # ball box size (px) per frame where observed
    tentative: list = field(default_factory=list)   # [[(frame, x, y), ...], ...] unconfirmed weak chains

    def summary(self) -> dict:
        from collections import Counter
        c = Counter(self.state)
        n = len(self.state)
        return {"frames": n, **{k: c.get(k, 0) for k in (DETECTED, WEAK, "tracknet", FILLED, NONE)},
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

def _predict(ts, xs, ys, t, tau=None, min_span=3.0):
    """Quadratic (or lower order with few points) extrapolation to time t.
    ts/t are frame indices; with tau, they're mapped to real time (in nominal frames)."""
    if tau is not None:
        ts, t = np.asarray(tau)[np.asarray(ts, int)], float(np.interp(t, np.arange(len(tau)), tau))
    ts = np.asarray(ts, float)
    deg = 2 if len(ts) >= 4 else (1 if len(ts) >= 2 else 0)
    if deg == 0:
        return xs[0], ys[0], 0.0
    span = ts.max() - ts.min()
    if deg == 2 and span < min_span:   # too short a baseline for curvature
        deg = 1
    px = np.polyfit(ts - t, xs, deg)
    py = np.polyfit(ts - t, ys, deg)
    vx = px[-2] if deg >= 1 else 0.0
    vy = py[-2] if deg >= 1 else 0.0
    return float(px[-1]), float(py[-1]), float(np.hypot(vx, vy))


def _velocity(ts, xs, ys, t, tau=None, min_span=3.0):
    """Local velocity (px per nominal frame) at time t from a quadratic/linear fit."""
    if tau is not None:
        ts, t = np.asarray(tau)[np.asarray(ts, int)], float(np.interp(t, np.arange(len(tau)), tau))
    ts = np.asarray(ts, float)
    deg = 2 if len(ts) >= 4 and (ts.max() - ts.min()) >= min_span else 1
    if len(ts) < 2:
        return 0.0, 0.0
    px = np.polyfit(ts - t, xs, deg)
    py = np.polyfit(ts - t, ys, deg)
    return float(px[-2]), float(py[-2])


# ── main entry ───────────────────────────────────────────────────────────────

def link(frames: list, fps: float, width: int, height: int, params: LinkParams | None = None,
         dup_flags: list[bool] | None = None, content: list | None = None,
         times: list | None = None) -> Track:
    """frames: list (one per video frame) of candidate lists [cx, cy, w, h, conf].

    dup_flags: per-frame "this video frame repeats the previous image" (from the
    detection pass, image-based). Without it, repeats are guessed from identical
    candidate positions, which misses repeats where nothing was detected."""
    from dataclasses import replace as _replace
    P = params or LinkParams()
    P = _replace(P, vmax_sizes=P.vmax_bw_s / fps, size_jump_rate=P.size_jump_rate_s / fps)
    n = len(frames)
    # time axis: real timestamps in nominal frames (= frame index for evenly timed video)
    TAU = np.asarray(times, float) * fps if times is not None and len(times) == n else np.arange(n, dtype=float)
    min_span = max(3.0, P.min_fit_span_s * fps)

    def pred(fs, xs, ys, t):
        return _predict(fs, xs, ys, t, TAU, min_span)
    # the actual picture inside the video frame (black side/top bars excluded)
    cx0, cy0, cx1, cy1 = content if content else (0, 0, width, height)
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
        a = np.minimum(C[i][:, 2], C[i][:, 3]) / np.maximum(s, 1e-6)
        x, y, w, h = C[i][:, 0], C[i][:, 1], C[i][:, 2], C[i][:, 3]
        at_edge = ((x - w / 2 <= cx0 + 2) | (y - h / 2 <= cy0 + 2) | (x + w / 2 >= cx1 - 2) | (y + h / 2 >= cy1 - 2))
        C[i] = C[i][(s >= P.size_lo * med) & (s <= P.size_hi * med) & ((a >= P.min_aspect) | at_edge)]
    # Stationary objects: something detected at the same spot across >= static_s
    # of video is not a ball in play (spare ball on the floor, a sign, a vent).
    # A ball in flight passes any given spot in a few frames.
    pts_all = [(i, k, C[i][k]) for i in range(n) if not dup[i] for k in range(len(C[i]))]
    if pts_all:
        arr = np.array([[i, c[0], c[1], max(c[2], c[3])] for i, _, c in pts_all])
        drop = set()
        span_need = P.static_s * fps
        for j, (i, k, c) in enumerate(pts_all):
            r = max(0.3 * max(c[2], c[3]), 0.1 * med)
            near = arr[np.hypot(arr[:, 1] - c[0], arr[:, 2] - c[1]) <= r]
            fr = np.unique(near[:, 0])
            if len(fr) >= max(4, int(round(P.static_min_s * fps))) and fr.max() - fr.min() >= span_need:
                drop.add((i, k))
        if drop:
            for i in range(n):
                keep = [k for k in range(len(C[i])) if (i, k) not in drop]
                C[i] = C[i][keep]
            # repeated frames carry the same candidates: drop there too
            for i in range(1, n):
                if dup[i]:
                    C[i] = C[i - 1].copy()
    usable = [i for i in range(n) if not dup[i]]
    max_miss = max(2, int(round(P.max_miss_s * fps)))
    # point-count minimums shrink at low frame rates (same duration, fewer frames)
    chain_min = max(3, min(P.weak_chain_min, int(round(P.weak_chain_min_s * fps))))

    def tol(size, speed, missed=0):
        r = max(P.tol_size * size, P.tol_min_bw * med) + P.tol_speed * speed
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
            px, py, sp = pred(fs, xs, ys, i)
            missed = i - fs[-1] - 1
            best, bd = None, None
            for k, c in enumerate(strong):
                if k in claimed:
                    continue
                d = np.hypot(c[0] - px, c[1] - py)
                size = max(c[2], c[3])
                # one point = no velocity yet: allow anything within the physical speed limit
                gate = ((P.vmax_sizes * (missed + 1) + P.jump_margin) * size if len(fs) == 1
                        else tol(size, sp, missed) * 1.5)
                if d <= gate and (bd is None or d < bd):
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
                px, py, sp = pred(near, xs, ys, f)
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

    # c2. weak-only chains: where no strong arc exists, a run of weak detections
    # can still be the ball if it is long, moves, and fits an arc tightly
    # (a far / net-covered ball). Static junk (signs, vents) fails "moves";
    # random flicker fails "long" and "fits".
    chains: list[dict] = []
    alive = []
    for i in usable:
        if i in owner:
            alive = []                     # don't chain across frames the main track owns
            continue
        cands = list(C[i])
        claimed = set()
        for ch in sorted(alive, key=lambda t: -len(t["pts"])):
            fs = sorted(ch["pts"])[-P.fit_pts:]
            px, py, sp = pred(fs, [ch["pts"][f][0] for f in fs], [ch["pts"][f][1] for f in fs], i)
            missed = i - fs[-1] - 1
            best, bd = None, None
            for k, c in enumerate(cands):
                if k in claimed:
                    continue
                d = np.hypot(c[0] - px, c[1] - py)
                size = max(c[2], c[3])
                gate = ((P.vmax_sizes * (missed + 1) + P.jump_margin) * size if len(fs) == 1
                        else tol(size, sp, missed))
                if d <= gate and (bd is None or d < bd):
                    best, bd = k, d
            if best is not None:
                c = cands[best]
                ch["pts"][i] = (c[0], c[1], max(c[2], c[3]), c[4])
                claimed.add(best)
        for k, c in enumerate(cands):
            if k not in claimed:
                ch = {"pts": {i: (c[0], c[1], max(c[2], c[3]), c[4])}}
                chains.append(ch)
                alive.append(ch)
        alive = [t for t in alive if i - max(t["pts"]) <= max_miss]

    added_chain = False
    tentative = []                         # short smooth moving chains: worth a closer look (zoom search)
    for ch in chains:
        pts = ch["pts"]
        if 3 <= len(pts) < chain_min or (len(pts) >= chain_min and
                                                  sum(p[3] for p in pts.values()) < P.weak_chain_conf):
            fs_ = sorted(pts)
            xs_ = np.array([pts[f][0] for f in fs_]); ys_ = np.array([pts[f][1] for f in fs_])
            sz_ = float(np.median([pts[f][2] for f in pts]))
            if np.hypot(xs_[-1] - xs_[0], ys_[-1] - ys_[0]) >= 1.5 * sz_:
                ff = np.array(fs_, float) - fs_[0]
                rr = np.hypot(xs_ - np.polyval(np.polyfit(ff, xs_, 1), ff), ys_ - np.polyval(np.polyfit(ff, ys_, 1), ff))
                if np.sqrt(np.mean(rr ** 2)) <= max(0.35 * sz_, 0.2 * med):
                    tentative.append([(f, pts[f][0], pts[f][1]) for f in fs_])
        if len(pts) < chain_min:
            continue
        fs = np.array(sorted(pts), float)
        xs = np.array([pts[f][0] for f in sorted(pts)])
        ys = np.array([pts[f][1] for f in sorted(pts)])
        sz = float(np.median([pts[f][2] for f in pts]))
        if np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]) < 2 * sz:
            continue                       # not moving: a sign, a vent, a head
        if sum(p[3] for p in pts.values()) < P.weak_chain_conf:
            continue
        deg = 2 if len(fs) >= 5 else 1
        rx = xs - np.polyval(np.polyfit(fs - fs[0], xs, deg), fs - fs[0])
        ry = ys - np.polyval(np.polyfit(fs - fs[0], ys, deg), fs - fs[0])
        if np.sqrt(np.mean(rx ** 2 + ry ** 2)) > max(0.35 * sz, 0.2 * med):
            continue                       # jittery: not one object on a ballistic path
        for f, p in pts.items():
            if f not in owner:
                owner[f] = (*p, DETECTED if p[3] >= P.strong_conf else WEAK)
                added_chain = True
    if added_chain:
        for _ in range(4):
            a = extend(+1)
            b = extend(-1)
            if not (a or b):
                break

    # c4. stationary / slowly drifting stretches: any part of a run that stays
    # within one ball-width for static_s seconds (a knee, a head, a ball held or
    # resting) is not the ball in play. Checked with a sliding window, so a long
    # run that drifts slowly (a person's head) is caught too.
    keys = sorted(owner)
    win = max(4, int(round(P.static_s * fps)))
    from core.cost_linker import _gravity           # clip gravity, px/frame², from confident arcs
    by_frame = {i: [(c[0], c[1], max(c[2], c[3]), None, c[4]) for c in C[i]] for i in range(n) if not dup[i]}
    g_px = _gravity(by_frame, [not d for d in dup], med)
    if keys:
        runs, cur = [], [keys[0]]
        for f in keys[1:]:
            if f - cur[-1] <= max_miss:
                cur.append(f)
            else:
                runs.append(cur)
                cur = [f]
        runs.append(cur)
        drop = set()
        for r in runs:
            r = [f for f in r if not dup[f]]
            if len(r) < 4:
                continue
            j = 0
            for i in range(len(r)):
                while r[i] - r[j] > win:
                    j += 1
                seg = r[j:i + 1]
                if len(seg) < 4 or (r[i] - r[j] < win and not (j == 0 and i == len(r) - 1)):
                    continue                       # need a full window (or the whole short run)
                xs = [owner[f][0] for f in seg]
                ys = [owner[f][1] for f in seg]
                sz = float(np.median([owner[f][2] for f in seg]))
                if max(xs) - min(xs) < sz and max(ys) - min(ys) < sz:
                    drop.update(seg)
                elif g_px > 0 and r[i] - r[j] >= win:
                    # A ball in the air can't stay level: over a window of L frames
                    # gravity changes its height by at least g·L²/8 (window centred
                    # on the apex). Much flatter than that = carried / rolling / a head.
                    L = r[i] - r[j]
                    edge = any(owner[f][1] - owner[f][2] / 2 <= cy0 + 2 or owner[f][1] + owner[f][2] / 2 >= cy1 - 2
                               for f in seg)          # clipped at top/bottom: height is biased, skip
                    slow = (max(xs) - min(xs)) / L < 0.15 * sz
                    if not edge and slow and max(ys) - min(ys) < 0.5 * g_px * L * L / 8:
                        drop.update(seg)
        for f in drop:
            del owner[f]

    # c3. one ball: consecutive points must be physically reachable ────────────
    # Each stage above is gated locally, but separate stretches (another court's
    # ball, a second ball in play) can still end up side by side in time, so the
    # track would jump between them. Split wherever a step is faster than any
    # real hit, then keep stretches strongest-first, accepting one only if it can
    # be reached from the already-kept stretches before and after it.
    def reachable(fa, pa, fb, pb, size):
        return reachable_pts(P, fa, pa, fb, pb, size, TAU)

    def pt(f):
        return owner[f]

    def ok_pair(fa, fb):
        return reachable(fa, pt(fa), fb, pt(fb), max(pt(fa)[2], pt(fb)[2]))

    for _ in range(3):
        # 1) isolated spikes: a point neither neighbour can reach, while the
        #    neighbours reach each other, is one bad point - drop only it.
        changed = True
        while changed:
            changed = False
            keys = sorted(owner)
            for a, b, c in zip(keys, keys[1:], keys[2:]):
                if not ok_pair(a, b) and not ok_pair(b, c) and ok_pair(a, c):
                    del owner[b]
                    changed = True
                    break
        keys = sorted(owner)
        if not keys:
            break
        # 2) split at impossible steps; keep stretches strongest-first, trimming a
        #    few edge points if that lets a stretch connect to what's already kept.
        segs = [[keys[0]]]
        for f in keys[1:]:
            if ok_pair(segs[-1][-1], f):
                segs[-1].append(f)
            else:
                segs.append([f])
        if len(segs) == 1:
            break
        score = lambda s: sum(owner[f][3] for f in s)
        kept: list[list[int]] = []
        TRIM = 5
        for s in sorted(segs, key=lambda s: -score(s)):
            before = [k for k in kept if k[-1] < s[0]]
            after = [k for k in kept if k[0] > s[-1]]
            lo, hi = 0, len(s)
            if before:
                b = max(before, key=lambda k: k[-1])[-1]
                lo = next((j for j in range(min(TRIM, len(s))) if ok_pair(b, s[j])), None)
            if after and lo is not None:
                a = min(after, key=lambda k: k[0])[0]
                hi = next((len(s) - j for j in range(min(TRIM, len(s))) if ok_pair(s[len(s) - 1 - j], a)), None)
            if lo is None or hi is None or hi - lo < 1:
                continue
            kept.append(s[lo:hi])
        keep = {f for s in kept for f in s}
        dropped = [f for f in keys if f not in keep]
        if not dropped:
            break
        for f in dropped:
            del owner[f]
        for _ in range(4):                 # freed frames: let the kept track grow back into them
            a = extend(+1)
            b = extend(-1)
            if not (a or b):
                break

    tentative = [t for t in tentative if not any(f in owner for f, _, _ in t)]
    return finalize(owner, n, fps, width, height, P, dup, (cx0, cy0, cx1, cy1), tentative, TAU)


def reachable_pts(P: LinkParams, fa, pa, fb, pb, size, tau=None) -> bool:
    """Can one ball get from point a (frame fa) to point b (frame fb)? pa/pb: (x, y, size, ...).

    Two checks: distance (speed limit) and apparent size. A ball's box grows or
    shrinks gradually as it moves toward/away from the camera; a sudden jump in
    size over a short time means a different ball (e.g. another court's)."""
    gap = abs(tau[fb] - tau[fa]) if tau is not None else abs(fb - fa)
    if np.hypot(pb[0] - pa[0], pb[1] - pa[1]) > (P.vmax_sizes * gap + P.jump_margin) * size:
        return False
    if len(pa) > 2 and len(pb) > 2 and pa[2] > 0 and pb[2] > 0:
        if abs(np.log(pb[2] / pa[2])) > P.size_jump + P.size_jump_rate * gap:
            return False
    return True


def finalize(owner: dict, n: int, fps: float, width: int, height: int, P: LinkParams,
             dup: list, content: tuple, tentative: list | None = None, TAU=None) -> Track:
    """Shared last step for any linker: turn the chosen points into per-frame states,
    copy repeated frames, and bridge short on-screen gaps (smooth arc, or a corner
    where a touch happened inside the gap). owner: frame -> (x, y, size, conf, state)."""
    cx0, cy0, cx1, cy1 = content
    if not P.vmax_sizes:                   # called directly (cost linker): derive per-frame values
        from dataclasses import replace as _replace
        P = _replace(P, vmax_sizes=P.vmax_bw_s / fps, size_jump_rate=P.size_jump_rate_s / fps)
    if TAU is None:
        TAU = np.arange(n, dtype=float)
    min_span = max(3.0, P.min_fit_span_s * fps)
    max_miss = max(2, int(round(P.max_miss_s * fps)))

    def reachable(fa, pa, fb, pb, size):
        return reachable_pts(P, fa, pa, fb, pb, size, TAU)

    # d. bridge short on-screen gaps ──────────────────────────────────────────
    state = [NONE] * n
    pos: list = [None] * n
    conf: list = [None] * n
    size: list = [None] * n
    for f, (x, y, s, c, st) in owner.items():
        state[f], pos[f], conf[f], size[f] = st, (x, y), c, s

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
        if not reachable(a, owner[a], b, owner[b], max(owner[a][2], owner[b][2])):
            continue                       # the two ends can't be the same ball: don't connect them
        # Ball touching the frame edge at either end of the gap may have left the
        # frame (and its box is clipped, so the position is biased): don't bridge.
        def at_edge(g):
            x, y, s = owner[g][0], owner[g][1], owner[g][2]
            return (x - s / 2 <= cx0 + 2 or y - s / 2 <= cy0 + 2 or
                    x + s / 2 >= cx1 - 2 or y + s / 2 >= cy1 - 2)
        if gap > 3 and (at_edge(a) or at_edge(b)):
            continue
        vxa, vya = _velocity(left, [owner[g][0] for g in left], [owner[g][1] for g in left], a, TAU, min_span)
        vxb, vyb = _velocity(right, [owner[g][0] for g in right], [owner[g][1] for g in right], b, TAU, min_span)
        xa, ya = owner[a][0], owner[a][1]
        xb, yb = owner[b][0], owner[b][1]
        T = float(TAU[b] - TAU[a])
        lx, ly = [owner[g][0] for g in left], [owner[g][1] for g in left]
        rx, ry = [owner[g][0] for g in right], [owner[g][1] for g in right]
        L = {f: _predict(left, lx, ly, f, TAU, min_span)[:2] for f in range(a + 1, b + 1)}
        R = {f: _predict(right, rx, ry, f, TAU, min_span)[:2] for f in range(a, b)}
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
                u = (TAU[f] - TAU[a]) / T
                h00, h10 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u
                h01, h11 = -2 * u**3 + 3 * u**2, u**3 - u**2
                x = h00 * xa + h10 * T * vxa + h01 * xb + h11 * T * vxb
                y = h00 * ya + h10 * T * vya + h01 * yb + h11 * T * vyb
                path.append((f, x, y))
        m = P.edge_margin
        if any(not (cx0 + m <= x <= cx1 - m and cy0 + m <= y <= cy1 - m) for _, x, y in path):
            continue                       # ball would leave the frame: don't invent it
        for f, x, y in path:
            if state[f] == NONE:
                state[f], pos[f] = FILLED, (x, y)
    # a repeated frame right after a filled one shows the same image too
    for i in range(1, n):
        if dup[i] and state[i - 1] == FILLED and state[i] in (NONE, FILLED):
            state[i], pos[i] = FILLED, pos[i - 1]

    return Track(fps=fps, width=width, height=height, state=state, pos=pos, conf=conf, dup=dup,
                 content=(cx0, cy0, cx1, cy1), tentative=tentative or [], size=size,
                 tau=[float(x) for x in TAU])
