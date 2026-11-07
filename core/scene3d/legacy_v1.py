"""The FIRST 3D mechanism (v1), kept only to compare against the global rally fit (core.scene3d.rally3d)
until it is checked on real touch labels. Delete this file (and `--v1` in tools.ball3d_run) then.

v1: take the 2D tracker's flight splits, join neighbours one arc fits (remerge), split flights
that fit badly (resplit), link consecutive flights at contacts, fit each chain.
"""
from __future__ import annotations

import numpy as np

from core.scene3d.ball3d import Link, Segment, fit_rally, free_px, out_of_picture

LINK_GAP_S = 0.3          # flights further apart than this aren't linked
REACH_M = 3.5             # a touch is within a jumping player's reach


def _median_err(seg, cam, edge) -> tuple[float, float, float]:
    """(median 2D error away from the edges, of the first half, of the second half) of one arc."""
    fit = fit_rally([seg], [], cam)
    e = np.hypot(*(cam.project(fit.at(0, seg.t)) - seg.uv).T)
    ok = ~edge[seg.frames]
    m = lambda x, o: float(np.median(x[o] if o.sum() >= 3 else x)) if len(x) else 0.0
    h = len(e) // 2
    return m(e, ok), m(e[:h], ok[:h]), m(e[h:], ok[h:])


def remerge(segs, spans, contacts, cam, edge, uv, box, fps, max_gap_s: float = 1.5):
    """Join neighbouring flights with no real touch between them (no contact, or one whose
    direction barely changes) when one arc fits both cleanly. Returns segments, spans and the
    joined pairs."""
    out_s, out_sp, joined = [], [], []
    for seg, sp in zip(segs, spans):
        if out_s:
            A, asp = out_s[-1], out_sp[-1]
            gap = (sp[0] - asp[1]) / fps
            c = [c for c in contacts if asp[1] - 2 <= c["frame"] <= sp[0] + 2 and c["kind"] in ("touch", "bounce")]
            if not c and 0 <= gap <= max_gap_s:
                full = np.arange(asp[0], sp[1] + 1)
                seen = np.isfinite(uv[full, 0])
                fr = np.r_[A.frames, seg.frames]
                out = out_of_picture(full, seen, edge)
                M = Segment(fr / fps, np.r_[A.uv, seg.uv], np.r_[A.size, seg.size], np.r_[A.w, seg.w],
                            frames=fr, out_t=out / fps, box=box)
                lim = min(free_px(A, edge), free_px(seg, edge))
                fit = fit_rally([M], [], cam)
                def piece(P):                              # the joined arc must fit EACH piece on its own
                    e = np.hypot(*(cam.project(fit.at(0, P.t)) - P.uv).T)
                    ok = ~edge[P.frames]
                    return float(np.median(e[ok] if ok.sum() >= 3 else e))
                if piece(A) <= lim and piece(seg) <= lim:
                    out_s[-1], out_sp[-1] = M, (asp[0], sp[1])
                    joined.append((asp, sp))
                    continue
        out_s.append(seg); out_sp.append(sp)
    return out_s, out_sp, joined


def resplit(segs, spans, cam, edge, fps, min_pts: int = 5):
    """Split flights that no single arc fits (median 2D error over free_px(), away from the edges) at
    the point where two arcs joined by a touch fit best, if both then fit cleanly. Returns the new
    segments, spans and {index of the first half: (last frame before, first frame after)}."""
    def clean(seg):
        fit = fit_rally([seg], [], cam)
        e = np.hypot(*(cam.project(fit.at(0, seg.t)) - seg.uv).T)
        ok = ~edge[seg.frames]
        return float(np.median(e[ok] if ok.sum() >= 4 else e))


    def part(seg, m):
        lo, hi = seg.t[m][0], seg.t[m][-1]
        o = seg.out_t[(seg.out_t >= lo) & (seg.out_t <= hi)]
        return Segment(seg.t[m], seg.uv[m], seg.size[m], seg.w[m], frames=seg.frames[m], out_t=o, box=seg.box)

    out_s, out_sp, inner = [], [], {}
    for seg, sp in zip(segs, spans):
        n = len(seg.t)
        lim = free_px(seg, edge)
        single = clean(seg) if n >= 2 * min_pts else 0.0
        if n < 2 * min_pts or single <= lim:
            out_s.append(seg); out_sp.append(sp); continue
        best = None
        for i in range(min_pts, n - min_pts + 1):
            A, B = part(seg, np.arange(n) < i), part(seg, np.arange(n) >= i)
            ia, ib = int(A.frames[-1]), int(B.frames[0])
            if edge[ia] or edge[ib]:                       # leaving / entering the picture isn't a touch
                continue
            if any(ia < t * fps < ib for t in seg.out_t):  # nor is a stretch where it was out of the picture
                continue
            L = [Link(0, (ia + ib) / 2 / fps, "touch", max(1, ib - ia) / 2 / fps)]
            fit = fit_rally([A, B], L, cam)
            touch = fit.at(0, fit.info["touch_t"][0])[0]
            if not (0 <= touch[2] <= REACH_M):             # nobody touches it up there (or under the floor)
                continue
            ea = np.hypot(*(cam.project(fit.at(0, A.t)) - A.uv).T)
            eb = np.hypot(*(cam.project(fit.at(1, B.t)) - B.uv).T)
            score = float(np.median(np.r_[ea, eb]))
            if best is None or score < best[0]:
                best = (score, i, A, B, float(np.median(ea)), float(np.median(eb)), ia, ib)
        if best and best[0] < 0.6 * single and best[4] <= lim and best[5] <= lim:
            _, i, A, B, _, _, ia, ib = best
            inner[len(out_s)] = (ia, ib)
            out_s += [A, B]
            out_sp += [(sp[0], ia), (ib, sp[1])]
        else:
            out_s.append(seg); out_sp.append(sp)
    return out_s, out_sp, inner


def solve_v1(uv, w, size, edge, box, fps, cam, flights, contacts, log=print):
    """The current mechanism: the tracker's 2D flights, merged / re-split by physics, linked at
    contacts into chains, each chain fitted. Returns [(chain index, segment, (f0, f1), fit, i)]."""
    # segments (2D flights) and the links between them
    segs, spans = [], []
    for fl in flights:
        fr = np.arange(fl["f0"], fl["f1"] + 1)
        ok = np.isfinite(uv[fr, 0])
        if ok.sum() < 6:
            continue
        out = out_of_picture(np.arange(fl["f0"], fl["f1"] + 1), ok, edge)
        fr = fr[ok]
        segs.append(Segment(fr / fps, uv[fr], size[fr], w[fr], frames=fr, out_t=out / fps, box=box))
        spans.append((int(fl["f0"]), int(fl["f1"])))

    # the opposite mistake: one flight cut in two by the 2D split where nothing touched it (it
    # went out of the picture and came back, or hid behind a player). If one arc explains both
    # pieces (passing outside the picture where it was out of it), they're one flight.
    segs, spans, merged = remerge(segs, spans, contacts, cam, edge, uv, box, fps)
    for (a0, a1), (b0, b1) in merged:
        log(f"flights {a0}-{a1} and {b0}-{b1} are one flight (no touch between them): joined")

    # a flight no single arc fits is usually two: a touch the 2D split missed (a block, a ball
    # touched while hidden behind a player). Find where one arc becomes two, by physics.
    segs, spans, inner = resplit(segs, spans, cam, edge, fps)
    for _k, (a0, a1) in inner.items():
        log(f"found a touch the 2D split missed between frames {a0} and {a1}")

    chains, cur, links_all = [], [0] if segs else [], []
    for k in range(1, len(segs)):
        a0, a1 = spans[k - 1][1], spans[k][0]
        c = next((c for c in contacts if a0 - 2 <= c["frame"] <= a1 + 2), None)
        if k - 1 in inner:                                 # a touch found by the re-split
            cur.append(k)
            ia, ib = inner[k - 1]
            links_all.append((k - 1, (ia + ib) / 2 / fps, "touch", max(1, ib - ia) / 2 / fps))
        elif c is not None and (a1 - a0) / fps <= LINK_GAP_S:
            cur.append(k)
            links_all.append((k - 1, c["frame"] / fps, "bounce" if c["kind"] == "bounce" else "touch", None))
        else:
            chains.append(cur)
            cur = [k]
    if cur:
        chains.append(cur)

    pieces = []
    for ci, ch in enumerate(chains):
        sub = [segs[k] for k in ch]
        idx = {k: i for i, k in enumerate(ch)}
        links = [Link(idx[k], t, kind, ts) for k, t, kind, ts in links_all if k in idx and k + 1 in idx]
        fit = fit_rally(sub, links, cam)            # drag = the ball's physics value (fitting it per rally is unstable with few flights)
        for i, k in enumerate(ch):
            pieces.append((ci, sub[i], spans[k], fit, i))
    return pieces
