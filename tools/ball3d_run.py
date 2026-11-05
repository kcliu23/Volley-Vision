"""The ball in 3D for one clip: 2D track (or your labels) + calibrated camera -> flights in meters.

    python -m tools.ball3d_run videos/test74.mov             # from the tracker's output
    python -m tools.ball3d_run videos/test74.mov --labels    # from your checked labels (data/verified)

Needs data/cameras/<clip>.json (tools.calibrate_court) and output/<clip>/{track,contacts}.json
(the pipeline / autolabel). Flights come from the 2D flight split; consecutive flights with a
contact between them are linked (same point at the touch; a "bounce" also sits on the floor),
and each linked chain is fitted together (core.ball3d).

Writes output/<clip>/ball3d.json and ball3d.png (top view, side view, 3D), and prints per flight:
apex height, speed, the 2D fit error, and physical checks (no flight below the floor; where a
flight crosses the net plane, its height there). Flights whose 2D error is large are not free
flights (ball held, rolling, a wrong split) and are flagged, not trusted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402
from core.ball3d import BALL_D, Link, Segment, fit_rally  # noqa: E402
from core.camera import Camera  # noqa: E402

LINK_GAP_S = 0.3          # flights further apart than this aren't linked
FREE_X = 2.0              # a free flight fits its 2D points within this many times their own jitter
FREE_MIN_PX = 8.0         # ... and never asks for better than this (median px): model error (calibration,
                          # drag) is a few px; what this catches (held, rolling, wrong splits) misses by 25-50
REACH_M = 3.5             # a touch is within a jumping player's reach


def sizes_for(stem: str, n: int) -> np.ndarray:
    """Ball diameter in pixels per frame: your box where you fixed a frame, else autolabel's
    proposal box (YOLO box at the ball, or the nearby median)."""
    out = np.full(n, np.nan)
    pp = paths.proposals(stem)
    if pp.exists():
        P = json.load(open(pp))
        for f, fr in enumerate(P["frames"][:n]):
            if fr.get("box"):
                out[f] = float(np.mean(fr["box"]))
    vp = paths.verified(stem)
    if vp.exists():
        for k, b in json.load(open(vp)).get("boxes", {}).items():
            if int(k) < n and b:
                out[int(k)] = float(np.mean(b))
    return out


def calibrate_size(flights, uv, w, size, fps, cam, min_n: int = 25, max_px: float = 4.0):
    """(corrected sizes, bias): bias = median over long, clean flights of (depth from gravity
    alone) / (depth from box size alone) = how much bigger the boxes are than the ball."""
    ratios = []
    for fl in flights:
        fr = np.arange(fl["f0"], fl["f1"] + 1)
        fr = fr[np.isfinite(uv[fr, 0])]
        if len(fr) < min_n or np.isfinite(size[fr]).sum() < min_n // 2:
            continue
        fit = fit_rally([Segment(fr / fps, uv[fr], np.full(len(fr), np.nan), w[fr])], [], cam)
        if not fit.rms_px <= max_px:
            continue
        dg = float(np.median(cam.depth(fit.at(0, fr / fps))))
        ds = float(np.nanmedian(cam.f * BALL_D / size[fr]))
        if 2 < dg < 60:
            ratios.append(dg / ds)
    bias = float(np.median(ratios)) if len(ratios) >= 3 else 1.0
    return size / bias, bias


def jitter_px(uv) -> float:
    """2D noise of a run of points from their frame-to-frame second differences (a smooth flight
    barely bends in one frame, so what's left is noise): robust sigma / sqrt(6)."""
    d2 = uv[2:] - 2 * uv[1:-1] + uv[:-2]
    if len(d2) < 4:
        return FREE_MIN_PX / FREE_X
    r = np.hypot(d2[:, 0], d2[:, 1])
    return float(np.median(r) / 0.6745 / np.sqrt(6) * 0.85)   # 0.85: 2D radius -> per-axis scale


def free_px(seg, edge) -> float:
    ok = ~edge[seg.frames]
    pts = seg.uv[ok] if ok.sum() >= 6 else seg.uv
    return max(FREE_MIN_PX, FREE_X * jitter_px(pts))


def edge_mask(stem: str, uv, cam, output) -> np.ndarray:
    """Frames whose ball is within one ball-width of the picture's edge (the video area: phone
    clips in a screen recording have dark side bars)."""
    x0, y0, x1, y1 = 0, 0, cam.width, cam.height
    sp = paths.cache_dir(stem, output) / "sample_dets.json"
    ball = 30.0
    if sp.exists():
        prof = json.load(open(sp)).get("profile") or {}
        x0, y0, x1, y1 = prof.get("content") or (x0, y0, x1, y1)
        ball = float(prof.get("ball_px") or ball)
    x, y = uv[:, 0], uv[:, 1]
    with np.errstate(invalid="ignore"):
        m = np.isfinite(x) & ((x < x0 + ball) | (x > x1 - ball) | (y < y0 + ball) | (y > y1 - ball))
    return m, (x0, y0, x1, y1)


def out_of_picture(fr, seen, edge) -> np.ndarray:
    """Frames of a flight with no ball where it left at an edge and came back at an edge: it was
    out of the picture there (not hidden behind someone)."""
    out = []
    i, n = 0, len(fr)
    while i < n:
        if seen[i]:
            i += 1; continue
        j = i
        while j < n and not seen[j]:
            j += 1
        if i > 0 and j < n and edge[fr[i - 1]] and edge[fr[j]]:
            out.extend(fr[i:j].tolist())
        i = j
    return np.array(out, float)


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
    for k, (a0, a1) in inner.items():
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


def player_reach(players, fps):
    """near(t, xyz): metres between the ball and the closest player's reach (a vertical cylinder
    over their court spot, radius 1 m, from the floor to feet height + 1.33 x standing height,
    i.e. standing reach plus any jump), or None where no player is known at that time."""
    by_f = {}
    for p in players:
        h = p.height_m if np.isfinite(p.height_m) else 1.85
        for k, f in enumerate(p.frames):
            if np.isfinite(p.ground[k, 0]):
                by_f.setdefault(int(f), []).append((p.ground[k, 0], p.ground[k, 1], p.feet_z[k] + 1.33 * h))
    by_f = {f: np.array(v) for f, v in by_f.items()}

    def near(t, x):
        f = int(round(t * fps))
        P = next((by_f[g] for g in (f, f - 1, f + 1, f - 2, f + 2, f - 3, f + 3) if g in by_f), None)
        if P is None:
            return None
        dxy = np.maximum(0.0, np.hypot(P[:, 0] - x[0], P[:, 1] - x[1]) - 1.0)
        dz = np.maximum(0.0, x[2] - P[:, 2])
        return float(np.min(np.hypot(dxy, dz)))
    return near


def solve_v2(uv, w, size, edge, box, fps, cam, contacts, log=print, rally_gap_s: float = 1.5, players=None):
    """The global rally fit: per rally (observed stretches separated by > rally_gap_s without the
    ball), core.rally3d.solve with the tracker's contacts as hints. Same output as solve_v1."""
    from core.rally3d import Obs, solve
    fr_all = np.where(np.isfinite(uv[:, 0]))[0]
    if not len(fr_all):
        return []
    cut = np.where(np.diff(fr_all) > rally_gap_s * fps)[0]
    rallies = np.split(fr_all, cut + 1)
    pieces = []
    for ri, fr in enumerate(rallies):
        if len(fr) < 8:
            continue
        full = np.arange(fr[0], fr[-1] + 1)
        out_f = out_of_picture(full, np.isfinite(uv[full, 0]), edge).astype(int)
        o = Obs(fr / fps, uv[fr], size[fr], w[fr], fr, edge[fr], out_f, box, fps)
        if players:
            o.near = player_reach(players, fps)
        hints = [c["frame"] for c in contacts if fr[0] <= c["frame"] <= fr[-1]]
        res, info = solve(o, cam, hints, log=log)
        nt = len(res) - 1
        log(f"rally {fr[0]}-{fr[-1]}: {nt} touch{'es' if nt != 1 else ''} chosen by the fit"
            + (f"; flights still fitting badly: {[res[k][1] for k in info['bad']]}" if info["bad"] else ""))
        for seg, span, fit, i in res:
            pieces.append((ri, seg, span, fit, i))
    return pieces


def positions(stem: str, track: dict, labels: bool):
    """(uv (n, 2) with NaN where no ball, weight (n,))."""
    n = len(track["frames"])
    uv, w = np.full((n, 2), np.nan), np.zeros(n)
    if labels:
        V = json.load(open(paths.verified(stem)))
        for k, v in V["labels"].items():
            if isinstance(v, list) and int(k) < n:
                uv[int(k)], w[int(k)] = v, 1.0
    else:
        for r in track["frames"]:
            if r.get("pos") is not None and r["state"] in ("detected", "weak", "tracknet"):
                uv[r["f"]] = r["pos"][:2]
                w[r["f"]] = 1.0 if r["state"] != "tracknet" else 0.5
    return uv, w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--labels", action="store_true", help="use your checked labels instead of the tracker's output")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--v2", action="store_true", help="the global rally fit (core.rally3d): touches chosen by the fit")
    a = ap.parse_args()
    stem = Path(a.video).stem
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    net_h = json.load(open(Path("data/cameras") / f"{stem}.json")).get("net_height", 2.43)
    track = json.load(open(paths.track(stem, a.output)))
    C = json.load(open(paths.contacts(stem, a.output)))
    fps = float(track.get("fps") or C.get("fps") or 60.0)
    n = len(track["frames"])
    uv, w = positions(stem, track, a.labels)
    size = sizes_for(stem, n)

    # ball-size bias of this clip: YOLO boxes on a moving, blurred ball run larger than the ball
    # (so size alone puts it too close). Gravity alone fixes depth well on long, clean flights:
    # the median ratio there says how much bigger the boxes are than the ball.
    size, bias = calibrate_size(C["flights"], uv, w, size, fps, cam)
    print(f"[ball3d] {stem}: ball boxes run {bias:.2f}x the true ball size here (from gravity on long flights); corrected")

    # a ball cut by the picture's edge is clicked off-centre (only part of it shows): such points
    # count less in the fit and not at all in the clean-flight check
    edge, box = edge_mask(stem, uv, cam, a.output)
    w = np.where(edge, 0.2 * w, w)

    if a.v2:
        players = None
        pj = Path(a.output) / stem / "players.json"
        if pj.exists():
            from core.players import build
            players = build(json.load(open(pj)), cam, fps)
            print(f"[ball3d] {stem}: {len(players)} players: a touch away from all of them costs more")
        pieces = solve_v2(uv, w, size, edge, box, fps, cam, C["contacts"], players=players,
                          log=lambda m: print(f"[ball3d] {stem}: " + m))
    else:
        pieces = solve_v1(uv, w, size, edge, box, fps, cam, C["flights"], C["contacts"],
                          log=lambda m: print(f"[ball3d] {stem}: " + m))

    flights_out, frames_out = [], {}
    nchains = 1 + max((p[0] for p in pieces), default=-1)
    if True:
        for ci, s, (f0, f1), fit, i in pieces:
            tt = np.arange(f0, f1 + 1) / fps
            X = fit.at(i, tt)
            sd = fit.std(i, tt)
            med = fit.seg_rms_px[i]
            ta, za = fit.apex(i, tt[0], tt[-1])
            v_start = (fit.at(i, tt[0] + 1e-3)[0] - fit.at(i, tt[0])[0]) / 1e-3
            uvp, _ = cam.project(X), None
            e = np.hypot(*(cam.project(fit.at(i, s.t)) - s.uv).T)
            inner_pts = ~edge[s.frames]
            e = e[inner_pts] if inner_pts.sum() >= 4 else e
            lim = free_px(s, edge)
            free = float(np.median(e)) <= lim
            cross = None
            xs = X[:, 0]
            sgn = np.sign(xs)
            j = np.where(sgn[:-1] * sgn[1:] < 0)[0]
            if len(j):
                j = j[0]
                wgt = xs[j] / (xs[j] - xs[j + 1])
                zc = X[j, 2] + wgt * (X[j + 1, 2] - X[j, 2])
                yc = X[j, 1] + wgt * (X[j + 1, 1] - X[j, 1])
                cross = {"frame": int(f0 + j), "height_m": round(float(zc), 2), "y_m": round(float(yc), 2),
                         "clears_net_by_m": round(float(zc - BALL_D / 2 - net_h), 2)}
            flights_out.append({"f0": f0, "f1": f1, "chain": ci, "free_flight": bool(free),
                                "n_points": int(len(s.t)), "limit_px": round(lim, 1),
                                "median_px": round(float(np.median(e)), 1), "apex_m": round(float(za), 2),
                                "speed_start_mps": round(float(np.linalg.norm(v_start)), 1),
                                "min_height_m": round(float(X[:, 2].min()), 2), "net_crossing": cross,
                                "depth_std_m": round(float(np.median(np.linalg.norm(sd, axis=1))), 2),
                                "drag_k": round(float(fit.drag), 4)})
            for g, x, d in zip(range(f0, f1 + 1), X, sd):
                frames_out[g] = {"xyz": [round(float(v), 3) for v in x], "std": [round(float(v), 3) for v in d],
                                 "flight": len(flights_out) - 1, "free": bool(free)}

    out = Path(a.output) / stem / "ball3d.json"
    json.dump({"schema": "vv-ball3d/1", "video": a.video, "fps": fps, "source": "labels" if a.labels else "track",
               "camera": str(Path("data/cameras") / f"{stem}.json"), "flights": flights_out,
               "frames": {str(k): v for k, v in sorted(frames_out.items())}}, open(out, "w"), indent=0)
    print(f"[ball3d] {stem}: {len(flights_out)} flights in {nchains} chains  →  {out}")
    print(f"  {'frames':>11s} {'free':>5s} {'2D px':>6s} {'apex m':>7s} {'speed m/s':>9s} {'lowest m':>8s} {'+-m':>5s}  net crossing")
    for fl in flights_out:
        nc = fl["net_crossing"]
        ncs = f"at {nc['height_m']:.2f} m (clears by {nc['clears_net_by_m']:+.2f})" if nc else ""
        print(f"  {fl['f0']:5d}-{fl['f1']:<5d} {'yes' if fl['free_flight'] else 'NO':>5s} {fl['median_px']:6.1f} "
              f"{fl['apex_m']:7.2f} {fl['speed_start_mps']:9.1f} {fl['min_height_m']:8.2f} {fl['depth_std_m']:5.2f}  {ncs}")
    plot(out.with_suffix(".png"), flights_out, frames_out, cam, net_h, stem)


def plot(path, flights, frames, cam, net_h, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.court import lines
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(122, projection="3d")
    cmap = plt.get_cmap("tab10")
    for p, q in lines(net_h):
        p, q = np.array(p), np.array(q)
        if p[2] == 0 and q[2] == 0:
            ax1.plot([p[0], q[0]], [p[1], q[1]], color="#2a7", lw=1)
        ax3.plot(*zip(p, q), color="#2a7" if p[2] == 0 and q[2] == 0 else "#444", lw=1)
    ax2.plot([-9, 9], [0, 0], color="#2a7", lw=1)
    ax2.plot([0, 0], [net_h - 1, net_h], color="#444", lw=3)
    C = cam.position
    ax1.scatter(C[0], C[1], color="crimson"); ax1.annotate("camera", (C[0], C[1]), color="crimson")
    ax3.scatter(*C, color="crimson")
    for i, fl in enumerate(flights):
        X = np.array([frames[g]["xyz"] for g in range(fl["f0"], fl["f1"] + 1)])
        col = cmap(i % 10)
        ls = "-" if fl["free_flight"] else ":"
        ax1.plot(X[:, 0], X[:, 1], ls, color=col, lw=1.5)
        ax2.plot(X[:, 0], X[:, 2], ls, color=col, lw=1.5)
        ax3.plot(X[:, 0], X[:, 1], X[:, 2], ls, color=col, lw=1.5)
    ax1.set_aspect("equal"); ax1.set_title("top view (m)"); ax1.set_xlim(-15, 12)
    ax2.set_aspect("equal"); ax2.set_title("side view: along the court vs height (m)"); ax2.set_xlim(-15, 12)
    ax2.set_ylim(-0.5, 8)
    ax3.set_xlim(-14, 11); ax3.set_ylim(-8, 8); ax3.set_zlim(0, 7); ax3.set_box_aspect((25, 16, 7))
    ax3.view_init(elev=18, azim=-65)
    fig.suptitle(f"{stem}: ball in 3D (dotted = not a free flight: held / rolling / bad fit)")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    print(f"  picture  →  {path}")


if __name__ == "__main__":
    main()
