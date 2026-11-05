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
FREE_PX = 6.0             # a free flight fits its 2D points within this (median px)


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

    # segments (2D flights) and the links between them
    segs, spans = [], []
    for fl in C["flights"]:
        fr = np.arange(fl["f0"], fl["f1"] + 1)
        ok = np.isfinite(uv[fr, 0])
        if ok.sum() < 6:
            continue
        fr = fr[ok]
        segs.append(Segment(fr / fps, uv[fr], size[fr], w[fr], frames=fr))
        spans.append((int(fl["f0"]), int(fl["f1"])))
    contacts = C["contacts"]
    chains, cur, links_all = [], [0] if segs else [], []
    for k in range(1, len(segs)):
        a0, a1 = spans[k - 1][1], spans[k][0]
        c = next((c for c in contacts if a0 - 2 <= c["frame"] <= a1 + 2), None)
        if c is not None and (a1 - a0) / fps <= LINK_GAP_S:
            cur.append(k)
            links_all.append((k - 1, c["frame"] / fps, "bounce" if c["kind"] == "bounce" else "touch"))
        else:
            chains.append(cur)
            cur = [k]
    if cur:
        chains.append(cur)

    flights_out, frames_out = [], {}
    for ch in chains:
        sub = [segs[k] for k in ch]
        idx = {k: i for i, k in enumerate(ch)}
        links = [Link(idx[k], t, kind) for k, t, kind in links_all if k in idx and k + 1 in idx]
        fit = fit_rally(sub, links, cam)
        for i, k in enumerate(ch):
            s = sub[i]
            f0, f1 = spans[k]
            tt = np.arange(f0, f1 + 1) / fps
            X = fit.at(i, tt)
            sd = fit.std(i, tt)
            med = fit.seg_rms_px[i]
            ta, za = fit.apex(i)
            if not (tt[0] <= ta <= tt[-1]):
                za = float(X[:, 2].max())
            p0, v0 = fit.params[i]
            v_start = v0 + (tt[0] - fit.tr[i]) * np.array([0, 0, -9.81])
            uvp, _ = cam.project(X), None
            e = np.hypot(*(cam.project(fit.at(i, s.t)) - s.uv).T)
            free = float(np.median(e)) <= FREE_PX
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
            flights_out.append({"f0": f0, "f1": f1, "chain": chains.index(ch), "free_flight": bool(free),
                                "median_px": round(float(np.median(e)), 1), "apex_m": round(float(za), 2),
                                "speed_start_mps": round(float(np.linalg.norm(v_start)), 1),
                                "min_height_m": round(float(X[:, 2].min()), 2), "net_crossing": cross,
                                "depth_std_m": round(float(np.median(np.linalg.norm(sd, axis=1))), 2)})
            for g, x, d in zip(range(f0, f1 + 1), X, sd):
                frames_out[g] = {"xyz": [round(float(v), 3) for v in x], "std": [round(float(v), 3) for v in d],
                                 "flight": len(flights_out) - 1, "free": bool(free)}

    out = Path(a.output) / stem / "ball3d.json"
    json.dump({"schema": "vv-ball3d/1", "video": a.video, "fps": fps, "source": "labels" if a.labels else "track",
               "camera": str(Path("data/cameras") / f"{stem}.json"), "flights": flights_out,
               "frames": {str(k): v for k, v in sorted(frames_out.items())}}, open(out, "w"), indent=0)
    print(f"[ball3d] {stem}: {len(flights_out)} flights in {len(chains)} chains  →  {out}")
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
