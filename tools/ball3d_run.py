"""The ball in 3D for one clip: 2D track (or your labels) + calibrated camera -> flights in meters.

    python -m tools.ball3d_run videos/test74.mov --labels    # from your checked labels (data/verified)
    python -m tools.ball3d_run videos/test74.mov             # from the tracker's output
    python -m tools.ball3d_run videos/test74.mov --v1        # the first mechanism (for comparison)

(tools.scene3d runs this together with frame timing, players and the viewer.)

Load -> solve -> save. Needs data/cameras/<clip>.json (tools.calibrate_court) and
output/<clip>/{track,contacts}.json; uses output/<clip>/players.json when there (tools.players)
and the clip's true frame times (core.common.frametime). The solving is core.scene3d.rally3d.solve_clip (touches
and arcs chosen together); the physics is core.scene3d.ball3d; every set number is in core.scene3d.params3d.

Writes output/<clip>/ball3d.json and ball3d.png (top view, side view, 3D) and prints per flight:
apex height, speed, 2D fit error, lowest point, and where it crosses the net plane.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths
from core.geometry.camera import Camera
from core.scene3d.ball3d import BALL_D, calibrate_size, edge_mask, free_px, sizes_for


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
    ap.add_argument("--v1", action="store_true", help="the first mechanism (core.scene3d.legacy_v1), for comparison")
    ap.add_argument("--v2", action="store_true", help=argparse.SUPPRESS)   # the default now; kept so old commands work
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

    # true frame times: screen recordings repeat a frame now and then and skip one a few frames
    # later (core.common.frametime); the global fit uses them (v1 keeps evenly spaced frames)
    T = None
    if not a.v1:
        from core.common.frametime import frame_times, summary
        T = frame_times(a.video, paths.clip_dir(stem, a.output))
        if len(T) < n:
            T = np.r_[T, T[-1] + np.arange(1, n - len(T) + 1)]
        print(f"[ball3d] {stem}: frame timing: {summary(T)}")
    # ball-size bias of this clip: YOLO boxes on a moving, blurred ball run larger than the ball;
    # gravity alone fixes depth on long, clean flights, and the ratio says by how much
    size, bias = calibrate_size(C["flights"], uv, w, size, fps, cam, T=T)
    print(f"[ball3d] {stem}: ball boxes run {bias:.2f}x the true ball size here (from gravity on long flights); corrected")

    # a ball cut by the picture's edge is clicked off-centre (only part of it shows): such points
    # count less in the fit and not at all in the clean-flight check
    edge, box = edge_mask(stem, uv, cam, a.output)
    w = np.where(edge, 0.2 * w, w)

    if a.v1:
        from core.scene3d.legacy_v1 import solve_v1
        pieces = solve_v1(uv, w, size, edge, box, fps, cam, C["flights"], C["contacts"],
                          log=lambda m: print(f"[ball3d] {stem}: " + m))
    else:
        from core.scene3d.rally3d import solve_clip
        players = None
        pj = paths.clip_dir(stem, a.output) / "players.json"
        if pj.exists():
            from core.people.players import build
            players = build(json.load(open(pj)), cam, fps)
            print(f"[ball3d] {stem}: {len(players)} players: a touch away from all of them costs more")
        poses = None
        pj2 = paths.clip_dir(stem, a.output) / "pose.json"
        if pj2.exists():
            poses = json.load(open(pj2))["frames"]
            print(f"[ball3d] {stem}: poses found: touches are anchored to the player who made them")
        bpx = float(np.nanmedian(size)) if np.isfinite(size).any() else 25.0
        pieces = solve_clip(uv, w, size, edge, box, fps, cam, C["contacts"], players=players, T=T, net_h=net_h,
                            poses=poses, ball_px=bpx,
                            log=lambda m: print(f"[ball3d] {stem}: " + m))

    flights_out, frames_out = [], {}
    nchains = 1 + max((p[0] for p in pieces), default=-1)
    for ci, s, (f0, f1), fit, i in pieces:
        tt = (T[f0:f1 + 1] if T is not None else np.arange(f0, f1 + 1)) / fps
        X = fit.at(i, tt)
        sd = fit.std(i, tt)
        _, za = fit.apex(i, tt[0], tt[-1])
        v_start = (fit.at(i, tt[0] + 1e-3)[0] - fit.at(i, tt[0])[0]) / 1e-3
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

    from core.geometry.court import lines
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
