"""The ball in 3D for one clip: 2D track (or your labels) + calibrated camera -> flights in meters.

    python -m tools.ball3d_run videos/test74.mov --labels    # from your checked labels (data/verified)
    python -m tools.ball3d_run videos/test74.mov             # from the tracker's output

(main.py runs this together with the 2D track, players and the viewer.)

Load -> solve -> save. Needs data/cameras/<clip>.json (tools.calibrate_court) and
output/<clip>/{track,contacts}.json; players from the pose tracking (tools.pose)
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


def positions(stem: str, track: dict, labels: bool, contacts: dict | None = None, my_ball: bool = False):
    """(uv (n, 2) with NaN where no ball, weight (n,))."""
    n = len(track["frames"])
    uv, w = np.full((n, 2), np.nan), np.zeros(n)
    if labels:
        V = json.load(open(paths.verified(stem)))
        for k, v in V["labels"].items():
            if isinstance(v, list) and int(k) < n:
                uv[int(k)], w[int(k)] = v, 1.0
    else:
        # A filled point is a 2D flight-model prediction, not an image detection. Give it
        # confidence from the source arc's residual and its distance from measured endpoints.
        flights = (contacts or {}).get("flights", [])
        fps = float(track.get("fps") or (contacts or {}).get("fps") or 60.0)

        def filled_weight(frame):
            inside = [f for f in flights if f["f0"] <= frame <= f["f1"]]
            if inside:
                source = min(inside, key=lambda f: min(frame - f["f0"], f["f1"] - frame))
                age = min(frame - source["f0"], source["f1"] - frame)
            else:
                left = [f for f in flights if f["f1"] < frame]
                right = [f for f in flights if f["f0"] > frame]
                choices = []
                if left:
                    f = max(left, key=lambda x: x["f1"])
                    choices.append((frame - f["f1"], f))
                if right:
                    f = min(right, key=lambda x: x["f0"])
                    choices.append((f["f0"] - frame, f))
                if not choices:
                    return 0.1
                age, source = min(choices, key=lambda x: x[0])
            rms = float(source.get("rms", 6.0))
            arc_quality = 1.0 / (1.0 + (rms / 3.0) ** 2)
            gap_quality = float(np.exp(-age / max(0.15 * fps, 1.0)))
            return max(0.05, min(0.6, arc_quality * gap_quality))

        for r in track["frames"]:
            if r.get("pos") is not None and r["state"] in ("detected", "weak", "tracknet", "filled"):
                uv[r["f"]] = r["pos"][:2]
                st = r["state"]
                w[r["f"]] = filled_weight(r["f"]) if st == "filled" else (0.5 if st == "tracknet" else 1.0)
    # --my-ball: your clicked ball positions (tools.review) win over the tracker wherever you
    # placed one. Off by default: on the holdouts it found 1 more touch but invented 7 more
    # (64 vs 57), since clicks add strong points during contacts where the tracker had none
    vp = paths.verified(stem)
    if my_ball and not labels and Path(vp).exists():
        for k, v in (json.load(open(vp)).get("labels") or {}).items():
            if isinstance(v, list) and int(k) < n:
                uv[int(k)], w[int(k)] = v[:2], 1.0
    return uv, w


def _named_player(poses, plabels, cam, fps):
    """named(frame, number) -> {"side", "spot"}: the side from the number (1-6 near, 7-12 far) and,
    when that player has a box (pose.json + your numbers), where they stood: the lowest box bottom
    in the half second up to the touch (a jumper's takeoff spot), on the floor."""
    def named(f: int, n: int) -> dict:
        out = {"side": -1 if n <= 6 else 1}
        if not poses or plabels is None:
            return out
        best = None
        for g in range(f - int(0.5 * fps), f + 3):
            for p in poses.get(str(g), []):
                if plabels.number(p["id"], g) == n and (best is None or p["box"][3] > best[1]):
                    best = ((p["box"][0] + p["box"][2]) / 2, p["box"][3])
        if best is not None:
            g = cam.floor_point([best], 0.0)[0]
            out["spot"] = (float(g[0]), float(g[1]))
        return out
    return named


TOUCH_GAP = 8          # frames between two flights that still make one touch (the ball held on the hands)
FLOOR_Z = 0.3          # m: a join this low is the ball on the floor (a bounce), not a touch


def write_touches(path, flights, frames, uv, T, fps, cam, poses, plabels, stem, serves=None, who=None, who_box=None):
    """output/<clip>/touches.json: every touch in the 3D (where one flight hands over to the next),
    with who touched it (core.people.touchers: the same rule review suggests with) and the side.

      {"touches": [{"frame", "kind": serve | touch, "time_s", "xyz", "side": -1 near / +1 far,
                    "toucher": {"number" | null, "track", "box", "d", "feet"} | null}]}"""
    from core.people.touchers import Players
    P = Players(poses, plabels) if poses else None
    from core.scene3d.params3d import SERVE_AUTO_X_M
    out = []
    events = []                                           # (frame, kind)
    for i, b in enumerate(flights):
        a = flights[i - 1] if i else None
        if a is not None and b["chain"] == a["chain"] and b["f0"] - a["f1"] <= TOUCH_GAP:
            Xa, Xb = frames.get(a["f1"], {}).get("xyz"), frames.get(b["f0"], {}).get("xyz")
            X = [round((u + v) / 2, 2) for u, v in zip(Xa, Xb)] if Xa and Xb else (Xa or Xb)
            if X is not None and X[2] < FLOOR_Z:          # the ball on the floor: a bounce, not a touch
                continue
            events.append(((a["f1"] + b["f0"] + 1) // 2, "touch", X))   # a touch joins two flights (your
            continue                                                    # marked contact frames sit between)
        X0 = frames.get(b["f0"], {}).get("xyz")           # a rally's first flight: from a serve when it
        if (X0 is not None and X0[2] > 1.0 and (abs(X0[0]) >= SERVE_AUTO_X_M     # starts far behind the net,
                                                or any(abs(g - b["f0"]) <= 6 for g in serves or []))):  # or you marked one
            events.append((b["f0"], "serve", X0))
    for t, kind, X in events:
        xy = None
        for g in sorted(range(t - 3, t + 4), key=lambda g: abs(g - t)):   # the ball in the picture, nearest frame
            if 0 <= g < len(uv) and np.isfinite(uv[g]).all():
                xy = [float(uv[g][0]), float(uv[g][1])]
                break
        mine = [g for g in (who or {}) if abs(g - t) <= TOUCH_GAP // 2 + 2]
        toucher = None
        if mine:                                          # you said who touched it (--use-labels): that's it
            g = min(mine, key=lambda g: abs(g - t))
            toucher = {"number": who[g], "track": None, "box": (who_box or {}).get(g), "d": None,
                       "feet": None, "labelled": True}
        elif P is not None and xy is not None:
            r = P.rank(t, xy, cam, X)
            if r:
                p = r[0]
                toucher = {"number": p["number"], "track": p["id"], "box": [round(float(v), 1) for v in p["box"]],
                           "d": p["d"], "feet": p.get("feet")}
        side = None
        if toucher and isinstance(toucher["number"], int):
            side = -1 if toucher["number"] <= 6 else 1
        elif toucher and toucher.get("feet"):
            side = -1 if toucher["feet"][0] < 0 else 1
        elif X is not None:
            side = -1 if X[0] < 0 else 1
        out.append({"frame": int(t), "kind": kind, "time_s": round(float(T[t] / fps if T is not None else t / fps), 3),
                    "xyz": X, "side": side, "toucher": toucher})
    json.dump({"schema": "vv-touches/1", "clip": stem, "touches": out}, open(path, "w"), indent=1)
    named = sum(1 for x in out if x["toucher"] and x["toucher"]["number"] is not None)
    seen = sum(1 for x in out if x["toucher"])
    print(f"[ball3d] {stem}: {len(out)} touches, toucher found for {seen} ({named} with a number)  →  {path}")


def write_rallies(d: Path, sfx: str, stem: str):
    """output/<clip>/rallies.json: each rally (serve -> the ball's first floor contact), who won it,
    and the running score (core.scene3d.rallies)."""
    from core.scene3d.rallies import find_rallies, score
    B = json.load(open(d / f"ball3d{sfx}.json"))
    T = json.load(open(d / f"touches{sfx}.json"))["touches"]
    rows = score(find_rallies(B, T))
    json.dump({"schema": "vv-rallies/1", "clip": stem, "rallies": rows}, open(d / f"rallies{sfx}.json", "w"), indent=1)
    team = {-1: "near", 1: "far", None: "?"}
    for i, r in enumerate(rows):
        print(f"[ball3d] {stem}: rally {i + 1}  frames {r['f0']}-{r['f1']}  {r['end']:4s}  won by {team[r['winner']]:4s}"
              f"  ({r['why']})  score near {r['near']} - far {r['far']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--labels", action="store_true", help="use your checked labels instead of the tracker's output")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--v2", action="store_true", help=argparse.SUPPRESS)   # the default now; kept so old commands work
    ap.add_argument("--picture-timing-only", action="store_true",
                    help="don't move catch-up skips to where the ball's double step is")
    ap.add_argument("--my-ball", action="store_true",
                    help="your clicked ball positions (tools.review) replace the tracker's where you placed one")
    ap.add_argument("--use-my-touches", action="store_true",
                    help="your touch marks (tools.review) are the touches; only the ball in play is fitted")
    ap.add_argument("--use-labels", action="store_true",
                    help="everything you labeled: your ball (--labels), touch marks (--use-my-touches), player "
                         "numbers and who touched it. Writes ball3d_labels.json / touches_labels.json / "
                         "ball3d_labels.png (the automatic results, without --use-labels, stay as they are)")
    a = ap.parse_args()
    if a.use_labels:
        a.labels = a.use_my_touches = True
    sfx = "_labels" if a.use_labels else ""            # automatic by default: none of your labels
    stem = Path(a.video).stem
    from core.common.timing import Laps
    TL = Laps(f"[ball3d] {stem}:")
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    net_h = json.load(open(Path("data/cameras") / f"{stem}.json")).get("net_height", 2.43)
    track = json.load(open(paths.track(stem, a.output)))
    C = json.load(open(paths.contacts(stem, a.output)))
    fps = float(track.get("fps") or C.get("fps") or 60.0)
    n = len(track["frames"])
    uv, w = positions(stem, track, a.labels, C, my_ball=a.my_ball)
    size = sizes_for(stem, n)

    # true frame times: screen recordings repeat a frame now and then and skip one a few frames
    # later (core.common.frametime); the fit uses the true times
    from core.common.frametime import frame_times, summary
    T = frame_times(a.video, paths.clip_dir(stem, a.output))
    if len(T) < n:
        T = np.r_[T, T[-1] + np.arange(1, n - len(T) + 1)]
    if not a.picture_timing_only:
        from core.common.frametime import place_skips_by_ball
        T = place_skips_by_ball(T, uv, w, log=lambda m: print(f"[ball3d] {stem}: {m}"))
    print(f"[ball3d] {stem}: frame timing: {summary(T)}")
    # ball-size bias of this clip: YOLO boxes on a moving, blurred ball run larger than the ball;
    # gravity alone fixes depth on long, clean flights, and the ratio says by how much
    size, bias = calibrate_size(C["flights"], uv, w, size, fps, cam, T=T)
    print(f"[ball3d] {stem}: ball boxes run {bias:.2f}x the true ball size here (from gravity on long flights); corrected")

    # a ball cut by the picture's edge is clicked off-centre (only part of it shows): such points
    # count less in the fit and not at all in the clean-flight check
    edge, box = edge_mask(stem, uv, cam, a.output)
    w = np.where(edge, 0.2 * w, w)

    from core.people.players import load_people
    from core.scene3d.rally3d import solve_clip
    TL.lap("inputs, frame times, ball sizes")
    players = load_people(stem, a.output, cam, fps, log=lambda m: print(f"[ball3d] {stem}: " + m),
                          use_labels=a.use_labels)
    if players:
        print(f"[ball3d] {stem}: {len(players)} player tracks: a touch away from all of them costs more")
    poses, plabels = None, None
    pj2 = paths.clip_dir(stem, a.output) / "pose.json"
    if pj2.exists():
        poses = json.load(open(pj2))["frames"]
        print(f"[ball3d] {stem}: poses found: touches are anchored to the player who made them")
        from core.people.player_labels import PlayerLabels
        if a.use_labels:
            plabels = PlayerLabels.load(stem, pj2, log=lambda m: print(f"[ball3d] {stem}: " + m))
        if plabels is not None:
            print(f"[ball3d] {stem}: your player numbers: each toucher's side of the net is known")
    bpx = float(np.nanmedian(size)) if np.isfinite(size).any() else 25.0
    fixed, keep = None, None
    if a.use_my_touches:
        from evaluation.inplay import in_play
        from evaluation.touches3d import your_touches
        V = json.load(open(paths.verified(stem)))
        keep = in_play(V)
        runs = your_touches(V)
        fixed = [(t0 + t1) / 2 for t0, t1, _ in runs]
        # the ball on someone's hands / arms is not in free flight: your marked contact
        # frames stay out of both flights (test20 942-943: the ball already resting on the
        # digger's arms bent the incoming flight 921-943 to 10 px)
        uv = uv.copy()
        for t0, t1, _ in runs:
            uv[t0:t1 + 1] = np.nan
        print(f"[ball3d] {stem}: your {len(fixed)} touch marks are the touches (ball in play only)")
    # your "who touched it" (tools.review): the 3D puts those touches at that player, on their side
    who, named = None, None
    vp = Path(paths.verified(stem))
    serves, who_box = None, None
    if vp.exists() and a.use_labels:
        Vj = json.load(open(vp))
        who = {int(k): int(v) for k, v in (Vj.get("touched_by") or {}).items()} or None
        serves = [int(k) for k, v in (Vj.get("contacts") or {}).items() if v == "serve"] or None
        who_box = {int(k): v for k, v in (Vj.get("touched_box") or {}).items()} or None
    if who:
        named = _named_player(poses, plabels, cam, fps)
        print(f"[ball3d] {stem}: your 'who touched it' on {len(who)} touches: each one is put at that player")
    TL.lap("players and poses")
    pieces = solve_clip(uv, w, size, edge, box, fps, cam, C["contacts"], players=players, T=T, net_h=net_h,
                        poses=poses, ball_px=bpx, player_labels=plabels, fixed_touches=fixed, keep=keep, who=who, named=named, serves=serves,
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
                            "drag_k": round(float(fit.drag), 4),
                            # spin lift at the start, m/s^2: + backspin (floats), - topspin (dives), 0 = none fitted
                            "spin_lift_mps2": round(fit.sp(i) * float(np.linalg.norm(v_start)), 2)})
        for g, x, d in zip(range(f0, f1 + 1), X, sd):
            frames_out[g] = {"xyz": [round(float(v), 2) for v in x], "std": [round(float(v), 2) for v in d],
                             "flight": len(flights_out) - 1, "free": bool(free)}

    out = Path(a.output) / stem / f"ball3d{sfx}.json"
    TL.lap("3D fit (all rallies)")
    json.dump({"schema": "vv-ball3d/1", "video": a.video, "fps": fps, "source": "labels" if a.labels else "track",
               "camera": str(Path("data/cameras") / f"{stem}.json"), "flights": flights_out,
               "frames": {str(k): v for k, v in sorted(frames_out.items())}}, open(out, "w"), indent=0)
    print(f"[ball3d] {stem}: {len(flights_out)} flights in {nchains} chains  →  {out}")
    print(f"  {'frames':>11s} {'free':>5s} {'2D px':>6s} {'apex m':>7s} {'speed m/s':>9s} {'lowest m':>8s} {'+-m':>5s} {'spin':>5s}  net crossing")
    for fl in flights_out:
        nc = fl["net_crossing"]
        ncs = f"at {nc['height_m']:.2f} m (clears by {nc['clears_net_by_m']:+.2f})" if nc else ""
        print(f"  {fl['f0']:5d}-{fl['f1']:<5d} {'yes' if fl['free_flight'] else 'NO':>5s} {fl['median_px']:6.1f} "
              f"{fl['apex_m']:7.2f} {fl['speed_start_mps']:9.1f} {fl['min_height_m']:8.2f} {fl['depth_std_m']:5.2f} {fl['spin_lift_mps2']:+5.1f}  {ncs}")
    write_touches(Path(a.output) / stem / f"touches{sfx}.json", flights_out, frames_out, uv, T, fps, cam, poses, plabels, stem, serves,
                  who, who_box)
    write_rallies(Path(a.output) / stem, sfx, stem)
    plot(out.with_suffix(".png"), flights_out, frames_out, cam, net_h, stem)
    TL.lap("output files and picture")
    TL.report("3D time")


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
