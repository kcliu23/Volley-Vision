"""Synthetic rallies for the 3D step (the one simulator): known 3D truth, flown through a real
calibrated camera, made as messy as real clips, then scored for each mechanism on the same rallies:

    ideal  the true touches handed to the joint fit (core.ball3d.fit_rally): the best the physics
           can do with this camera and this noise (what tools.sim_ball3d used to measure)
    v1     the first mechanism (core.legacy_v1: tracker splits + merge / re-split)
    v2     the global rally fit (core.rally3d): touches chosen by the fit

--cal-error fits with a camera that is off by the calibration's own reported uncertainty, so the
cost of a calibration error shows. Note: the rallies use the same physics as the fit, so this
tests the search and the noise handling, not whether the physics matches real video.

Each rally (drag on, through a real calibrated camera): serve -> pass -> set -> attack -> (block,
30%) -> floor bounce -> rebound, random positions / heights / speeds. Then made as messy as real
clips: 2D noise, size noise, 10% missing frames, 2% wrong points, the ball leaving the picture
(whatever leaves it), an occlusion window (50% of rallies, 6-12 frames, half of them hiding a
touch), and tracker contact hints that miss 25% of real touches, are +-2 frames off, and add
about one fake touch per rally (a fake split).

Scores: per-frame 3D error (median / 95%), coverage (frames with a 3D position), "teleports"
(frames off by more than 1 m), touch recall / precision (within 3 frames).

    python -m tools.sim_rally --rallies 20                 # ideal, v1, v2
    python -m tools.sim_rally --rallies 20 --only v2 --cal-error
"""
from __future__ import annotations

import argparse
import time

import numpy as np


from core.ball3d import BALL_D, DRAG_K, G, Link, fit_rally, out_of_picture, traj
from core.camera import Camera

FPS = 60.0


def flight(p0, p1, T, k=0.0):
    """Start velocity that takes the ball from p0 to p1 in T seconds (with drag k: shooting)."""
    v0 = (p1 - p0 - 0.5 * G * T * T) / T
    if k == 0:
        return v0
    from scipy.optimize import fsolve
    return fsolve(lambda v: traj(p0, v, 0.0, [T], k)[0] - p1, v0)


def make_rally(rng, block: bool):
    side = rng.uniform(-3.5, 3.5, 8)
    pts = [np.array([rng.uniform(9.5, 11), side[0], rng.uniform(2.4, 3.0)]),        # serve
           np.array([rng.uniform(-8, -5), side[1], rng.uniform(0.6, 1.0)]),          # pass
           np.array([rng.uniform(-1.2, -0.5), side[2], rng.uniform(2.2, 2.7)]),      # set
           np.array([rng.uniform(-2.0, -1.0), rng.choice([-1, 1]) * rng.uniform(2.5, 4), rng.uniform(2.9, 3.3)])]  # attack
    T = [rng.uniform(0.9, 1.3), rng.uniform(1.0, 1.6), rng.uniform(0.9, 1.4)]
    if block:                                   # block at the net, ball back to the near court
        pts.append(np.array([rng.uniform(0.15, 0.35), pts[-1][1] * rng.uniform(0.6, 1.0), rng.uniform(2.6, 3.0)]))
        T.append(rng.uniform(0.08, 0.15))
        pts.append(np.array([rng.uniform(-7, -3), side[4], BALL_D / 2]))
        T.append(rng.uniform(0.5, 0.8))
    else:
        pts.append(np.array([rng.uniform(4, 8.5), side[4], BALL_D / 2]))
        T.append(rng.uniform(0.35, 0.55))
    last = pts[-1]
    pts.append(np.array([last[0] + np.sign(last[0]) * rng.uniform(1.5, 3), side[5], rng.uniform(0.8, 1.4)]))
    T.append(rng.uniform(0.35, 0.6))
    return pts, T


def simulate(cam: Camera, rng, a):
    k = DRAG_K * (1 + rng.uniform(-0.2, 0.2))
    pts, T = make_rally(rng, rng.random() < a.block)
    t_touch = np.cumsum([0.0] + T)                        # touch times (incl. start / end)
    n = int(t_touch[-1] * FPS)
    tt = np.arange(n) / FPS
    X = np.zeros((n, 3))
    for i in range(len(T)):
        v0 = flight(pts[i], pts[i + 1], T[i], k)
        m = (tt >= t_touch[i]) & (tt < t_touch[i + 1])
        X[m] = traj(pts[i], v0, t_touch[i], tt[m], k)
    uv = cam.project(X)
    depth = cam.depth(X)
    W, H = cam.width, cam.height
    inside = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H) & (depth > 0.5)
    seen = inside & (rng.random(n) > a.missing)
    true_touch_frames = [int(round(t * FPS)) for t in t_touch[1:-1]]
    if rng.random() < 0.5:                                # an occlusion window
        L = int(rng.integers(6, 13))
        c = rng.choice(true_touch_frames) if rng.random() < 0.5 else int(rng.integers(L, n - L))
        seen[max(0, c - L // 2):c + L // 2] = False
    obs_uv = uv + rng.normal(0, a.noise_px, uv.shape)
    bad = rng.random(n) < a.outliers
    obs_uv[bad] += rng.normal(0, 150, (bad.sum(), 2))
    size = cam.f * BALL_D / np.maximum(depth, 0.5) * (1 + rng.normal(0, 0.10, n))
    obs_uv[~seen] = np.nan
    size[~seen] = np.nan
    hints = [f + int(rng.integers(-2, 3)) for f in true_touch_frames if rng.random() < 0.75]
    hints += [int(rng.integers(5, n - 5)) for _ in range(rng.poisson(1.0))]
    return dict(X=X, uv=obs_uv, size=size, seen=seen, touches=true_touch_frames, hints=sorted(hints), n=n,
                t_touch=t_touch[1:-1])


def edge_and_box(cam, uv, ball=30.0):
    box = (0, 0, cam.width, cam.height)
    x, y = uv[:, 0], uv[:, 1]
    with np.errstate(invalid="ignore"):
        e = np.isfinite(x) & ((x < ball) | (x > cam.width - ball) | (y < ball) | (y > cam.height - ball))
    return e, box


def tracker_flights(seen, hints):
    """What the 2D splitter gives: runs of seen frames, cut at contact hints and gaps > 3 frames."""
    fl, cur, last = [], [], None
    hs = set(hints)
    for f in np.where(seen)[0]:
        if cur and (f - last > 3 or any(last < h <= f for h in hs)):
            fl.append((cur[0], cur[-1])); cur = []
        cur.append(int(f)); last = f
    if cur:
        fl.append((cur[0], cur[-1]))
    return [{"f0": a, "f1": b} for a, b in fl if b - a >= 2]


def run_ideal(cam, s, edge, box):
    """The true touches, handed to the joint fit."""
    from core.rally3d import Obs
    fr = np.where(s["seen"])[0]
    full = np.arange(fr[0], fr[-1] + 1)
    out_f = out_of_picture(full, s["seen"][full], edge).astype(int)
    o = Obs(fr / FPS, s["uv"][fr], s["size"][fr], np.where(edge[fr], 0.2, 1.0), fr, edge[fr], out_f, box, FPS)
    cuts = [0] + [int(np.searchsorted(fr, tf)) for tf in s["touches"]] + [len(fr)]
    keep = [k for k in range(len(cuts) - 1) if cuts[k + 1] - cuts[k] >= 3]
    segs = [o.segment(cuts[k], cuts[k + 1]) for k in keep]
    links = [Link(j, float(s["t_touch"][keep[j + 1] - 1]), "touch") for j in range(len(keep) - 1)
             if keep[j + 1] == keep[j] + 1]
    fit = fit_rally(segs, links, cam)
    out = {}
    for i, k in enumerate(keep):
        f0, f1 = int(fr[cuts[k]]), int(fr[cuts[k + 1] - 1])
        g = np.arange(f0, f1 + 1)
        for f, x in zip(g, fit.at(i, g / FPS)):
            out[int(f)] = x
    return out, [0.5 * (fr[cuts[k] - 1] + fr[cuts[k]]) for k in keep[1:]]


def run_v1(cam, s, edge, box):
    from core.legacy_v1 import solve_v1
    w = np.where(edge, 0.2, 1.0)
    contacts = [{"frame": float(h), "kind": "touch"} for h in s["hints"]]
    pieces = solve_v1(s["uv"], w, s["size"], edge, box, FPS, cam, tracker_flights(s["seen"], s["hints"]), contacts,
                      log=lambda m: None)
    out, touches = {}, []
    for k, (ci, seg, (f0, f1), fit, i) in enumerate(pieces):
        fr = np.arange(f0, f1 + 1)
        for f, x in zip(fr, fit.at(i, fr / FPS)):
            out[int(f)] = x
        if k > 0 and pieces[k - 1][0] == ci:
            touches.append(0.5 * (pieces[k - 1][2][1] + f0))
    return out, touches


def run_v2(cam, s, edge, box):
    from core.rally3d import Obs, solve
    fr = np.where(s["seen"])[0]
    full = np.arange(fr[0], fr[-1] + 1)
    out_f = out_of_picture(full, s["seen"][full], edge).astype(int)
    o = Obs(fr / FPS, s["uv"][fr], s["size"][fr], np.where(edge[fr], 0.2, 1.0), fr, edge[fr], out_f, box, FPS)
    pieces, info = solve(o, cam, s["hints"], log=lambda m: None)
    out, touches = {}, []
    for k, (seg, (f0, f1), fit, i) in enumerate(pieces):
        g = np.arange(f0, f1 + 1)
        for f, x in zip(g, fit.at(i, g / FPS)):
            out[int(f)] = x
        if k > 0:
            p0 = pieces[k - 1][1][1]
            for f in range(p0 + 1, f0):                     # the frames at the touch: either side's arc
                out[f] = fit.at(i, [f / FPS])[0]
            touches.append(0.5 * (p0 + f0))
    return out, touches


def score(s, out, touches):
    X = s["X"]
    err = np.array([np.linalg.norm(out[f] - X[f]) for f in out if 0 <= f < len(X)])
    cov = sum(1 for f in range(len(X)) if f in out) / len(X)
    true = s["touches"]
    hit = sum(1 for t in true if any(abs(t - u) <= 3 for u in touches))
    good = sum(1 for u in touches if any(abs(t - u) <= 3 for t in true))
    return err, cov, hit, len(true), good, len(touches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="data/cameras/test74.json")
    ap.add_argument("--rallies", type=int, default=30)
    ap.add_argument("--noise-px", type=float, default=3.0)
    ap.add_argument("--missing", type=float, default=0.10)
    ap.add_argument("--outliers", type=float, default=0.02)
    ap.add_argument("--block", type=float, default=0.3)
    ap.add_argument("--only", choices=["ideal", "v1", "v2"], default=None)
    ap.add_argument("--cal-error", action="store_true",
                    help="fit with a camera off by the calibration's reported uncertainty (f_std, position_std_m)")
    ap.add_argument("--touch-cost", type=float, default=None)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    if a.touch_cost is not None:
        import core.rally3d as R
        R.TOUCH_COST = a.touch_cost
    cam = Camera.load(a.camera)
    rng = np.random.default_rng(a.seed)
    rallies = [simulate(cam, rng, a) for _ in range(a.rallies)]
    fcam = cam                                           # the camera the fits get
    if a.cal_error:
        fs = cam.info.get("f_std", 0.015 * cam.f)
        ps = np.asarray(cam.info.get("position_std_m", [0.1, 0.05, 0.05]))
        C = cam.position + rng.normal(0, 1, 3) * ps
        fcam = Camera(cam.width, cam.height, cam.f + rng.normal(0, fs), list(cam.rvec), list(-cam.R @ C), cam.k1,
                      cam.cx, cam.cy)
    for name, run in (("ideal", run_ideal), ("v1 first", run_v1), ("v2 global", run_v2)):
        if a.only and not name.startswith(a.only):
            continue
        E, cov, hit, ntrue, good, nfound, tele, t0 = [], [], 0, 0, 0, 0, 0, time.time()
        for s in rallies:
            edge, box = edge_and_box(cam, s["uv"])
            try:
                out, touches = run(fcam, s, edge, box)
            except Exception as e:                           # count a crash as nothing found
                print(f"  {name}: rally failed ({type(e).__name__}: {e})")
                out, touches = {}, []
            e_, c_, h_, nt, g_, nf = score(s, out, touches)
            E.extend(e_); cov.append(c_); hit += h_; ntrue += nt; good += g_; nfound += nf
            tele += int((e_ > 1.0).sum()) if len(e_) else 0
        E = np.array(E)
        print(f"{name:11s} 3D error {100 * np.median(E):5.1f} / {100 * np.percentile(E, 95):6.1f} cm (median / 95%)   "
              f"coverage {np.mean(cov):.0%}   teleports (>1 m) {tele / max(len(E), 1):.1%} of frames   "
              f"touches: recall {hit / max(ntrue, 1):.0%}, precision {good / max(nfound, 1):.0%}   "
              f"[{(time.time() - t0) / len(rallies):.1f} s/rally]")


if __name__ == "__main__":
    main()
