"""Synthetic test of core.ball3d.fit_rally: realistic rallies flown through a calibrated camera.

Each rally: serve (far side) -> pass -> set -> attack -> floor bounce -> bounce up, with random
positions, heights and flight times. Projected at 60 fps through the camera, then made as messy
as our tracking: 2D noise, ball-size noise, missing frames, a few wrong points (another object),
contact times off by a frame. Reports 3D error per frame, apex height error per flight, contact
point and landing point errors, and whether the reported uncertainty is honest.

    python -m tools.sim_ball3d --camera data/cameras/test74.json [--rallies 100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ball3d import BALL_D, G, Link, Segment, fit_rally  # noqa: E402
from core.camera import Camera  # noqa: E402

FPS = 60.0
KINDS = ["serve", "pass", "set", "attack", "rebound"]


def rally(rng):
    """Contact points (x, y, z) and flight durations, near side = x < 0 (the camera's side)."""
    side = rng.uniform(-3.5, 3.5, 6)
    pts = [np.array([rng.uniform(9.5, 11), side[0], rng.uniform(2.4, 3.0)]),       # serve contact
           np.array([rng.uniform(-8, -5), side[1], rng.uniform(0.6, 1.0)]),         # pass
           np.array([rng.uniform(-1.2, -0.5), side[2], rng.uniform(2.2, 2.7)]),     # set
           np.array([rng.uniform(-2.5, -1.0), rng.choice([-1, 1]) * rng.uniform(2.5, 4), rng.uniform(2.8, 3.3)]),  # attack
           np.array([rng.uniform(4, 8.5), side[4], BALL_D / 2]),                   # floor bounce
           np.array([rng.uniform(7, 11), side[5], rng.uniform(0.8, 1.5)])]          # after the bounce
    T = [rng.uniform(0.9, 1.3), rng.uniform(1.0, 1.6), rng.uniform(0.9, 1.4), rng.uniform(0.35, 0.55), rng.uniform(0.35, 0.6)]
    return pts, T


def flight(p0, p1, T):
    v0 = (p1 - p0 - 0.5 * G * T * T) / T
    return v0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="data/cameras/test74.json")
    ap.add_argument("--rallies", type=int, default=100)
    ap.add_argument("--noise-px", type=float, default=3.0, help="2D noise, px (0.1 ball-width ~ 3 px)")
    ap.add_argument("--size-noise", type=float, default=0.10, help="relative ball-size noise")
    ap.add_argument("--missing", type=float, default=0.10, help="fraction of frames missing")
    ap.add_argument("--outliers", type=float, default=0.02, help="fraction of wrong 2D points")
    ap.add_argument("--no-size", action="store_true", help="don't use ball size")
    ap.add_argument("--cal-error", action="store_true",
                    help="fit with a camera off by the calibration's own reported uncertainty (f_std, position_std_m)")
    a = ap.parse_args()
    cam = Camera.load(a.camera)
    rng = np.random.default_rng(1)
    W, H = cam.width, cam.height
    err3d, apex_err, contact_err, land_err, cover, rms = [], {k: [] for k in KINDS}, [], [], [], []
    by_kind = {k: [] for k in KINDS}
    depth_err = []
    for _ in range(a.rallies):
        pts, T = rally(rng)
        t0, segs, truth, links, true_c = 0.0, [], [], [], []
        for k in range(5):
            v0 = flight(pts[k], pts[k + 1], T[k])
            n = int(T[k] * FPS)
            t = t0 + np.arange(1, n) / FPS                    # frames strictly inside the flight
            dt = t - t0
            X = pts[k] + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, G)
            uv = cam.project(X)
            depth = cam.depth(X)
            keep = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H) & (rng.random(len(t)) > a.missing)
            obs = uv + rng.normal(0, a.noise_px, uv.shape)
            bad = rng.random(len(t)) < a.outliers
            obs[bad] += rng.normal(0, 150, (bad.sum(), 2))
            size = cam.f * BALL_D / depth * (1 + rng.normal(0, a.size_noise, len(t)))
            if a.no_size:
                size[:] = np.nan
            segs.append(Segment(t[keep], obs[keep], size[keep]))
            truth.append((t[keep], X[keep]))
            t0 += T[k]
            if k < 4:
                tc = t0 + rng.choice([-1, 0, 1]) / FPS            # contact detection off by a frame
                links.append(Link(k, tc, "bounce" if k == 3 else "touch"))
                true_c.append(pts[k + 1])
        if any(len(s.t) < 3 for s in segs):
            continue
        fcam = cam
        if a.cal_error:
            fs = cam.info.get("f_std", 0.015 * cam.f)
            ps = np.asarray(cam.info.get("position_std_m", [0.1, 0.05, 0.05]))
            C = cam.position + rng.normal(0, 1, 3) * ps
            R = cam.R
            fcam = Camera(cam.width, cam.height, cam.f + rng.normal(0, fs), list(cam.rvec), list(-R @ C), cam.k1, cam.cx, cam.cy)
        fit = fit_rally(segs, links, fcam)
        rms.append(fit.rms_px)
        for k in range(5):
            t, X = truth[k]
            est = fit.at(k, t)
            e = np.linalg.norm(est - X, axis=1)
            err3d.extend(e)
            by_kind[KINDS[k]].extend(e)
            sd = fit.std(k, t)
            cover.extend((np.abs(est - X) <= 2 * np.maximum(sd, 1e-6)).all(axis=1))
            depth_err.extend(np.abs(cam.depth(est) - cam.depth(X)))
            v0 = flight(pts[k], pts[k + 1], T[k])
            tz = v0[2] / 9.81
            true_apex = pts[k][2] + v0[2] * tz - 0.5 * 9.81 * tz * tz if 0 < tz < T[k] else max(pts[k][2], pts[k + 1][2])
            ta, za = fit.apex(k)
            seg_t0 = sum(T[:k])
            if not (seg_t0 < ta < seg_t0 + T[k]):                 # apex outside the flight: compare the max height
                tt = np.linspace(seg_t0, seg_t0 + T[k], 50)
                za = float(fit.at(k, tt)[:, 2].max())
            apex_err[KINDS[k]].append(abs(za - true_apex))
        for L, P, tc in zip(links, true_c, fit.info["touch_t"]):
            mid = 0.5 * (fit.at(L.k, tc)[0] + fit.at(L.k + 1, tc)[0])
            (land_err if L.kind == "bounce" else contact_err).append(np.linalg.norm(mid - P))
    cm = lambda v: f"{100 * np.median(v):5.1f} / {100 * np.percentile(v, 95):5.1f}"
    print(f"{len(rms)} rallies through {a.camera}: 2D noise {a.noise_px} px, size noise {a.size_noise:.0%}"
          f"{' (size NOT used)' if a.no_size else ''}, {a.missing:.0%} missing, {a.outliers:.0%} wrong points")
    print(f"  fit: median 2D error {np.median(rms):.1f} px")
    print(f"  ball position, every frame     {cm(err3d)} cm (median / 95%)   depth part {cm(depth_err)} cm")
    print(f"  contact points (touches)       {cm(contact_err)} cm")
    print(f"  landing point (bounce)         {cm(land_err)} cm")
    for k in KINDS:
        print(f"  {k:8s}  position {cm(by_kind[k])} cm   apex height {cm(apex_err[k])} cm")
    print(f"  reported uncertainty honest: truth within 2 sigma in {np.mean(cover):.0%} of frames (should be ~90%+)")


if __name__ == "__main__":
    main()
