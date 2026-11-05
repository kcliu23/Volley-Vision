"""Synthetic test of core.camera.solve: random phone-like cameras behind the court, click noise,
full or partial court visibility. Reports focal error, camera position error, and the 3D error a
perfectly tracked ball would get from the calibration alone (distance from the true ball to the
estimated camera's ray through its pixel), at heights 0-5 m over the court.

    python -m tools.sim_camera [--trials 200] [--noise 1.5]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.camera import Camera, solve  # noqa: E402
from core.court import keypoints  # noqa: E402

W, H = 2622, 1206

SETS = {
    "all points": None,
    "no near corners": lambda n: "near end corner" not in n,
    "middle only (attack lines, centre, net)": lambda n: "corner" not in n,
    "floor only (no net / antenna)": lambda n: not n.startswith(("net", "antenna")),
    "one side only (left)": lambda n: n.endswith("left") or n.startswith(("net top", "net bottom")),
}


def random_camera(rng) -> Camera:
    import cv2
    pos = np.array([rng.uniform(-15, -9.5), rng.uniform(-3, 3), rng.uniform(1.0, 2.5)])
    target = np.array([rng.uniform(-2, 3), rng.uniform(-1, 1), rng.uniform(0.5, 1.5)])
    fwd = target - pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])                    # world -> camera rows: x right, y down, z forward
    roll = math.radians(rng.uniform(-3, 3))
    Rr = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    R = Rr @ R
    rvec = cv2.Rodrigues(R)[0].ravel()
    tvec = -R @ pos
    f = rng.uniform(0.7, 1.4) * W
    return Camera(W, H, f, list(rvec), list(tvec))


def ray_error(true: Camera, est: Camera, P) -> np.ndarray:
    uv = true.project(P)
    c, d = est.rays(uv)
    v = P - c
    closest = c + (v * d).sum(1, keepdims=True) * d
    return np.linalg.norm(P - closest, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--noise", type=float, default=1.5, help="click noise, px (1-sigma)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    kp = keypoints()
    probe = np.array([[x, y, z] for x in np.linspace(-9, 9, 7) for y in np.linspace(-4.5, 4.5, 5)
                      for z in (0.0, 1.5, 3.0, 5.0)])
    print(f"{a.trials} random cameras per row, click noise {a.noise} px, picture {W}x{H}\n")
    print(f"{'points clicked':42s} {'n':>4s} {'f err %':>8s} {'pos err m':>10s} "
          f"{'ball 3D err cm (median / 95%)':>30s} {'f_std ok':>9s}")
    for label, keep in SETS.items():
        fe, pe, be, n_used, cover = [], [], [], [], []
        for _ in range(a.trials):
            cam = random_camera(rng)
            names = [n for n in kp if keep is None or keep(n)]
            X = np.array([kp[n] for n in names])
            uv = cam.project(X)
            vis = (cam.depth(X) > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            pts = {n: tuple(uv[i] + rng.normal(0, a.noise, 2)) for i, n in enumerate(names) if vis[i]}
            if len(pts) < 6:
                continue
            try:
                est = solve(pts, W, H)
            except ValueError:
                continue
            fe.append(abs(est.f - cam.f) / cam.f * 100)
            pe.append(float(np.linalg.norm(est.position - cam.position)))
            P = probe[(cam.depth(probe) > 0)]
            pu = cam.project(P)
            P = P[(pu[:, 0] >= 0) & (pu[:, 0] < W) & (pu[:, 1] >= 0) & (pu[:, 1] < H)]
            be.extend(ray_error(cam, est, P) * 100)
            n_used.append(len(pts))
            if "f_std" in est.info:
                cover.append(abs(est.f - cam.f) <= 2 * est.info["f_std"])
        if not fe:
            print(f"{label:42s}  (too few points visible)"); continue
        print(f"{label:42s} {np.median(n_used):4.0f} {np.median(fe):8.2f} {np.median(pe):10.3f} "
              f"{np.median(be):14.1f} / {np.percentile(be, 95):5.1f} {np.mean(cover) if cover else float('nan'):9.0%}")
    print("\nf_std ok = how often the true focal length is within 2 reported sigma (should be ~95%)")


if __name__ == "__main__":
    main()
