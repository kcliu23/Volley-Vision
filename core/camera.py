"""Single-camera calibration from known court points: pixels <-> 3D rays in meters.

The clips are screen recordings, so the phone's lens specs don't apply (the picture is scaled
and maybe cropped). Everything is solved from the court instead:
  unknowns   focal length f (pixels; square pixels, principal point at the picture centre),
             optional radial distortion k1 (only with enough points spread out), camera pose
  data       >= 6 clicked points with known court positions (core.court), not all on one line

solve() searches f coarse-to-fine (best pose for each f by PnP), then refines everything together
by least squares and reports the reprojection error and the uncertainty of f and of the camera
position (from the fit's covariance).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    width: int
    height: int
    f: float
    rvec: list
    tvec: list
    k1: float = 0.0
    cx: float | None = None
    cy: float | None = None
    info: dict = field(default_factory=dict)        # fit quality: rms_px, f_std, position_std_m, n_points, ...

    def __post_init__(self):
        if self.cx is None:
            self.cx = self.width / 2
        if self.cy is None:
            self.cy = self.height / 2

    # --- geometry -------------------------------------------------------------------------
    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], float)

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, 0, 0, 0, 0], float)

    @property
    def R(self) -> np.ndarray:
        import cv2
        return cv2.Rodrigues(np.asarray(self.rvec, float))[0]

    @property
    def position(self) -> np.ndarray:
        """Camera centre in court coordinates (meters)."""
        return -self.R.T @ np.asarray(self.tvec, float)

    def project(self, X) -> np.ndarray:
        """(N, 3) court points -> (N, 2) pixels."""
        import cv2
        X = np.asarray(X, float).reshape(-1, 3)
        uv, _ = cv2.projectPoints(X, np.asarray(self.rvec, float), np.asarray(self.tvec, float), self.K, self.dist)
        return uv.reshape(-1, 2)

    def depth(self, X) -> np.ndarray:
        """Distance in front of the camera (meters) of court points; negative = behind it."""
        X = np.asarray(X, float).reshape(-1, 3)
        return (X @ self.R.T + np.asarray(self.tvec, float))[:, 2]

    def rays(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Pixels -> (camera centre, unit directions (N, 3)) in court coordinates."""
        import cv2
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        n = cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)
        d = np.c_[n, np.ones(len(n))] @ self.R            # camera -> world (R^T applied to rows)
        return self.position, d / np.linalg.norm(d, axis=1, keepdims=True)

    def floor_point(self, uv) -> np.ndarray:
        """Where pixels hit the floor (z = 0): (N, 3); NaN for rays that never reach it."""
        c, d = self.rays(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -c[2] / d[:, 2]
        out = c + s[:, None] * d
        out[(s <= 0) | ~np.isfinite(s)] = np.nan
        return out

    @property
    def fov_deg(self) -> float:
        """Horizontal field of view."""
        return math.degrees(2 * math.atan(self.width / 2 / self.f))

    # --- io -------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"schema": "vv-camera/1", "width": self.width, "height": self.height, "f": self.f,
                "cx": self.cx, "cy": self.cy, "k1": self.k1, "rvec": list(map(float, self.rvec)),
                "tvec": list(map(float, self.tvec)), "position_m": [round(float(v), 3) for v in self.position],
                "fov_deg": round(self.fov_deg, 2), "info": self.info}

    @classmethod
    def from_json(cls, d: dict) -> "Camera":
        return cls(d["width"], d["height"], d["f"], d["rvec"], d["tvec"], d.get("k1", 0.0),
                   d.get("cx"), d.get("cy"), d.get("info", {}))

    @classmethod
    def load(cls, path) -> "Camera":
        return cls.from_json(json.load(open(path)))

    def save(self, path, extra: dict | None = None):
        d = self.to_json()
        d.update(extra or {})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(d, open(path, "w"), indent=1)


# --- solving ----------------------------------------------------------------------------------

def _pnp(X, uv, f, cx, cy, k1=0.0):
    """Best pose for a given focal length: (rms px, rvec, tvec) or (inf, None, None)."""
    import cv2
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    dist = np.array([k1, 0, 0, 0, 0], float)
    best = (math.inf, None, None)
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rv, tv = cv2.solvePnP(X, uv, K, dist, flags=flag)
            if not ok:
                continue
            rv, tv = cv2.solvePnPRefineLM(X, uv, K, dist, rv, tv)
        except cv2.error:
            continue
        R = cv2.Rodrigues(rv)[0]
        if np.any((X @ R.T + tv.ravel())[:, 2] <= 0):     # points behind the camera: wrong solution
            continue
        if (-R.T @ tv.ravel())[2] <= 0:                    # camera under the floor: the mirror image
            continue
        pr, _ = cv2.projectPoints(X, rv, tv, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - uv) ** 2, axis=1))))
        if rms < best[0]:
            best = (rms, rv.ravel(), tv.ravel())
    return best


def solve(points: dict, width: int, height: int, net_height: float = 2.43, fit_k1: bool | None = None,
          cx: float | None = None, cy: float | None = None) -> Camera:
    """points: {keypoint name (core.court): (u, v) pixel}. Returns the Camera with info:
    rms_px, n_points, f_std, position_std_m (1-sigma, from the fit), per-point residuals."""
    from scipy.optimize import least_squares
    from core.court import keypoints
    kp = keypoints(net_height)
    names = [n for n in points if n in kp]
    if len(names) < 6:
        raise ValueError(f"need at least 6 known points, got {len(names)}")
    X = np.array([kp[n] for n in names], float)
    uv = np.array([points[n] for n in names], float)
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    if fit_k1 is None:
        fit_k1 = len(names) >= 10

    # 1. focal length: coarse log grid, then a finer one around the best
    grid = np.exp(np.linspace(math.log(0.25 * width), math.log(5.0 * width), 60))
    errs = [_pnp(X, uv, f, cx, cy)[0] for f in grid]
    i = int(np.argmin(errs))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    fine = np.linspace(lo, hi, 30)
    errs = [_pnp(X, uv, f, cx, cy)[0] for f in fine]
    f0 = float(fine[int(np.argmin(errs))])
    rms0, rv, tv = _pnp(X, uv, f0, cx, cy)
    if rv is None:
        raise ValueError("no camera pose fits these points (check the point names)")

    # 2. everything together: f (log), k1, rvec, tvec
    def unpack(p):
        f = math.exp(p[0])
        k1 = p[1] if fit_k1 else 0.0
        return f, k1, p[2:5], p[5:8]

    def resid(p):
        import cv2
        f, k1, r, t = unpack(p)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        pr, _ = cv2.projectPoints(X, r, t, K, np.array([k1, 0, 0, 0, 0], float))
        return (pr.reshape(-1, 2) - uv).ravel()

    p0 = np.r_[math.log(f0), 0.0, rv, tv]
    lo = np.r_[math.log(0.2 * width), -0.5, [-np.inf] * 6]
    hi = np.r_[math.log(10 * width), 0.5, [np.inf] * 6]
    r = least_squares(resid, p0, method="trf", bounds=(lo, hi))
    f, k1, rvec, tvec = unpack(r.x)
    res = resid(r.x).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    cam = Camera(width, height, f, list(rvec), list(tvec), k1, cx, cy)

    # 3. uncertainty from the fit: covariance of the parameters, propagated to f and position
    info = {"rms_px": round(rms, 2), "n_points": len(names), "fit_k1": bool(fit_k1),
            "residual_px": {n: round(float(np.hypot(*e)), 1) for n, e in zip(names, res)}}
    dof = 2 * len(names) - (8 if fit_k1 else 7)
    if dof > 0:
        J = r.jac if fit_k1 else np.delete(r.jac, 1, axis=1)
        s2 = float(np.sum(res ** 2)) / dof
        try:
            cov = np.linalg.inv(J.T @ J) * max(s2, 1.0)    # never claim better than 1 px clicks
            info["f_std"] = round(float(f * math.sqrt(cov[0, 0])), 1)
            # camera position = -R(r)^T t: numerical Jacobian w.r.t. (r, t)
            idx = slice(2, 8) if fit_k1 else slice(1, 7)
            pr = np.r_[rvec, tvec]

            def pos(q):
                import cv2
                return -cv2.Rodrigues(q[:3])[0].T @ q[3:]
            Jp = np.zeros((3, 6))
            for k in range(6):
                dq = np.zeros(6); dq[k] = 1e-6
                Jp[:, k] = (pos(pr + dq) - pos(pr - dq)) / 2e-6
            pc = Jp @ cov[idx, idx] @ Jp.T
            info["position_std_m"] = [round(float(math.sqrt(max(v, 0))), 3) for v in np.diag(pc)]
        except np.linalg.LinAlgError:
            pass
    cam.info = info
    return cam
