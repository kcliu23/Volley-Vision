"""The ball in 3D (meters, court coordinates) from one calibrated camera.

One camera gives a line of sight per frame, not a depth. The depth comes from physics and the
scene:
  gravity      between touches the ball flies a parabola with known g = 9.81 m/s^2 straight
               down (core.court's z axis): 6 unknowns per flight (position + velocity), 2
               measurements per frame, so a handful of frames fixes the flight in meters
  ball size    a 21 cm ball looks smaller further away: depth ~ f * 0.21 / size_px. Noisy per
               frame (blur, box fit), so a weak term; it matters most for flights going toward
               or away from the camera, where gravity alone pins depth poorly
  touches      a flight ends where the next one starts (same point, same time): short flights
               (blocks, tips) borrow depth from their neighbours
  bounces      at a floor bounce the ball's centre is one radius above the floor

fit_rally() solves all flights of a rally together by robust least squares (soft-L1: a few
wrong 2D points, e.g. a shoe, don't drag the fit) and returns per-frame 3D positions with 1-sigma
uncertainties from the fit. Air drag is ignored (first version): check the residuals on fast
attacks before trusting it there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = np.array([0.0, 0.0, -9.81])
BALL_D = 0.21                      # m, volleyball diameter (65-67 cm circumference)


@dataclass
class Segment:
    """2D observations of one flight (between two contacts)."""
    t: np.ndarray                  # seconds
    uv: np.ndarray                 # (N, 2) pixels
    size: np.ndarray               # (N,) ball diameter in pixels, NaN where unknown
    w: np.ndarray | None = None    # (N,) weights (1 = YOLO-confirmed; less for finder-only points)
    frames: np.ndarray | None = None

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        self.uv = np.asarray(self.uv, float).reshape(-1, 2)
        self.size = np.asarray(self.size, float)
        self.w = np.ones(len(self.t)) if self.w is None else np.asarray(self.w, float)
        self.tr = float(np.mean(self.t)) if len(self.t) else 0.0     # reference time of the parameters


@dataclass
class Link:
    """Segment k ends and segment k+1 starts at time t (a touch), or the ball bounces there."""
    k: int
    t: float
    kind: str = "touch"            # "touch" (same point) or "bounce" (same point, on the floor)


@dataclass
class Fit3D:
    params: list                   # per segment: (p0 (3,), v0 (3,)) at the segment's reference time
    tr: list
    cov: list                      # per segment 6x6 covariance (or None)
    rms_px: float                  # median reprojection error (px): robust to the few wrong 2D points
    seg_rms_px: list
    info: dict = field(default_factory=dict)

    def at(self, k: int, t) -> np.ndarray:
        p0, v0 = self.params[k]
        dt = np.atleast_1d(np.asarray(t, float)) - self.tr[k]
        return p0 + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, G)

    def std(self, k: int, t) -> np.ndarray:
        """1-sigma position uncertainty (N, 3) at times t."""
        c = self.cov[k]
        dt = np.atleast_1d(np.asarray(t, float)) - self.tr[k]
        if c is None:
            return np.full((len(dt), 3), np.nan)
        out = np.zeros((len(dt), 3))
        for i, d in enumerate(dt):
            A = np.hstack([np.eye(3), d * np.eye(3)])                 # p = A theta + const
            out[i] = np.sqrt(np.maximum(np.diag(A @ c @ A.T), 0))
        return out

    def apex(self, k: int) -> tuple[float, float]:
        """(time, height) of the highest point of flight k (unconstrained by its time span)."""
        p0, v0 = self.params[k]
        dt = v0[2] / 9.81
        return self.tr[k] + dt, float(p0[2] + v0[2] * dt - 0.5 * 9.81 * dt ** 2)


class _Proj:
    """Fast vectorized projection with the calibrated camera (pinhole + k1)."""

    def __init__(self, cam):
        self.R = cam.R
        self.t = np.asarray(cam.tvec, float)
        self.f, self.cx, self.cy, self.k1 = cam.f, cam.cx, cam.cy, cam.k1
        self.c = cam.position
        self.cam = cam

    def __call__(self, X):
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        x, y = Xc[:, 0] / z, Xc[:, 1] / z
        r = 1 + self.k1 * (x * x + y * y)
        return np.c_[self.f * x * r + self.cx, self.f * y * r + self.cy], z


def _init_segment(seg: Segment, P: _Proj, default_depth: float) -> np.ndarray:
    """Initial (p0, v0): each frame's point at the depth its ball size suggests (or a default
    depth), then the best parabola through those points (linear least squares, g known)."""
    c, d = P.cam.rays(seg.uv)
    depth = np.where(np.isfinite(seg.size) & (seg.size > 2), P.f * BALL_D / np.maximum(seg.size, 1e-6), np.nan)
    if np.isfinite(depth).sum() >= 2:
        depth = np.where(np.isfinite(depth), depth, np.nanmedian(depth))
    else:
        depth = np.full(len(seg.t), default_depth)
    # ray distance -> point (depth here is along the optical axis; close enough for a start)
    X = c + d * depth[:, None]
    dt = seg.t - seg.tr
    Y = X - 0.5 * np.outer(dt ** 2, G)
    A = np.c_[np.ones(len(dt)), dt]
    sol, *_ = np.linalg.lstsq(A * np.sqrt(seg.w)[:, None], Y * np.sqrt(seg.w)[:, None], rcond=None)
    return np.r_[sol[0], sol[1] if len(dt) > 1 else np.zeros(3)]


def fit_rally(segs: list[Segment], links: list[Link], cam, sigma_px: float = 3.0, size_rel: float = 0.15,
              link_sigma: float = 0.10, bounce_sigma: float = 0.05, default_depth: float = 12.0,
              loss: str = "soft_l1", time_sigma: float = 1 / 60) -> Fit3D:
    """All flights of a rally together. sigma_px: 2D position noise; size_rel: relative noise of
    the ball's pixel size; link_sigma (m): how exactly consecutive flights meet at a touch;
    time_sigma (s): how far off a detected touch time may be. Each touch's exact time is solved
    too (at attack speed one frame is ~30 cm, so a touch a frame off would bend both flights)."""
    from scipy.optimize import least_squares
    P = _Proj(cam)
    K = len(segs)
    theta0 = np.concatenate([_init_segment(s, P, default_depth) for s in segs])
    NL = len(links)
    theta0 = np.r_[theta0, np.zeros(NL)]                  # + one time offset per touch
    def unpack(th):
        return [(th[6 * k:6 * k + 3], th[6 * k + 3:6 * k + 6]) for k in range(K)]

    def pos(k, th, t):
        p0, v0 = th[6 * k:6 * k + 3], th[6 * k + 3:6 * k + 6]
        dt = np.atleast_1d(t) - segs[k].tr
        return p0 + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, G)

    def resid(th):
        out = []
        for k, s in enumerate(segs):
            X = pos(k, th, s.t)
            uv, z = P(X)
            sw = np.sqrt(s.w)
            out.append(((uv - s.uv) / sigma_px * sw[:, None]).ravel())
            ok = np.isfinite(s.size) & (s.size > 2)
            if ok.any():
                pred = P.f * BALL_D / np.maximum(z[ok], 0.1)
                out.append((pred - s.size[ok]) / (size_rel * s.size[ok]) * sw[ok])
        for j, L in enumerate(links):
            if not 0 <= L.k < K - 1:
                continue
            tc = L.t + th[6 * K + j]
            a, b = pos(L.k, th, tc)[0], pos(L.k + 1, th, tc)[0]
            out.append((a - b) / link_sigma)
            out.append(np.array([th[6 * K + j] / time_sigma]))
            if L.kind == "bounce":
                out.append(np.array([a[2] - BALL_D / 2, b[2] - BALL_D / 2]) / bounce_sigma)
        return np.concatenate(out)

    r = least_squares(resid, theta0, loss=loss, f_scale=2.0, x_scale="jac", max_nfev=200 * len(theta0))
    th = r.x
    # per-segment reprojection rms and covariance
    seg_rms, all_e = [], []
    for k, s in enumerate(segs):
        uv, _ = P(pos(k, th, s.t))
        e = np.hypot(*(uv - s.uv).T)
        seg_rms.append(float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"))
        all_e.extend(e.tolist())
    covs = [None] * K
    try:
        J = r.jac
        res = r.fun
        dof = max(len(res) - len(th), 1)
        s2 = max(float(np.sum(res ** 2)) / dof, 1.0)
        C = np.linalg.pinv(J.T @ J) * s2
        covs = [C[6 * k:6 * k + 6, 6 * k:6 * k + 6] for k in range(K)]
    except Exception:
        pass
    touch_t = [L.t + float(th[6 * K + j]) for j, L in enumerate(links)]
    return Fit3D(unpack(th), [s.tr for s in segs], covs, float(np.median(all_e)) if all_e else float("nan"),
                 seg_rms, {"cost": float(r.cost), "nfev": int(r.nfev), "status": int(r.status),
                           "touch_t": touch_t, "median_px": float(np.median(all_e)) if all_e else None})
