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

  air drag     a volleyball is light for its size: dv/dt = g - k |v| v with k = rho Cd A / (2 m)
               ~ 0.035 /m (terminal speed ~17 m/s): at 10 m/s drag decelerates it by a third of
               g, so a pure parabola is wrong on long and fast flights. k comes from the ball's
               own weight and size (FIVB: 260-280 g, 65-67 cm round) and air density, Cd ~0.45
               below the drag crisis; it can also be fitted per clip (fit_rally(fit_drag=True))

fit_rally() solves all flights of a rally together by robust least squares (soft-L1: a few
wrong 2D points, e.g. a shoe, don't drag the fit) and returns per-frame 3D positions with 1-sigma
uncertainties from the fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import math

G = np.array([0.0, 0.0, -9.81])
BALL_D = 0.21                      # m, volleyball diameter (65-67 cm circumference)
BALL_M = 0.27                      # kg (260-280 g)
AIR_RHO, BALL_CD = 1.2, 0.45
DRAG_K = AIR_RHO * BALL_CD * math.pi * (BALL_D / 2) ** 2 / (2 * BALL_M)     # ~0.035 1/m
STEP = 1 / 240                     # integration step (s)


def traj(p0, v0, tr: float, ts, k: float = DRAG_K) -> np.ndarray:
    """Ball positions (N, 3) at times ts for a flight that is at p0 with velocity v0 at time tr,
    under gravity and quadratic air drag k (k = 0: the exact parabola). RK4 in plain floats
    (small loops: much faster than numpy here), both directions from tr, linear interpolation
    between the 1/240 s steps (error ~1e-5 m)."""
    ts = np.atleast_1d(np.asarray(ts, float))
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float)
    if k == 0:
        dt = ts - tr
        return p0 + np.outer(dt, v0) + 0.5 * np.outer(dt ** 2, G)
    out = np.empty((len(ts), 3))
    g = -9.81
    for sgn in (1, -1):
        m = (ts >= tr) if sgn == 1 else (ts < tr)
        if not m.any():
            continue
        T = float(np.max(np.abs(ts[m] - tr)))
        n = max(1, int(math.ceil(T / STEP)))
        h = sgn * T / n
        x, y, z = p0
        u, v, w = v0
        X = [(x, y, z)]
        for _ in range(n):
            if u * u + v * v + w * w > 1e4:                # > 100 m/s: a diverging guess; freeze it
                X.append((x, y, z))
                continue
            def acc(u, v, w):
                s = k * math.sqrt(u * u + v * v + w * w)
                return -s * u, -s * v, g - s * w
            a1 = acc(u, v, w)
            u2, v2, w2 = u + 0.5 * h * a1[0], v + 0.5 * h * a1[1], w + 0.5 * h * a1[2]
            a2 = acc(u2, v2, w2)
            u3, v3, w3 = u + 0.5 * h * a2[0], v + 0.5 * h * a2[1], w + 0.5 * h * a2[2]
            a3 = acc(u3, v3, w3)
            u4, v4, w4 = u + h * a3[0], v + h * a3[1], w + h * a3[2]
            a4 = acc(u4, v4, w4)
            x += h / 6 * (u + 2 * u2 + 2 * u3 + u4)
            y += h / 6 * (v + 2 * v2 + 2 * v3 + v4)
            z += h / 6 * (w + 2 * w2 + 2 * w3 + w4)
            u += h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
            v += h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
            w += h / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
            X.append((x, y, z))
        X = np.array(X)
        grid = np.abs(np.arange(n + 1) * h)
        q = np.abs(ts[m] - tr)
        out[m] = np.c_[np.interp(q, grid, X[:, 0]), np.interp(q, grid, X[:, 1]), np.interp(q, grid, X[:, 2])]
    return out


@dataclass
class Segment:
    """2D observations of one flight (between two contacts)."""
    t: np.ndarray                  # seconds
    uv: np.ndarray                 # (N, 2) pixels
    size: np.ndarray               # (N,) ball diameter in pixels, NaN where unknown
    w: np.ndarray | None = None    # (N,) weights (1 = YOLO-confirmed; less for finder-only points)
    frames: np.ndarray | None = None
    out_t: np.ndarray | None = None  # times when the ball was OUT of the picture (left at an edge
    box: tuple | None = None         # and came back): there the arc must project outside `box`

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        self.uv = np.asarray(self.uv, float).reshape(-1, 2)
        self.size = np.asarray(self.size, float)
        self.w = np.ones(len(self.t)) if self.w is None else np.asarray(self.w, float)
        self.out_t = np.zeros(0) if self.out_t is None else np.asarray(self.out_t, float)
        # reference time of the parameters: the first frame, so the flight integrates forwards
        # (backwards through drag the speed grows without bound)
        self.tr = float(self.t[0]) if len(self.t) else 0.0


@dataclass
class Link:
    """Segment k ends and segment k+1 starts at time t (a touch), or the ball bounces there."""
    k: int
    t: float
    kind: str = "touch"            # "touch" (same point) or "bounce" (same point, on the floor)
    t_sigma: float | None = None   # how well the touch time is known (s); None = fit_rally's time_sigma


@dataclass
class Fit3D:
    params: list                   # per segment: (p0 (3,), v0 (3,)) at the segment's reference time
    tr: list
    cov: list                      # per segment 6x6 covariance (or None)
    rms_px: float                  # median reprojection error (px): robust to the few wrong 2D points
    seg_rms_px: list
    info: dict = field(default_factory=dict)
    drag: float = DRAG_K

    def at(self, k: int, t) -> np.ndarray:
        p0, v0 = self.params[k]
        return traj(p0, v0, self.tr[k], t, self.drag)

    def std(self, k: int, t) -> np.ndarray:
        """1-sigma position uncertainty (N, 3) at times t (fit covariance pushed through the
        trajectory by finite differences)."""
        c = self.cov[k]
        t = np.atleast_1d(np.asarray(t, float))
        if c is None:
            return np.full((len(t), 3), np.nan)
        p0, v0 = self.params[k]
        th = np.r_[p0, v0]
        base = traj(p0, v0, self.tr[k], t, self.drag)
        J = np.zeros((len(t), 3, 6))
        for i in range(6):
            d = np.zeros(6); d[i] = 1e-4
            q = th + d
            J[:, :, i] = (traj(q[:3], q[3:], self.tr[k], t, self.drag) - base) / 1e-4
        return np.sqrt(np.maximum(np.einsum("nij,jk,nik->ni", J, c, J), 0))

    def apex(self, k: int, t0: float | None = None, t1: float | None = None) -> tuple[float, float]:
        """(time, height) of the highest point of flight k between t0 and t1 (default: 1.5 s
        either side of its reference time)."""
        t0 = self.tr[k] - 1.5 if t0 is None else t0
        t1 = self.tr[k] + 1.5 if t1 is None else t1
        ts = np.arange(t0, t1 + STEP, STEP)
        X = self.at(k, ts)
        i = int(np.argmax(X[:, 2]))
        return float(ts[i]), float(X[i, 2])


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
              loss: str = "soft_l1", time_sigma: float = 1 / 60, drag: float = DRAG_K,
              fit_drag: bool = False) -> Fit3D:
    """All flights of a rally together. sigma_px: 2D position noise; size_rel: relative noise of
    the ball's pixel size; link_sigma (m): how exactly consecutive flights meet at a touch;
    time_sigma (s): how far off a detected touch time may be. Each touch's exact time is solved
    too (at attack speed one frame is ~30 cm, so a touch a frame off would bend both flights).
    drag: air-drag constant k (0 = pure parabola); fit_drag: also fit k (one value for the rally)."""
    from scipy.optimize import least_squares
    P = _Proj(cam)
    K = len(segs)
    theta0 = np.concatenate([_init_segment(s, P, default_depth) for s in segs])
    NL = len(links)
    theta0 = np.r_[theta0, np.zeros(NL)]                  # + one time offset per touch
    if fit_drag:
        theta0 = np.r_[theta0, math.log(max(drag, 1e-4))]

    def kd(th):
        return math.exp(th[-1]) if fit_drag else drag

    def unpack(th):
        return [(th[6 * k:6 * k + 3], th[6 * k + 3:6 * k + 6]) for k in range(K)]

    def pos(k, th, t):
        return traj(th[6 * k:6 * k + 3], th[6 * k + 3:6 * k + 6], segs[k].tr, t, kd(th))

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
            if len(s.out_t) and s.box is not None:         # out of the picture: no closer than its edge
                uo, _ = P(pos(k, th, s.out_t))
                x0, y0, x1, y1 = s.box
                inside = np.minimum.reduce([uo[:, 0] - x0, x1 - uo[:, 0], uo[:, 1] - y0, y1 - uo[:, 1]])
                out.append(np.maximum(inside, 0.0) / sigma_px)
        for j, L in enumerate(links):
            if not 0 <= L.k < K - 1:
                continue
            tc = L.t + th[6 * K + j]
            a, b = pos(L.k, th, tc)[0], pos(L.k + 1, th, tc)[0]
            out.append((a - b) / link_sigma)
            out.append(np.array([th[6 * K + j] / (L.t_sigma or time_sigma)]))
            if L.kind == "bounce":
                out.append(np.array([a[2] - BALL_D / 2, b[2] - BALL_D / 2]) / bounce_sigma)
        if fit_drag:                                       # weak prior: the ball's physics value, x/÷ 2
            out.append(np.array([(th[-1] - math.log(DRAG_K)) / math.log(2)]))
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
                           "touch_t": touch_t, "median_px": float(np.median(all_e)) if all_e else None,
                           "drag": kd(th)}, kd(th))
