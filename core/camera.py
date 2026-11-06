"""Single-camera calibration from known court points: pixels <-> 3D rays in meters.

The clips are screen recordings, so the phone's lens specs don't apply (the picture is scaled
and maybe cropped). Everything is solved from the court instead:
  unknowns   focal length f (pixels; square pixels, principal point at the picture centre),
             camera pose (radial distortion k1 only on request: phone video is already undistorted)
  data       clicked points with known court positions (core.court), and/or clicks anywhere
             along known lines (2+ per line; core.court.named_lines): a line counts where its
             corners can't be seen (faint far lines, a net wider than the court)

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


def _fit_line2d(pts):
    """Total-least-squares 2D line through clicks: (point on it, unit direction)."""
    P = np.asarray(pts, float)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[0]


def _intersect2d(l1, l2):
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    s = np.linalg.solve(A, q - p)
    return p + s[0] * d


def _intersect3d(a, b, c, d):
    """Where 3D segments a-b and c-d cross (None if they don't meet)."""
    u, v, w = b - a, d - c, a - c
    A = np.array([[u @ u, -(u @ v)], [u @ v, -(v @ v)]])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s, t = np.linalg.solve(A, [-(w @ u), -(w @ v)])
    P, Q = a + s * u, c + t * v
    return P if np.linalg.norm(P - Q) < 1e-6 else None


def solve(points: dict, width: int, height: int, net_height: float = 2.43, fit_k1: bool | None = None,
          cx: float | None = None, cy: float | None = None, line_clicks: dict | None = None) -> Camera:
    """points: {keypoint name (core.court): (u, v)}; line_clicks: {line name (core.court.named_lines):
    [(u, v), ...]} - clicks anywhere along a line, 2+ per line. Points must sit on the projected
    point; line clicks only on the projected line (any position along it), so a line can be used
    where its corners are hard to see. Returns the Camera with info: rms_px (points and line clicks
    together, px), n_points, n_line_clicks, f_std, position_std_m (1-sigma, from the fit),
    residual_px per point, line_px per line (rms distance of its clicks)."""
    import cv2
    from scipy.optimize import least_squares
    from core.court import keypoints, named_lines
    kp = keypoints(net_height)
    NL = named_lines(net_height)
    names = [n for n in points if n in kp]
    X = np.array([kp[n] for n in names], float).reshape(-1, 3)
    uv = np.array([points[n] for n in names], float).reshape(-1, 2)
    lc = {n: np.asarray(v, float) for n, v in (line_clicks or {}).items() if n in NL and len(v) >= 2}
    LA = {n: np.asarray(NL[n][0], float) for n in lc}
    LB = {n: np.asarray(NL[n][1], float) for n in lc}
    n_lc = sum(len(v) for v in lc.values())
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    if fit_k1 is None:
        # off by default: phone video comes out of the phone already undistorted, and on the
        # holdout clips a fitted k1 only bent the camera toward guessed clicks (lines off by
        # ~4.5 px with it, ~3 px without)
        fit_k1 = False

    # starting correspondences: clicked points + where clicked lines cross (in the picture, and
    # in 3D: a sideline and a floor cross-line meet at a known court point, seen or not)
    init_X, init_uv = list(X), list(uv)
    fits = {n: _fit_line2d(v) for n, v in lc.items()}
    ln = list(lc)
    for i in range(len(ln)):
        for j in range(i + 1, len(ln)):
            P3 = _intersect3d(LA[ln[i]], LB[ln[i]], LA[ln[j]], LB[ln[j]])
            if P3 is None:
                continue
            q = _intersect2d(fits[ln[i]], fits[ln[j]])
            if q is None or not (-2 * width < q[0] < 3 * width and -2 * height < q[1] < 3 * height):
                continue
            if any(np.allclose(P3, x) for x in init_X):
                continue
            init_X.append(P3); init_uv.append(q)
    init_X, init_uv = np.array(init_X, float).reshape(-1, 3), np.array(init_uv, float).reshape(-1, 2)
    if 2 * len(names) + n_lc < 10:
        raise ValueError("need more clicks (about 6 points, or 3+ lines with 2-3 clicks each)")

    # 1. focal length candidates: from PnP over a grid when the starting points are not all on
    # the floor; with only floor points every focal length fits them, so try a spread
    planar = len(init_X) == 0 or np.allclose(init_X[:, 2], init_X[0, 2])
    if not planar and len(init_X) >= 6:
        grid = np.exp(np.linspace(math.log(0.25 * width), math.log(5.0 * width), 60))
        errs = np.array([_pnp(init_X, init_uv, f, cx, cy)[0] for f in grid])
        f_cands = list(grid[np.argsort(errs)[:3]])
    else:
        f_cands = list(width * np.array([0.45, 0.6, 0.75, 0.9, 1.1, 1.4]))

    def unpack(p):
        return math.exp(p[0]), (p[1] if fit_k1 else 0.0), p[2:5], p[5:8]

    def proj(p, P):
        f, k1, r, t = unpack(p)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        pr, _ = cv2.projectPoints(np.asarray(P, float).reshape(-1, 3), r, t, K, np.array([k1, 0, 0, 0, 0], float))
        return pr.reshape(-1, 2)

    def line_dist(p, n):
        """Signed px distance of line n's clicks from where the camera draws it."""
        R = cv2.Rodrigues(np.asarray(p[2:5], float))[0]
        a, b = LA[n], LB[n]
        za, zb = (R @ a + p[5:8])[2], (R @ b + p[5:8])[2]
        # the part of the (extended) line in front of the camera: depth z(s) = za + s (zb - za) > 0.5
        s0, s1 = -1.0, 2.0
        if abs(zb - za) > 1e-9:
            sc = (0.5 - za) / (zb - za)
            if zb > za:
                s0 = max(s0, sc + 1e-3)
            else:
                s1 = min(s1, sc - 1e-3)
        elif za <= 0.5:
            s1 = s0 - 1
        if s1 <= s0:
            return np.full(len(lc[n]), 50.0)
        e = proj(p, np.array([a + s0 * (b - a), a + s1 * (b - a)]))
        d = e[1] - e[0]
        d = d / max(np.linalg.norm(d), 1e-9)
        nrm = np.array([-d[1], d[0]])
        return (lc[n] - e[0]) @ nrm

    def resid(p):
        r = [(proj(p, X) - uv).ravel()] if len(X) else []
        r += [line_dist(p, n) for n in lc]
        return np.concatenate(r) if r else np.zeros(1)

    lo = np.r_[math.log(0.2 * width), -0.5, [-np.inf] * 6]
    hi = np.r_[math.log(10 * width), 0.5, [np.inf] * 6]
    starts = []                                            # (f, rvec, tvec) to start the fit from
    if len(init_X) >= 4:
        for f0 in f_cands:
            _, rv, tv = _pnp(init_X, init_uv, f0, cx, cy)
            if rv is not None:
                starts.append((f0, rv, tv))
    if not starts:
        # too few points to start from (e.g. only the net and the sidelines): try plausible
        # phone spots behind the near end line, looking at the court
        for x0 in (-13.0, -10.0, -16.0, -20.0):
            for y0 in (0.0,):
                for f0 in width * np.array([0.7, 0.5, 1.0]):
                    C = np.array([x0, y0, 1.4])
                    fwd = np.array([0.0, 0.0, 1.0]) - C
                    fwd /= np.linalg.norm(fwd)
                    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
                    down = np.cross(fwd, right)
                    R = np.array([right, down, fwd])          # camera axes: x right, y down, z forward
                    starts.append((f0, cv2.Rodrigues(R)[0].ravel(), -R @ C))
    best = None
    for f0, rv, tv in starts:
        p0 = np.clip(np.r_[math.log(f0), 0.0, rv, tv], lo + 1e-9, hi - 1e-9)
        try:
            r = least_squares(resid, p0, method="trf", bounds=(lo, hi))
        except ValueError:
            continue
        R = cv2.Rodrigues(r.x[2:5])[0]
        if (-R.T @ r.x[5:8])[2] <= 0:                      # camera under the floor: mirror image
            continue
        if best is None or r.cost < best.cost:
            best = r
        if not len(init_X) >= 4 and math.sqrt(2 * best.cost / max(len(r.fun), 1)) < 2.0:
            break                                          # a phone-spot start that fits: done
    if best is None:
        raise ValueError("no camera fits these clicks (check the names)")
    r = best
    f, k1, rvec, tvec = unpack(r.x)
    res_all = resid(r.x)
    nres = len(res_all)
    res_pts = res_all[:2 * len(names)].reshape(-1, 2)
    per_click = np.r_[np.hypot(res_pts[:, 0], res_pts[:, 1]), np.abs(res_all[2 * len(names):])]
    rms = float(np.sqrt(np.mean(per_click ** 2)))
    cam = Camera(width, height, f, list(rvec), list(tvec), k1, cx, cy)

    # 3. uncertainty from the fit: covariance of the parameters, propagated to f and position
    line_px, k = {}, 2 * len(names)
    for n in lc:
        e = res_all[k:k + len(lc[n])]; k += len(lc[n])
        line_px[n] = round(float(np.sqrt(np.mean(e ** 2))), 1)
    info = {"rms_px": round(rms, 2), "n_points": len(names), "n_line_clicks": int(n_lc), "fit_k1": bool(fit_k1),
            "residual_px": {n: round(float(np.hypot(*e)), 1) for n, e in zip(names, res_pts)},
            "line_px": line_px}
    dof = nres - (8 if fit_k1 else 7)
    if dof > 0:
        J = r.jac if fit_k1 else np.delete(r.jac, 1, axis=1)
        s2 = float(np.sum(res_all ** 2)) / dof
        try:
            cov = np.linalg.inv(J.T @ J) * max(s2, 1.0)    # never claim better than 1 px clicks
            info["f_std"] = round(float(f * math.sqrt(cov[0, 0])), 1)
            idx = slice(2, 8) if fit_k1 else slice(1, 7)
            pr = np.r_[rvec, tvec]

            def pos(q):
                return -cv2.Rodrigues(q[:3])[0].T @ q[3:]
            Jp = np.zeros((3, 6))
            for kk in range(6):
                dq = np.zeros(6); dq[kk] = 1e-6
                Jp[:, kk] = (pos(pr + dq) - pos(pr - dq)) / 2e-6
            pc = Jp @ cov[idx, idx] @ Jp.T
            info["position_std_m"] = [round(float(math.sqrt(max(v, 0))), 3) for v in np.diag(pc)]
        except np.linalg.LinAlgError:
            pass
    # distance along the court is only fixed by something across it on the floor
    cross = {"centre line", "near attack line", "far attack line", "far end line", "near end line"}
    floor_x = {round(kp[n][0], 2) for n in names if kp[n][2] == 0}
    if not (cross & set(lc)) and len(floor_x) < 2:
        info["warning"] = "add a line across the court (centre / attack / end line): distance is a guess without one"
    cam.info = info
    return cam


# --- snapping to the painted lines ---------------------------------------------------------------

def _ridge_maps(gray: np.ndarray, sigmas=(1.2, 2.5)):
    """Second derivatives (scale-normalized, max over scales by strength) of the picture: a thin
    BRIGHT line has a strongly negative second derivative across it and ~0 along it."""
    import cv2
    g = gray.astype(np.float32)
    best = None
    for s in sigmas:
        b = cv2.GaussianBlur(g, (0, 0), s)
        hxx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3) * s * s
        hyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3) * s * s
        hxy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3) * s * s
        if best is None:
            best = [hxx, hyy, hxy]
        else:                                           # keep the scale with the stronger ridge
            tr_new = -(hxx + hyy)
            tr_old = -(best[0] + best[1])
            m = tr_new > tr_old
            for k, h in enumerate((hxx, hyy, hxy)):
                best[k] = np.where(m, h, best[k])
    return best


def _sample(img, x, y):
    """Bilinear lookup, NaN outside."""
    h, w = img.shape
    ok = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    out = np.full(np.shape(x), np.nan, np.float32)
    xi, yi = np.floor(x[ok]).astype(int), np.floor(y[ok]).astype(int)
    fx, fy = x[ok] - xi, y[ok] - yi
    out[ok] = (img[yi, xi] * (1 - fx) * (1 - fy) + img[yi, xi + 1] * fx * (1 - fy)
               + img[yi + 1, xi] * (1 - fx) * fy + img[yi + 1, xi + 1] * fx * fy)
    return out


def refine_to_lines(cam: Camera, image: np.ndarray, points: dict, net_height: float = 2.43,
                    radii=(30, 12, 6, 3), step_px: float = 12.0, click_sigma: float = 6.0,
                    k1_sigma: float = 0.05, skip=(), check_radius: float = 25.0, log=print) -> Camera:
    """EXPERIMENTAL, not used by the tools: on the holdout clips (faint lines, few of them found)
    the refit drifted along the focal-length / distance trade-off and made held-out lines worse.
    Its last step is still useful as a check: with radii=() nothing is fitted, and
    info["line_px"] says how far the given camera draws each line from where it is found.

    Snap a clicked calibration onto the painted court lines and the net's tape / band.

    Each court line (core.court.lines) is looked for as a whole: within +- radius px of where the
    current camera draws it, the straight line (offset + small tilt) along which the picture has
    the strongest thin ridge running the same way (second derivative across it, so reflections
    and lines at an angle don't count), lighter or darker than the floor, whichever is clearer.
    Summed over the whole visible length, so wood grain, players and glare don't win; a line that
    isn't clearly there, or has a similar line next to it, is skipped. The camera (focal length,
    distortion, pose) is then refit to put the court lines on the found ones, each line counting
    about the same, with your clicks kept as a weak anchor (click_sigma px) and distortion kept
    small (k1_sigma). Repeated with a shrinking search radius. Returns a new Camera; info["lines"]
    says how many lines were found and how far off they sit."""
    import cv2
    from scipy.optimize import least_squares
    from core.court import keypoints, lines

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    hxx, hyy, hxy = _ridge_maps(gray)
    H, W = gray.shape
    kp = keypoints(net_height)
    names = [n for n in points if n in kp]
    Xc = np.array([kp[n] for n in names], float)
    uvc = np.array([points[n] for n in names], float)
    segs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in lines(net_height)]
    segs = [s for s in segs if not (s[0][0] == s[1][0] == 0 and s[0][2] != s[1][2])]   # antennas: no
    cx, cy = cam.cx, cam.cy
    # noise level of the ridge response, for a "real line here" threshold
    tr = hxx + hyy
    noise = float(np.median(np.abs(tr - np.median(tr)))) * 1.4826 + 1e-6

    def proj(p, X):
        f, k1, r, t = math.exp(p[0]), p[1], p[2:5], p[5:8]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
        pr, _ = cv2.projectPoints(np.asarray(X, float).reshape(-1, 3), r, t, K, np.array([k1, 0, 0, 0, 0], float))
        return pr.reshape(-1, 2)

    def depth_ok(p, X):
        R = cv2.Rodrigues(np.asarray(p[2:5], float))[0]
        return (X @ R.T + p[5:8])[:, 2] > 0.5

    def oriented(px, py, n_, sign):
        rxx, ryy, rxy = _sample(hxx, px, py), _sample(hyy, px, py), _sample(hxy, px, py)
        across = -sign * (n_[0] ** 2 * rxx + 2 * n_[0] * n_[1] * rxy + n_[1] ** 2 * ryy)
        along = np.abs(n_[1] ** 2 * rxx - 2 * n_[0] * n_[1] * rxy + n_[0] ** 2 * ryy)
        return np.where(across > 2 * along, across, 0.0)          # a line running this way

    def find(p, radius):
        """Each court line as a straight line in the picture: within +- radius px of where the camera
        puts it, the offset and small tilt whose length best follows a bright ridge (summed along
        the whole visible line, so wood grain, reflections and players don't win; a line that
        isn't clearly there is skipped). Returns (3D samples, target point, normal, line id)."""
        Xs, qs, ns, ids = [], [], [], []
        offs = np.arange(-radius, radius + 0.25, 0.5)
        tilts = np.tan(np.radians(np.arange(-2.0, 2.01, 0.25)))
        for li, (a, b) in enumerate(segs):
            X = a + np.linspace(0.02, 0.98, 200)[:, None] * (b - a)
            X = X[depth_ok(p, X)]
            if len(X) < 10:
                continue
            u = proj(p, X)
            m = (u[:, 0] > radius) & (u[:, 0] < W - radius) & (u[:, 1] > radius) & (u[:, 1] < H - radius)
            X, u = X[m], u[m]
            if len(X) < 10:
                continue
            plen = float(np.sum(np.linalg.norm(np.diff(u, axis=0), axis=1)))
            if plen < 60:
                continue
            keep = np.linspace(0, len(X) - 1, max(10, min(len(X), int(plen / step_px)))).astype(int)
            X, u = X[keep], u[keep]
            t = u[-1] - u[0]
            t = t / np.linalg.norm(t)
            n_ = np.array([-t[1], t[0]])
            c = u.mean(axis=0)
            sa = (u - c) @ t                                       # along-line position of each sample
            # painted lines are usually lighter than the floor, sometimes darker (a shared
            # basketball line): each court line takes whichever runs along it more clearly
            grids = []
            for sign in (1.0, -1.0):
                grid = np.zeros((len(offs), len(tilts)))
                for j, tl in enumerate(tilts):
                    for i, d in enumerate(offs):
                        q = u + (d + sa[:, None] * tl) * n_
                        sc = oriented(q[:, 0], q[:, 1], n_, sign)
                        grid[i, j] = np.nanmean(sc) if np.isfinite(sc).any() else 0.0
                grids.append(grid)
            grid = max(grids, key=lambda g_: g_.max())
            i, j = np.unravel_index(int(np.argmax(grid)), grid.shape)
            top = grid[i, j]
            if abs(offs[i]) > radius - 2:                          # best at the search edge: not this line
                continue
            if top < 3 * noise or top < 2.5 * np.median(grid):
                continue
            far = np.abs(offs - offs[i]) > 4
            if far.any() and grid[far].max() > 0.8 * top:            # two similar lines: ambiguous
                continue
            d, tl = offs[i], tilts[j]
            for X_, s_, u_ in zip(X, sa, u):
                Xs.append(X_); qs.append(c + s_ * t + (d + s_ * tl) * n_); ns.append(n_); ids.append(li)
        return np.array(Xs), np.array(qs), np.array(ns), np.array(ids)

    p = np.r_[math.log(cam.f), cam.k1, np.asarray(cam.rvec, float), np.asarray(cam.tvec, float)]
    stats = {}
    for radius in radii:
        Xs, qs, ns, ids = find(p, radius)
        if len(Xs):
            m = ~np.isin(ids, list(skip))                         # lines held out (for checking)
            Xs, qs, ns, ids = Xs[m], qs[m], ns[m], ids[m]
        if len(set(ids.tolist())) < 3:
            log(f"[calibrate] snapping: fewer than 3 lines found within {radius} px; keeping the clicks")
            break
        wt = 1.0 / np.sqrt(np.bincount(ids)[ids] / 10.0)            # each line counts about the same

        def resid(q):
            r_line = np.sum((proj(q, Xs) - qs) * ns, axis=1) * wt       # distance across the line
            r_click = ((proj(q, Xc) - uvc) / click_sigma).ravel()
            return np.r_[r_line, r_click, (q[1] / k1_sigma)]
        sol = least_squares(resid, p, loss="soft_l1", f_scale=1.5, method="trf")
        p = sol.x
        rl = np.sum((proj(p, Xs) - qs) * ns, axis=1)
        stats = {"lines": int(len(set(ids.tolist()))), "samples": int(len(Xs)), "radius_px": radius, "median_px": round(float(np.median(np.abs(rl))), 2),
                 "within_2px": round(float(np.mean(np.abs(rl) < 2)), 2)}
    new = Camera(cam.width, cam.height, math.exp(p[0]), list(p[2:5]), list(p[5:8]), float(p[1]), cx, cy)
    # where the final camera draws each line vs where it is found (within check_radius px)
    Xs, qs, ns, ids = find(p, check_radius)
    line_px = {}
    if len(Xs):
        off = np.sum((proj(p, Xs) - qs) * ns, axis=1)
        line_px = {int(li): round(float(np.median(np.abs(off[ids == li]))), 2) for li in sorted(set(ids.tolist()))}
    res = proj(p, Xc) - uvc
    info = dict(cam.info)
    info.update({"lines": stats, "line_px": line_px, "clicks_rms_px": round(float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), 2),
                 "residual_px": {n: round(float(np.hypot(*e)), 1) for n, e in zip(names, res)},
                 "snapped": bool(stats)})
    new.info = info
    return new
