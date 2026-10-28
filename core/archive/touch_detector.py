"""Detect ball touches (contacts) from a per-frame ball trajectory.

Idea: between touches the ball is in free flight, so each image axis is
well described by a parabola in time (gravity + mild perspective). A touch
is an impulse — the velocity jumps while the position stays continuous.

For every candidate frame i we fit, per axis, a "split" model on the
detections in a short window before and after i:

    p(t) = p0 + v_in * t  + c*t^2   for t < 0      (t = frame - i, in seconds)
    p(t) = p0 + v_out * t + c*t^2   for t > 0

i.e. shared position at the touch (continuity) and shared acceleration
(gravity doesn't change), but separate velocities. Δv = v_out - v_in is the
impulse. Pure flight gives Δv ≈ 0 — including at the apex of a vertical set,
which is why this is more robust than "vy flips sign".

We also fit a single parabola across the whole window; the drop in squared
error from single -> split ("gain") is how much evidence there is for a
touch at exactly frame i, and is what we peak-pick on.

Uses raw detections, not the Kalman output: the constant-velocity Kalman
lags behind and smears a sharp velocity change over several frames.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Touch:
    frame: int
    t: float                 # seconds
    x: float                 # px, fitted contact position
    y: float
    vx_in: float             # px/s, image coords (+y is down)
    vy_in: float
    vx_out: float
    vy_out: float
    speed_in: float          # px/s
    speed_out: float
    angle_deg: float         # direction change, 0..180
    impulse: float           # |v_out - v_in| in frame-heights/s
    rel_impulse: float       # |Δv| / max(speed_in, speed_out)
    gain: float              # split-vs-single fit evidence (normalised)
    in_gap: bool             # contact itself happened while ball undetected

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 3)
        return d


@dataclass
class TouchParams:
    win_s: float = 0.20         # max span of points used on each side of the candidate
    max_gap_s: float = 0.15     # tolerated missing stretch inside a window
    min_pts: int = 6            # per side
    min_impulse: float = 0.45   # frame-heights/s
    min_rel_impulse: float = 0.35
    min_gain: float = 20.0      # split fit must explain this much more (in noise units)
    min_sep_s: float = 0.05     # don't place a touch right on top of another
    max_speed: float = 4.0      # frame-heights/s; faster than a real hit at this distance -> bad fit
    noise_px: float = 3.0       # detection jitter floor
    jitter_frames: float = 0.5  # timing jitter (dropped/duplicated frames), in frames of motion


def clean_track(frames, xs, ys, fps):
    """Drop repeated frames (identical position as the previous frame — the
    video duplicated a frame, so its timestamp is wrong) and isolated outlier
    detections (a single-frame jump to some other object)."""
    f, x, y = (np.asarray(a, dtype=float) for a in (frames, xs, ys))
    if len(f) == 0:
        return f, x, y
    keep = np.ones(len(f), bool)
    same = (np.diff(x) == 0) & (np.diff(y) == 0)
    keep[1:][same] = False
    f, x, y = f[keep], x[keep], y[keep]

    keep = np.ones(len(f), bool)
    k = 4
    for i in range(len(f)):
        lo, hi = max(0, i - k), min(len(f), i + k + 1)
        nb = [j for j in range(lo, hi) if j != i and abs(f[j] - f[i]) <= 2 * k]
        if len(nb) < 4:
            continue
        t = f[nb] - f[i]
        A = np.stack([np.ones_like(t), t, t ** 2], 1)
        px = np.linalg.lstsq(A, x[nb], rcond=None)[0]
        py = np.linalg.lstsq(A, y[nb], rcond=None)[0]
        step = np.hypot(px[1], py[1])               # px per frame near i
        res = np.hypot(x[i] - px[0], y[i] - py[0])
        if res > max(40.0, 3 * step):
            keep[i] = False
    return f[keep], x[keep], y[keep]


def _fit_split(tb, pb, ta, pa):
    """Least squares for [p0, v_in, v_out, c] on one axis. Returns (params, sse)."""
    n_b, n_a = len(tb), len(ta)
    A = np.zeros((n_b + n_a, 4))
    A[:, 0] = 1.0
    A[:n_b, 1] = tb
    A[n_b:, 2] = ta
    A[:n_b, 3] = tb ** 2
    A[n_b:, 3] = ta ** 2
    y = np.concatenate([pb, pa])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ sol
    return sol, float(r @ r), r


def _fit_single(t, p):
    A = np.stack([np.ones_like(t), t, t ** 2], axis=1)
    sol, *_ = np.linalg.lstsq(A, p, rcond=None)
    r = p - A @ sol
    return float(r @ r)


def _candidate(frames, xs, ys, i, fps, frame_h, P, lo_bar, hi_bar):
    """Evaluate a touch at frame i using points strictly between the barriers."""
    w = int(round(P.win_s * fps))
    gap = int(round(P.max_gap_s * fps))
    ib = np.nonzero((frames < i) & (frames > lo_bar) & (frames >= i - w - gap))[0]
    ia = np.nonzero((frames > i) & (frames < hi_bar) & (frames <= i + w + gap))[0]
    # stop each side at the first big hole, walking outward from i
    def trim(idx, outward):
        if len(idx) == 0:
            return idx
        seq = idx[::-1] if outward < 0 else idx
        out = [seq[0]] if abs(frames[seq[0]] - i) <= gap else []
        for a, b in zip(seq, seq[1:]):
            if not out or abs(frames[b] - frames[a]) > gap:
                break
            out.append(b)
        out = [j for j in out if abs(frames[j] - i) <= w + gap]
        return np.array(sorted(out), dtype=int)
    ib, ia = trim(ib, -1), trim(ia, +1)
    if len(ib) < P.min_pts or len(ia) < P.min_pts:
        return None
    tb = (frames[ib] - i) / fps
    ta = (frames[ia] - i) / fps
    sx, sse_x, _ = _fit_split(tb, xs[ib], ta, xs[ia])
    sy, sse_y, _ = _fit_split(tb, ys[ib], ta, ys[ia])
    t_all = np.concatenate([tb, ta])
    sse1 = _fit_single(t_all, np.concatenate([xs[ib], xs[ia]])) + \
           _fit_single(t_all, np.concatenate([ys[ib], ys[ia]]))
    sse2 = sse_x + sse_y

    v_in = np.array([sx[1], sy[1]])
    v_out = np.array([sx[2], sy[2]])
    s_in, s_out = float(np.linalg.norm(v_in)), float(np.linalg.norm(v_out))
    if max(s_in, s_out) / frame_h > P.max_speed:
        return None
    n = len(t_all)
    jitter = P.jitter_frames * max(s_in, s_out) / fps
    noise2 = max(sse2 / max(2 * n - 8, 1), P.noise_px ** 2, jitter ** 2)
    gain = (sse1 - sse2) / noise2

    dv = v_out - v_in
    imp = float(np.linalg.norm(dv))
    cosang = float(v_in @ v_out) / max(s_in * s_out, 1e-6)
    return Touch(
        frame=int(i), t=i / fps, x=float(sx[0]), y=float(sy[0]),
        vx_in=float(v_in[0]), vy_in=float(v_in[1]),
        vx_out=float(v_out[0]), vy_out=float(v_out[1]),
        speed_in=s_in, speed_out=s_out,
        angle_deg=float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))),
        impulse=imp / frame_h, rel_impulse=imp / max(s_in, s_out, 1e-6),
        gain=float(gain), in_gap=not np.any(np.abs(frames - i) <= 1),
    )


def detect_touches(
    frames: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    fps: float,
    frame_h: int,
    params: TouchParams | None = None,
) -> list[Touch]:
    """frames/xs/ys: raw detections (pixel coords), sorted by frame.

    Greedy: take the strongest touch, then re-evaluate everything with that
    touch as a barrier (windows can't reach across it), repeat. This stops a
    real touch from also showing up a few frames later as an "echo", without a
    fixed suppression radius that would hide genuinely quick touch pairs
    (block -> dig).
    """
    P = params or TouchParams()
    f, x, y = clean_track(frames, xs, ys, fps)
    if len(f) < 2 * P.min_pts:
        return []
    sep = max(1, int(round(P.min_sep_s * fps)))
    kept: list[Touch] = []
    bars = [-np.inf, np.inf]
    while True:
        best = None
        for i in range(int(f[0]) + 1, int(f[-1])):
            if any(abs(i - k.frame) <= sep for k in kept):
                continue
            lo = max(b for b in bars if b < i)
            hi = min(b for b in bars if b > i)
            c = _candidate(f, x, y, i, fps, frame_h, P, lo, hi)
            if c is None or c.gain < P.min_gain:
                continue
            if c.impulse < P.min_impulse or c.rel_impulse < P.min_rel_impulse:
                continue
            if best is None or c.gain > best.gain:
                best = c
        if best is None:
            break
        kept.append(best)
        bars.append(best.frame)
    return sorted(kept, key=lambda c: c.frame)
