"""core.scene3d.ball3d.fit_rally: recovers known 3D flights from noisy 2D through a known camera."""
import numpy as np

from core.scene3d.ball3d import BALL_D, DRAG_K, Link, Segment, fit_rally, traj
from evaluation.sim_rally import FPS, flight


def observe(cam, X, rng, noise=2.0):
    uv = cam.project(X) + rng.normal(0, noise, (len(X), 2))
    size = cam.f * BALL_D / cam.depth(X)
    return uv, size


def segment(cam, p0, p1, T, t0, rng):
    v0 = flight(p0, p1, T, DRAG_K)
    t = t0 + np.arange(int(T * FPS)) / FPS
    X = traj(p0, v0, t0, t, DRAG_K)
    uv, size = observe(cam, X, rng)
    return Segment(t, uv, size), X


def test_one_flight(camera, rng):
    seg, X = segment(camera, np.array([-6.0, 1.0, 1.0]), np.array([-1.0, -0.5, 2.5]), 1.2, 0.0, rng)
    fit = fit_rally([seg], [], camera)
    err = np.linalg.norm(fit.at(0, seg.t) - X, axis=1)
    assert np.median(err) < 0.15, f"median 3D error {np.median(err):.3f} m"


def test_pass_then_set_with_a_touch(camera, rng):
    a, b, c = np.array([-6.0, 1.0, 1.0]), np.array([-1.0, -0.5, 2.5]), np.array([-1.5, 3.5, 3.0])
    s1, X1 = segment(camera, a, b, 1.2, 0.0, rng)
    s2, X2 = segment(camera, b, c, 1.0, 1.2, rng)
    fit = fit_rally([s1, s2], [Link(0, 1.2)], camera)
    e = np.r_[np.linalg.norm(fit.at(0, s1.t) - X1, axis=1), np.linalg.norm(fit.at(1, s2.t) - X2, axis=1)]
    assert np.median(e) < 0.15
    assert np.linalg.norm(fit.at(0, 1.2) - fit.at(1, 1.2)) < 0.3          # the flights meet at the touch


def test_drag_flight_matches_a_fine_reference():
    """traj (RK4 at 1/60 s + Hermite) against a 1/4000 s integration, both time directions."""
    import core.scene3d.ball3d as B
    p0, v0, ts = np.array([-6.0, 1.0, 1.0]), np.array([14.0, -2.0, 7.0]), np.linspace(-0.5, 2.0, 200)
    fast = traj(p0, v0, 0.0, ts, DRAG_K)
    step, B.STEP = B.STEP, 1 / 4000
    try:
        ref = traj(p0, v0, 0.0, ts, DRAG_K)
    finally:
        B.STEP = step
    assert np.abs(fast - ref).max() < 1e-5                       # < 0.01 mm
    assert np.allclose(traj(p0, v0, 0.0, [0.0], DRAG_K), [p0])   # the reference time itself
