"""core.camera: projection round trips and the calibration solve on synthetic cameras."""
import numpy as np
import pytest

from core.camera import Camera, solve
from core.court import keypoints, named_lines
from tools.sim_camera import H, W, random_camera


def visible(cam, X):
    uv = cam.project(X)
    return (cam.depth(X) > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)


def test_floor_point_inverts_project(camera, rng):
    X = np.c_[rng.uniform(-9, 9, 50), rng.uniform(-4.5, 4.5, 50), np.zeros(50)]
    X = X[visible(camera, X)]
    back = camera.floor_point(camera.project(X))
    assert np.abs(back - X).max() < 1e-6


def test_floor_point_at_height(camera):
    X = np.array([[-3.0, 1.0, 0.09], [2.0, -2.0, 0.09]])
    back = camera.floor_point(camera.project(X), 0.09)
    assert np.abs(back - X).max() < 1e-6


def test_json_round_trip(camera):
    again = Camera.from_json(camera.to_json())
    X = np.array([[0.0, 0.0, 2.43], [-9.0, 4.5, 0.0]])
    assert np.allclose(again.project(X), camera.project(X))


@pytest.mark.parametrize("seed", range(8))
def test_solve_recovers_camera_from_points(seed):
    rng = np.random.default_rng(seed)
    cam = random_camera(rng)
    kp = keypoints()
    names = list(kp)
    X = np.array([kp[n] for n in names])
    ok = visible(cam, X)
    uv = cam.project(X)
    pts = {n: tuple(uv[i] + rng.normal(0, 1.0, 2)) for i, n in enumerate(names) if ok[i]}
    if len(pts) < 6:
        pytest.skip("too few court points in view for this random camera")
    est = solve(pts, W, H)
    assert abs(est.f - cam.f) / cam.f < 0.03
    assert np.linalg.norm(est.position - cam.position) < 0.4
    assert est.info["rms_px"] < 2.5


def test_solve_from_line_clicks_and_two_points(rng):
    """Corners hard to see: a few points plus clicks anywhere along lines still calibrate."""
    cam = random_camera(np.random.default_rng(1))
    kp = keypoints()
    pts = {}
    for n in ("net top left", "net top right", "centre line left", "centre line right"):
        X = np.array([kp[n]])
        if visible(cam, X)[0]:
            pts[n] = tuple(cam.project(X)[0] + rng.normal(0, 1.0, 2))
    clicks = {}
    for name, (a, b) in named_lines().items():
        s = np.linspace(0.1, 0.9, 3)[:, None]
        X = np.asarray(a) + s * (np.asarray(b) - np.asarray(a))
        X = X[visible(cam, X)]
        if len(X) >= 2:
            clicks[name] = [tuple(p) for p in cam.project(X) + rng.normal(0, 1.0, (len(X), 2))]
    est = solve(pts, W, H, line_clicks=clicks)
    assert abs(est.f - cam.f) / cam.f < 0.05
    assert np.linalg.norm(est.position - cam.position) < 0.6
