"""Spin lift in the flight model (core.scene3d.ball3d.traj / fit_rally)."""
import numpy as np

from core.scene3d.ball3d import DRAG_K, traj

P0, V0 = np.array([-6.0, 1.0, 1.0]), np.array([7.0, 0.5, 6.0])
T = np.arange(0, 1.4, 1 / 60)


def test_no_spin_is_unchanged():
    assert np.allclose(traj(P0, V0, 0.0, T, DRAG_K, 0.0), traj(P0, V0, 0.0, T, DRAG_K))


def test_backspin_floats_topspin_dives():
    apex = {s: traj(P0, V0, 0.0, T, DRAG_K, s)[:, 2].max() for s in (-0.3, 0.0, 0.3)}
    assert apex[0.3] > apex[0.0] > apex[-0.3]


def test_spin_lift_is_across_the_travel():
    """Lift is perpendicular to the velocity: at the start it changes direction, not speed."""
    dt = 1e-4
    a0 = (traj(P0, V0, 0.0, [dt], 0.0, 0.0)[0] - P0 - V0 * dt) / (0.5 * dt * dt)
    a1 = (traj(P0, V0, 0.0, [dt], 0.0, 0.3)[0] - P0 - V0 * dt) / (0.5 * dt * dt)
    lift = a1 - a0
    assert abs(np.dot(lift, V0)) / (np.linalg.norm(lift) * np.linalg.norm(V0)) < 1e-2
    assert np.isclose(np.linalg.norm(lift), 0.3 * np.linalg.norm(V0), rtol=1e-2)
