"""Touches from your marks, and the one-camera toucher check (core.scene3d.rally3d)."""
import numpy as np

from core.scene3d.rally3d import Obs, _anchors, _fixed_bounds


class _Cam:
    f = 1000.0

    def depth(self, X):                       # camera at the origin looking along +y
        return np.asarray(X, float)[:, 1]


def _obs(n=20, size=21.0):
    fr = np.arange(n)
    return Obs(fr / 60.0, np.zeros((n, 2)), np.full(n, size), np.ones(n), fr, np.zeros(n, bool))


def test_marks_cut_between_frames():
    o = _obs()
    o.fixed = [5.5, 12]
    assert _fixed_bounds(o, print) == [0, 6, 13, 20]


def test_mark_too_close_to_an_end_is_skipped():
    o = _obs()
    o.fixed = [0.5]
    assert _fixed_bounds(o, lambda m: None) == [0, 20]


def test_toucher_at_another_depth_is_ignored():
    o = _obs(size=21.0)                       # ball 1000 * 0.21 / 21 = 10 m from the camera
    o.cam = _Cam()
    near = {"id": 1, "side": -1, "spot": (-3.0, 10.0), "foot": (-3.0, 10.0), "player": 2}
    behind = {"id": 2, "side": 1, "spot": (4.0, 19.0), "foot": (4.0, 19.0), "player": 8}
    o.touchers = lambda f, uv: [near, behind]
    a = _anchors(o, [0, 10, 20])
    assert a and all(x["side"] == -1 for x in a)       # the player behind the ball is not the toucher


def test_both_sides_left_gives_no_anchor():
    o = _obs()
    o.cam = _Cam()
    a1 = {"id": 1, "side": -1, "spot": None, "foot": None, "player": 2}
    a2 = {"id": 2, "side": 1, "spot": None, "foot": None, "player": 8}
    o.touchers = lambda f, uv: [a1, a2]
    assert _anchors(o, [0, 10, 20]) == []


def test_facing_alone_does_not_set_a_side():
    o = _obs()
    o.cam = _Cam()
    o.touchers = lambda f, uv: [{"id": 1, "side": 1, "spot": None, "foot": None, "player": None}]
    assert _anchors(o, [0, 10, 20]) == []
