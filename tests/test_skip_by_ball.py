"""Catch-up skips placed where the ball's double step is (core.common.frametime)."""
import numpy as np

from core.common.frametime import place_skips_by_ball


def _clip(true_steps):
    """A ball moving 10 px per real frame; true_steps: real frames between shown frames."""
    t = np.r_[0, np.cumsum(true_steps)].astype(float)
    uv = np.c_[100 + 10 * t, np.full(len(t), 300.0)]
    return t, uv, np.ones(len(t))


def test_skip_moved_to_the_balls_double_step():
    true = [1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1]          # repeat at 3->4, real skip at 5->6
    t, uv, w = _clip(true)
    picture = t.copy()
    picture[6:9] -= 1                                        # the picture check put it at 8->9
    got = place_skips_by_ball(picture, uv, w)
    assert np.allclose(got, t)


def test_right_skip_left_alone():
    t, uv, w = _clip([1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1])
    assert np.allclose(place_skips_by_ball(t, uv, w), t)


def test_slow_ball_says_nothing():
    t, uv, w = _clip([1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1])
    uv = np.c_[100 + 1 * t, np.full(len(t), 300.0)]          # 1 px per frame: no evidence
    picture = t.copy(); picture[6:9] -= 1
    assert np.allclose(place_skips_by_ball(picture, uv, w), picture)
