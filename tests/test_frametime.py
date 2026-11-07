"""core.common.frametime: repeated + skipped frames of a screen recording -> true time."""
import numpy as np

from core.common.frametime import steps_from_changes


def changes(n, repeats, skips, rng):
    d = 1.0 + rng.normal(0, 0.05, n)
    d[list(repeats)] = 0.01                  # nothing changed: a repeated frame
    d[list(skips)] = 2.0                     # twice the usual change: a frame was dropped
    return d


def test_repeat_paired_with_skip(rng):
    d = changes(120, [30, 80], [33, 84], rng)
    s = steps_from_changes(d)
    assert s[30] == 0 and s[33] == 2 and s[80] == 0 and s[84] == 2
    assert np.cumsum(s)[-1] == 119           # time catches up: no frame lost overall


def test_plain_video_has_no_steps(rng):
    s = steps_from_changes(changes(200, [], [], rng))
    assert (s[1:] == 1).all()


def test_pause_without_skip_stays_a_pause(rng):
    s = steps_from_changes(changes(100, [50], [], rng))
    assert s[50] == 0 and (np.delete(s[1:], 49) == 1).all()
