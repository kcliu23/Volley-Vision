"""Rally pass A (core.rally.pass_a): windows where the ball goes above the net tape."""
import numpy as np

from core.rally.pass_a import windows


class FlatCam:
    """The tape at picture y = 300 everywhere."""
    def project(self, X):
        X = np.asarray(X, float)
        return np.stack([500 + 100 * X[:, 1], np.full(len(X), 300.0)], axis=1)


def _pos(n, high_runs, fps=60):
    P = np.full((n, 2), [500.0, 600.0])                 # seen, low
    for a, b in high_runs:
        P[a:b, 1] = 100.0                                # above the tape
    return P


def test_rally_kept_with_padding():
    W = windows(_pos(3000, [(600, 700), (900, 1000)]), 60, FlatCam())
    assert len(W) == 1                                   # 200 frames apart: one rally
    a, b = W[0]
    assert a <= 600 - 150 and b >= 1000 + 100            # padded before / after


def test_short_blip_dropped():
    assert windows(_pos(3000, [(600, 615)]), 60, FlatCam()) == []     # 15 frames: thrown to the server


def test_far_apart_rallies_split():
    assert len(windows(_pos(6000, [(600, 700), (3000, 3100)]), 60, FlatCam())) == 2
