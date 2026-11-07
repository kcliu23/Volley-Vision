"""Accuracy gate for the global rally fit (core.scene3d.rally3d) on synthetic rallies (evaluation.sim_rally).

Slow (about a minute per rally): run with  pytest -m slow . The numbers are well inside what v2
scores on the 20-rally benchmark (5.4 cm median, 100% touch recall), so a failure means a real
regression, not noise.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.sim_camera import random_camera
from evaluation.sim_rally import edge_and_box, run_v2, score, simulate

pytestmark = pytest.mark.slow
MESS = SimpleNamespace(noise_px=3.0, missing=0.10, outliers=0.02, block=0.3)


@pytest.mark.parametrize("seed", [3, 7])
def test_rally_fit_accuracy(seed):
    rng = np.random.default_rng(seed)
    cam = random_camera(rng)
    s = simulate(cam, rng, MESS)
    edge, box = edge_and_box(cam, s["uv"])
    out, touches = run_v2(cam, s, edge, box)
    err, cov, hit, n_true, good, n_found = score(s, out, touches)
    assert np.median(err) < 0.15, f"median 3D error {np.median(err):.3f} m"
    assert cov > 0.8
    assert hit >= n_true - 1, f"found {hit} of {n_true} touches"
    assert good >= n_found - 1, f"{n_found - good} touches that didn't happen"
