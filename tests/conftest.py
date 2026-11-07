"""Shared test setup. Tests run from the repo root (the tools use paths like data/ and output/)."""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


def need(*paths):
    """Skip the test when local data (gitignored clips, outputs) isn't on this machine."""
    missing = [p for p in paths if not (ROOT / p).exists()]
    if missing:
        pytest.skip(f"needs local data: {', '.join(map(str, missing))}")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def camera(rng):
    """A random phone-like camera behind the near end line (evaluation.sim_camera)."""
    from evaluation.sim_camera import random_camera
    return random_camera(rng)
