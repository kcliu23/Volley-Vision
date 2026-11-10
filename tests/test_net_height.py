"""The calibration tells a women's net from a men's one by itself (core.geometry.camera.check_net_height)."""
import json
from pathlib import Path

import pytest

from core.geometry.camera import check_net_height

P = Path("data/cameras/test20.json")              # a women's net (KC, 2026-11-10)
pytestmark = pytest.mark.data


@pytest.mark.skipif(not P.exists(), reason="needs data/cameras/test20.json")
def test_women_net_found_from_a_mens_guess():
    d = json.load(open(P))
    h, rms = check_net_height(d["points"], d["width"], d["height"], 2.43, line_clicks=d["line_clicks"])
    assert h == 2.24 and rms[2.24] < 0.5 * rms[2.43]


@pytest.mark.skipif(not P.exists(), reason="needs data/cameras/test20.json")
def test_right_height_is_kept():
    d = json.load(open(P))
    h, _ = check_net_height(d["points"], d["width"], d["height"], 2.24, line_clicks=d["line_clicks"])
    assert h == 2.24


def test_no_net_clicks_no_change():
    assert check_net_height({}, 100, 100, 2.43, line_clicks={}) == (2.43, {})
