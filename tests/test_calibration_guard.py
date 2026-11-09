"""A ball finder is never scored with a calibration fitted on nothing (the wasb_ft3 scoreboard
of 2026-11-08 ran with n=0 and made 0 YOLO crops, so the new model looked much worse than it was)."""
import json

import pytest

from core.ball.confidence import Calibration
from tools.calibrate import ensure_cached

M = {"coef": [2.0], "intercept": -1.0, "n": 100, "pos": 30}


def test_empty_calibration_is_refused():
    cal = Calibration({"tracknet": M, "wasb_ft3": {"coef": [1.0], "intercept": -2.0, "n": 0}})
    with pytest.raises(ValueError, match="0 examples"):
        cal.for_finder("wasb_ft3")


def test_fitted_calibration_is_used():
    cal = Calibration({"tracknet": {"coef": [1.0], "intercept": 0.0, "n": 5}, "wasb_ft3": M})
    assert cal.for_finder("wasb_ft3").models["tracknet"] == M


def test_scoreboard_candidates_are_reused_for_calibration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "output/dev/scoreboard/finder-first_best4_wasb_ft3/test1/cache/tracknet_wasb_ft3.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps({"frames": [[[10, 20, 30, 30, 0.9]]]}))
    ensure_cached("models/wasb/wasb_ft3.pth.tar", ["test1"])
    assert json.load(open(tmp_path / "output/test1/cache/tracknet_wasb_ft3.json"))["frames"][0][0][4] == 0.9


def test_crop_cache_is_redone_when_the_calibration_changes(tmp_path):
    """The 0-crop results made with the empty calibration must not be reused after a proper fit."""
    from core.ball.finder_first import crop_finds
    from core.common import paths
    cands = [[[100.0, 100.0, 30, 30, 0.9]] for _ in range(5)]
    loose = paths.cache_dir("clip", tmp_path) / "crops_wasb_ft3.json"
    loose.parent.mkdir(parents=True)
    loose.write_text(json.dumps({"found": {str(f): {"c": [100, 100], "cands": [[101, 99, 30, 30, 0.8]]}
                                           for f in range(5)}}))
    bad = crop_finds("clip", "none.mov", cands, 30.0, "wasb_ft3", None, lambda pk: 0.0, 2, 0.1, root=tmp_path)
    good = crop_finds("clip", "none.mov", cands, 30.0, "wasb_ft3", None, lambda pk: 0.9, 2, 0.1, root=tmp_path)
    assert bad["crops"] == 0 and good["crops"] == 5 and len(good["found"]) == 5
