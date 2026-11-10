"""core.people.player_labels: your numbers per tracker piece, cuts included, and their side."""
import json

from core.common.safe_io import file_key
from core.people.player_labels import PlayerLabels


def test_numbers_sides_and_cuts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pose = tmp_path / "pose.json"
    pose.write_text("{}")
    (tmp_path / "data/player_ids").mkdir(parents=True)
    json.dump({"pose_key": file_key(pose), "pieces": {"8": 3, "8@115": 9, "5": "x"}, "cuts": {"8": [115]}},
              open(tmp_path / "data/player_ids/clip.json", "w"))
    L = PlayerLabels.load("clip", pose)
    assert L.number(8, 100) == 3 and L.side(8, 100) == -1       # before the cut: player 3, near side
    assert L.number(8, 200) == 9 and L.side(8, 200) == 1        # after: player 9, far side
    assert L.number(5, 0) == "x" and L.side(5, 0) is None       # not a player
    assert L.number(42, 0) is None


def test_labels_made_on_another_pose_run_are_not_used(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pose = tmp_path / "pose.json"
    pose.write_text("{}")
    (tmp_path / "data/player_ids").mkdir(parents=True)
    json.dump({"pose_key": "something-else", "pieces": {"1": 1}, "cuts": {}}, open(tmp_path / "data/player_ids/clip.json", "w"))
    assert PlayerLabels.load("clip", pose, log=lambda m: None) is None
