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


def test_labels_made_on_another_pose_run_are_carried_over_by_boxes(tmp_path, monkeypatch):
    """A new pose run (other tracker ids) keeps your numbers: each new box takes the number of the
    saved box it overlaps (test95: --redo pose left the 3D without numbers before this)."""
    monkeypatch.chdir(tmp_path)
    box_a, box_b = [100, 100, 160, 300], [400, 120, 460, 320]
    frames = {str(f): [{"box": box_a, "player": 3}, {"box": box_b, "player": 9}] for f in range(10)}
    pose = tmp_path / "pose.json"
    json.dump({"frames": {str(f): [{"id": 51, "box": box_a}, {"id": 77, "box": box_b}] for f in range(10)}},
              open(pose, "w"))
    (tmp_path / "data/player_ids").mkdir(parents=True)
    json.dump({"pose_key": "something-else", "pieces": {"1": 3, "2": 9}, "cuts": {}, "frames": frames},
              open(tmp_path / "data/player_ids/clip.json", "w"))
    L = PlayerLabels.load("clip", pose, log=lambda m: None)
    assert L.number(51, 5) == 3 and L.number(77, 5) == 9 and L.number(1, 5) is None
