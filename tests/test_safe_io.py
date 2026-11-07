"""core.safe_io: saving hand-made data can't lose it."""
import json

import pytest

import core.safe_io as S


@pytest.fixture
def clock(monkeypatch):
    """Backups are named by the second: tick a fake clock instead of sleeping."""
    t = [0]

    def strftime(fmt):
        t[0] += 1
        return f"20260101-{t[0]:06d}"
    monkeypatch.setattr(S.time, "strftime", strftime)


def test_save_is_complete_and_leaves_no_temp(tmp_path):
    p = tmp_path / "clip.json"
    S.save_json(p, {"v": 1})
    assert json.load(open(p)) == {"v": 1} and not list(tmp_path.glob("*.tmp"))


def test_damaged_file_loads_the_backup(tmp_path, clock):
    p = tmp_path / "clip.json"
    S.save_json(p, {"v": 1})
    S.backup(p)
    p.write_text("{ broken")
    assert S.load_json(p, log=lambda m: None) == {"v": 1}


def test_damaged_file_never_becomes_a_backup(tmp_path, clock):
    p = tmp_path / "clip.json"
    S.save_json(p, {"v": 1})
    S.backup(p)
    p.write_text("{ broken")
    assert S.backup(p) is None and len(S.backups(p)) == 1


def test_keeps_the_newest_five(tmp_path, clock):
    p = tmp_path / "clip.json"
    for i in range(7):
        S.save_json(p, {"v": i})
        S.backup(p, keep=5)
    kept = [json.load(open(b))["v"] for b in S.backups(p)]
    assert kept == [2, 3, 4, 5, 6]


def test_no_backup_and_damaged_raises(tmp_path):
    p = tmp_path / "clip.json"
    p.write_text("{ broken")
    with pytest.raises(json.JSONDecodeError):
        S.load_json(p, log=lambda m: None)
