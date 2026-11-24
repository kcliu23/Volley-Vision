"""core.common.safe_io: saving hand-made data can't lose it."""
import json

import pytest

import core.common.safe_io as S


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


def test_backups_stay_out_of_the_label_folder(tmp_path, clock):
    """Tools list <dir>/*.json as clips: a backup must never show up there."""
    p = tmp_path / "clip.json"
    S.save_json(p, {"v": 1})
    S.backup(p)
    assert [q.name for q in tmp_path.glob("*.json")] == ["clip.json"]
    assert len(S.backups(p)) == 1


def test_old_backups_next_to_the_file_still_load(tmp_path):
    p = tmp_path / "clip.json"
    S.save_json(tmp_path / "clip.backup-20000101-000000.json", {"v": 0})
    p.write_text("{ broken")
    assert S.load_json(p, log=lambda m: None) == {"v": 0}


def test_guarded_save_never_overwrites_another_windows_changes(tmp_path, clock):
    p = tmp_path / "clip.json"
    S.save_json(p, {"v": "start"})
    a, b = S.Guarded(p, log=lambda m: None), S.Guarded(p, log=lambda m: None)   # two windows on one clip
    assert a.save({"v": "a1"}) == p and a.save({"v": "a2"}) == p                  # its own saves: fine
    went = b.save({"v": "b"})                                                    # b's copy is stale
    assert json.load(open(p)) == {"v": "a2"}                                     # a's work kept
    assert went.parent.name == ".conflicts" and json.load(open(went)) == {"v": "b"}
    assert b.save({"v": "b2"}) == went and json.load(open(went)) == {"v": "b2"}  # b keeps saving there
    assert a.save({"v": "a3"}) == p


def test_guarded_new_file_created_elsewhere_is_kept(tmp_path, clock):
    p = tmp_path / "clip.json"
    g = S.Guarded(p, log=lambda m: None)                  # opened with no file yet
    S.save_json(p, {"v": "other"})
    assert g.save({"v": "mine"}) != p and json.load(open(p)) == {"v": "other"}
