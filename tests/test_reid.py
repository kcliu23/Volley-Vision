"""core.people.reid.build_ids on synthetic tracker pieces (no video: colours are skipped)."""
import numpy as np

from core.people.reid import build_ids


class FlatCamera:
    """Pixels are court meters x 100 (x along the court, y across); enough for re-ID."""
    def floor_point(self, uv, height=0.0):
        uv = np.asarray(uv, float).reshape(-1, 2)
        return np.c_[uv / 100.0, np.full(len(uv), height)]


def person(x, y):
    kp = [[0.0, 0.0, 0.0]] * 17
    kp = [list(p) for p in kp]
    for i in (5, 6, 11, 12):
        kp[i] = [x * 100, y * 100 - 100, 0.9]
    kp[15] = [x * 100 - 5, y * 100, 0.9]
    kp[16] = [x * 100 + 5, y * 100, 0.9]
    return kp


def frames(pieces):
    """pieces: {tracker id: (first frame, last frame, x0, y0, x1, y1)} (moving in a straight line)"""
    out = {}
    for tid, (f0, f1, x0, y0, x1, y1) in pieces.items():
        for f in range(f0, f1 + 1):
            a = (f - f0) / max(f1 - f0, 1)
            x, y = x0 + a * (x1 - x0), y0 + a * (y1 - y0)
            out.setdefault(str(f), []).append({"id": tid, "box": [x * 100 - 30, y * 100 - 180, x * 100 + 30, y * 100],
                                               "kp": person(x, y)})
    return out


def test_joins_a_broken_track_and_keeps_sides_apart():
    pf = frames({1: (0, 60, -6.0, 1.0, -5.0, 1.0),       # near side, lost at 60 ...
                 2: (70, 140, -4.9, 1.1, -4.0, 1.0),     # ... found again nearby at 70: same player
                 3: (0, 140, 5.0, -1.0, 5.0, -1.0),      # far side, all clip
                 4: (0, 140, -6.0, -3.0, -6.0, -3.0)})   # near side, on screen with 1 and 2
    ids = build_ids(pf, "no-video.mov", FlatCamera(), fps=60.0)
    assert ids[1]["player"] == ids[2]["player"]
    assert ids[3]["player"] != ids[1]["player"] and ids[3]["side"] == 1 and ids[1]["side"] == -1
    assert ids[4]["player"] != ids[1]["player"]


def test_too_far_to_run_is_not_joined():
    pf = frames({1: (0, 60, -6.0, 1.0, -6.0, 1.0),
                 2: (65, 140, -1.0, -3.0, -1.0, -3.0)})  # 6+ m away 5 frames later
    ids = build_ids(pf, "no-video.mov", FlatCamera(), fps=60.0)
    assert ids[1]["player"] != ids[2]["player"]


def test_cache_reused_until_the_camera_changes(tmp_path, monkeypatch):
    import core.people.reid as reid
    from core.geometry.camera import Camera
    calls = []
    monkeypatch.setattr(reid, "build_ids", lambda *a: calls.append(1) or {7: {"player": 0, "side": 1, "on_court": True}})
    monkeypatch.setattr(Camera, "load", staticmethod(lambda p: None))
    cam, cache = tmp_path / "cam.json", tmp_path / "cache" / "reid.json"
    cam.write_text("a")
    first = reid.build_ids_cached({}, "pose1", "v.mov", cam, 60.0, cache)
    again = reid.build_ids_cached({}, "pose1", "v.mov", cam, 60.0, cache)
    assert first == again == {7: {"player": 0, "side": 1, "on_court": True}} and len(calls) == 1
    cam.write_text("b")                                   # recalibrated: guesses redone
    reid.build_ids_cached({}, "pose1", "v.mov", cam, 60.0, cache)
    reid.build_ids_cached({}, "pose2", "v.mov", cam, 60.0, cache)   # new pose run: redone
    assert len(calls) == 3
