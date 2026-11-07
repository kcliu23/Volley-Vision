"""core.reid.build_ids on synthetic tracker pieces (no video: colours are skipped)."""
import numpy as np

from core.reid import build_ids


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
