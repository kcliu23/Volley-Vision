"""Headless replays of real review sessions on test74 (tools.review), checked against the saved file.

    pytest -m slow tests/test_review_replay.py      # needs videos/test74.mov + data/proposals/test74.json

Each scenario clicks and presses keys the way a person does, quits, and checks what is on disk:
the other balls marked, and that the game-ball labels were never touched.
"""
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import pytest

if not all(Path(p).exists() for p in ("videos/test74.mov", "data/proposals/test74.json", "data/verified/test74.json")):
    pytest.skip("needs the local test74 clip, proposals and labels", allow_module_level=True)

import tools.review as T  # noqa: E402
from core.common import paths  # noqa: E402

pytestmark = [pytest.mark.data, pytest.mark.slow]
STEM = "test74"
P = json.load(open(f"data/proposals/{STEM}.json"))
SCALE = min(T.MAX_W / P["width"], (T.MAX_H - T.TIMELINE_H - T.TOP_H) / P["height"], 1.0)
HELD = (2131, 572)                          # a spare ball held still by a bench player, frames ~0-205


def start_labels():
    V = json.load(open(f"data/verified/{STEM}.json"))
    V["others"] = {}
    return V


def screen(xy):
    return int(xy[0] * SCALE), int(xy[1] * SCALE) + T.TOP_H


def run(steps, verified, monkeypatch):
    """steps: ("goto", frame) | ("shift", xy) | ("alt", xy) | ("mouse", xy) | ("key", ch). Starts at
    frame 0 (test74 is fully checked). Returns (saved labels, log text)."""
    d = Path(tempfile.mkdtemp())
    json.dump(verified, open(d / f"{STEM}.json", "w"))
    monkeypatch.setattr(paths, "verified", lambda stem: d / f"{stem}.json")
    cb, seq, f = {}, [], 0
    for kind, v in steps:
        if kind == "goto":
            seq += [("key", "D")] * ((v - f) // 10) + [("key", "d")] * ((v - f) % 10)
            f = v
        else:
            seq.append((kind, v))
    it = iter(seq)

    def wait(ms):
        try:
            kind, v = next(it)
        except StopIteration:
            return ord("q")
        if kind in ("shift", "alt"):
            flags = cv2.EVENT_FLAG_SHIFTKEY | (cv2.EVENT_FLAG_ALTKEY if kind == "alt" else 0)
            cb["f"](cv2.EVENT_LBUTTONDOWN, *screen(v), flags, None)
            return 255
        if kind == "mouse":
            cb["f"](cv2.EVENT_MOUSEMOVE, *screen(v), 0, None)
            return 255
        return ord(v)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda w, fn: cb.setdefault("f", fn))
    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "waitKey", wait)
    T.review(f"videos/{STEM}.mov")
    out = json.load(open(d / f"{STEM}.json"))
    log = (d / ".logs" / f"{STEM}.log").read_text()
    shutil.rmtree(d)
    return out, log


def marked(out, xy=None, r=40):
    return sorted(int(g) for g, v in out["others"].items()
                  if v and (xy is None or any(abs(o[0] - xy[0]) + abs(o[1] - xy[1]) <= r for o in v)))


def game(out):
    return {k: out[k] for k in ("labels", "boxes", "src", "reviewed")}


def test_still_ball_one_click_then_reopen(monkeypatch):
    V = start_labels()
    out, log = run([("shift", HELD)], V, monkeypatch)
    fr = marked(out, HELD)
    assert fr[0] == 0 and 190 <= fr[-1] <= 215 and len(fr) == fr[-1] + 1, (fr[0], fr[-1], len(fr))
    assert game(out) == game(V)                                   # game-ball labels untouched
    assert "still ball" in log
    again, _ = run([], out, monkeypatch)                          # reopen, change nothing
    assert again["others"] == out["others"]


def test_moving_ball_keyframes_and_g(monkeypatch):
    V = start_labels()
    lab = {int(k): v for k, v in V["labels"].items()}
    keys = [g for g in (400, 420, 440) if isinstance(lab.get(g), list)]
    assert len(keys) == 3, "test74 has the game ball labelled at 400/420/440 (a moving ball to click)"
    steps = []
    for g in keys:
        steps += [("goto", g), ("shift", lab[g])]
    out, log = run(steps + [("key", "g")], V, monkeypatch)
    fr = marked(out)
    assert fr == list(range(400, 441)), fr[:5]
    mid = out["others"]["410"][0]
    want = [(lab[400][i] + lab[420][i]) / 2 for i in (0, 1)]
    assert abs(mid[0] - want[0]) < 2 and abs(mid[1] - want[1]) < 2
    assert game(out) == game(V)
    assert "between the keyframes" in log


def test_undone_keyframe_is_not_used_by_g(monkeypatch):
    V = start_labels()
    lab = {int(k): v for k, v in V["labels"].items()}
    out, _ = run([("goto", 400), ("shift", lab[400]), ("goto", 420), ("shift", lab[420]), ("key", "u"),
                  ("goto", 440), ("shift", lab[440]), ("key", "g")], V, monkeypatch)
    mid = out["others"]["420"][0]                                  # on the 400 -> 440 line, not at the undone click
    want = [(lab[400][i] + lab[440][i]) / 2 for i in (0, 1)]
    assert abs(mid[0] - want[0]) < 2 and abs(mid[1] - want[1]) < 2


def test_r_and_capital_r_and_cut(monkeypatch):
    V = start_labels()
    out, _ = run([("shift", HELD), ("goto", 100), ("mouse", HELD), ("key", "r"),
                  ("goto", 120), ("key", "R"), ("goto", 150), ("shift", HELD)], V, monkeypatch)
    fr = marked(out, HELD)
    assert 100 not in fr and 120 not in fr and 99 in fr and 101 in fr and 119 in fr and 121 in fr
    assert fr[-1] == 149                                           # cut from 150 on; everything before stays
    assert game(out) == game(V)


def test_option_click_cuts_backwards(monkeypatch):
    V = start_labels()
    out, _ = run([("shift", HELD), ("goto", 50), ("alt", HELD)], V, monkeypatch)
    fr = marked(out, HELD)
    assert fr[0] == 51 and 190 <= fr[-1] <= 215
