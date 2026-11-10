"""Headless replay of KC's holdout1 loop in tools.label_players (2026-11-10).

    pytest -m slow tests/test_label_holdout1.py     # needs holdout/holdout1.mov + output/holdout1/pose.json

Tracker track 45 is a man by the bench at frame 774 and player 1 by frame 840 (the track drifts from
one to the other). Marking 774 x and 840 1, then going back and forth, must keep both.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import pytest

import tools.label_players as T

CLIP, VIDEO = "holdout1", "holdout/holdout1.mov"
if not all(Path(p).exists() for p in (VIDEO, f"output/{CLIP}/pose.json")):
    pytest.skip("needs the local holdout1 clip and its pose.json", allow_module_level=True)
pytestmark = [pytest.mark.data, pytest.mark.slow]
POSE = json.load(open(f"output/{CLIP}/pose.json"))["frames"]
W, H = 2622, 1206
SCALE = min(T.MAX_W / W, (T.MAX_H - T.TOP_H) / H, 1.0)


def point(f, tid):
    b = [p["box"] for p in POSE[str(f)] if p["id"] == tid][0]
    return int((b[0] + b[2]) / 2 * SCALE), int((b[1] + b[3]) / 2 * SCALE) + T.TOP_H


def run(steps):
    d = Path(tempfile.mkdtemp())
    T.OUT_DIR = d
    cb = {}
    cv2.namedWindow = lambda *a, **k: None
    cv2.setMouseCallback = lambda w, f: cb.setdefault("f", f)
    cv2.imshow = lambda *a: None
    cv2.destroyAllWindows = lambda: None
    seq, at = [], 0
    for s in steps:
        if s[0] == "goto":
            seq += [("key", ord("." if s[1] > at else ","))] * abs(s[1] - at)
            at = s[1]
        else:
            seq.append(s)
    it = iter(seq)

    def wait(ms):
        try:
            kind, v = next(it)
        except StopIteration:
            return ord("q")
        if kind == "click":
            cb["f"](cv2.EVENT_LBUTTONDOWN, v[0], v[1], 0, None)
            return -1
        return v
    cv2.waitKeyEx = wait
    sys.argv = ["x", VIDEO]
    T.main()
    out = json.load(open(d / f"{CLIP}.json"))
    shutil.rmtree(d)
    return out


def value(L, tid, f):
    cuts = {int(k): v for k, v in L["cuts"].items()}
    return L["pieces"].get(T.piece_of(cuts, tid, f))


def test_bench_guy_then_player_back_and_forth():
    out = run([("goto", 774), ("click", point(774, 45)), ("key", ord("x")),
               ("goto", 840), ("click", point(840, 45)), ("key", ord("1")),
               ("goto", 774), ("click", point(774, 45)), ("key", ord("x")),
               ("goto", 840), ("click", point(840, 45)), ("key", ord("1"))])
    assert value(out, 45, 774) == "x" and value(out, 45, 700) == "x"
    assert value(out, 45, 840) == 1 and value(out, 45, 943) == 1
    assert out["cuts"]["45"] == [840]
    # the saved per-frame boxes agree (they're what survives a new pose run)
    box = [p["box"] for p in POSE["774"] if p["id"] == 45][0]
    assert {"box": box, "player": "x"} in out["frames"]["774"]


def test_r_makes_the_whole_track_one_name():
    out = run([("goto", 774), ("click", point(774, 45)), ("key", ord("x")),
               ("goto", 840), ("click", point(840, 45)), ("key", ord("1")), ("key", ord("r"))])
    assert value(out, 45, 774) == 1 and "45" not in out["cuts"]
