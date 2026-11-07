"""Headless replays of real labeling sessions for tools.label_players (no window needed).

    python -m tests.test_label_players            # needs videos/test74.mov + output/test74/pose.json

Each scenario feeds the tool clicks and keys the way a person does, then checks the saved file.
Run it after any change to the labeling tool.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.label_players as T  # noqa: E402

CLIP = "test74"
VIDEO = f"videos/{CLIP}.mov"
POSE = json.load(open(f"output/{CLIP}/pose.json"))["frames"]
W, H = 2622, 1206
SCALE = min(T.MAX_W / W, (T.MAX_H - T.TOP_H) / H, 1.0)


def point(f, tid):
    b = [p["box"] for p in POSE[str(f)] if p["id"] == tid][0]
    return int((b[0] + b[2]) / 2 * SCALE), int((b[1] + b[3]) / 2 * SCALE) + T.TOP_H


def run(steps, start_labels=None):
    """steps: ("click", (x, y)) | ("key", code) | ("goto", frame) | ("check", fn(labels_dict))."""
    d = Path(tempfile.mkdtemp())
    if start_labels:
        json.dump(start_labels, open(d / f"{CLIP}.json", "w"))
    T.OUT_DIR = d
    cb, shown = {}, []
    cv2.namedWindow = lambda *a, **k: None
    cv2.setMouseCallback = lambda w, f: cb.setdefault("f", f)
    cv2.imshow = lambda w, img: shown.append(img)
    cv2.destroyAllWindows = lambda: None
    seq = []
    for s in steps:
        if s[0] == "goto":                                  # step there frame by frame from 0 (we start at 0)
            seq += [("key", ord("."))] * s[1]
        else:
            seq.append(s)
    it = iter(seq)
    checks = []

    def wait(ms):
        while True:
            try:
                kind, v = next(it)
            except StopIteration:
                return ord("q")
            if kind == "check":
                checks.append(v(json.load(open(d / f"{CLIP}.json")) if (d / f"{CLIP}.json").exists() else {}))
                continue
            if kind == "click":
                cb["f"](cv2.EVENT_LBUTTONDOWN, v[0], v[1], 0, None)
                return -1
            return v
    cv2.waitKeyEx = wait
    sys.argv = ["x", VIDEO]
    T.main()
    out = json.load(open(d / f"{CLIP}.json")) if (d / f"{CLIP}.json").exists() else {}
    shutil.rmtree(d)
    return out, checks


def scenario_swap():
    """KC's loop on test74: tracker piece 8 is one player at frame 0 and another from ~112 on.
    Naming it 7 at frame 0, then (where 7 now clashes) 5 at frame 115, must cut it, not flip it."""
    base = json.load(open(f"data/player_ids/{CLIP}.json"))
    base["pieces"]["9"] = 7                                   # the real 7 at frame 115 (so 7 clashes there)
    out, _ = run([("click", point(0, 8)), ("key", ord("7")), ("goto", 115), ("click", point(115, 8)),
                  ("key", ord("5"))], base)
    cut = out["cuts"].get("8", [])
    ok = len(cut) == 1 and 1 <= cut[0] <= 115 and out["pieces"].get("8") == 7 and out["pieces"].get(f"8@{cut[0]}") == 5
    return ok, f"cuts {cut}, names {[(k, v) for k, v in out['pieces'].items() if k.split('@')[0] == '8']}"


def scenario_plain_fix():
    """A plain correction (no clash where you fix it) just renames: no cut."""
    out, _ = run([("click", point(0, 8)), ("key", ord("3")), ("goto", 300), ("key", ord("]"))])
    first = out["pieces"].get("8")
    out2, _ = run([("click", point(0, 8)), ("key", ord("3")), ("key", ord("4"))])
    ok = first == 3 and out2["pieces"].get("8") == 4 and not out2["cuts"].get("8")
    return ok, f"first {first}, after fix {out2['pieces'].get('8')}, cuts {out2['cuts'].get('8')}"


def scenario_undo():
    """z after a naming restores the old number and clears the selection (a number key then asks
    for a click instead of naming a stale piece)."""
    out, _ = run([("click", point(0, 8)), ("key", ord("3")), ("key", ord("z")), ("key", ord("4"))])
    ok = "8" not in out.get("pieces", {})
    return ok, f"piece 8 after undo + key: {out.get('pieces', {}).get('8')}"


def scenario_autosave():
    """Every change is on disk at once (no Enter needed)."""
    out, checks = run([("click", point(0, 8)), ("key", ord("3")), ("check", lambda L: L.get("pieces", {}).get("8"))])
    return checks == [3], f"on disk right after the key: {checks}"


def scenario_damaged_file():
    """A damaged labels file loads the latest backup instead of crashing."""
    d = Path("data/player_ids")
    good = json.load(open(d / f"{CLIP}.json"))
    tmpd = Path(tempfile.mkdtemp())
    (tmpd / f"{CLIP}.json").write_text("{ broken")
    json.dump(good, open(tmpd / f"{CLIP}.backup-20000101-000000.json", "w"))
    T.OUT_DIR = tmpd
    it = iter([ord("q")])
    cv2.namedWindow = lambda *a, **k: None
    cv2.setMouseCallback = lambda w, f: None
    cv2.imshow = lambda *a: None
    cv2.waitKeyEx = lambda ms: next(it, ord("q"))
    sys.argv = ["x", VIDEO]
    try:
        T.main()
        ok, msg = True, "loaded the backup"
    except Exception as e:                                   # noqa: BLE001
        ok, msg = False, f"crashed: {e}"
    shutil.rmtree(tmpd)
    return ok, msg


def scenario_pose_rerun():
    """A new pose run renumbers the tracker: the labels follow the people (by box), not the ids."""
    base = json.load(open(f"data/player_ids/{CLIP}.json"))
    base["pose_key"] = "an-older-pose-run"
    shift = {f: [{**p, "id": p["id"] + 1000 if p["id"] >= 0 else -1} for p in ps] for f, ps in POSE.items()}
    names, cuts, _, report = T.remap_by_boxes(base["frames"], shift)
    agree = tot = 0
    for f, row in base["frames"].items():
        for r in row:
            m = [p for p in shift[f] if p["box"] == r["box"]]
            if not m:
                continue
            tid = m[0]["id"]
            c = [s for s in cuts.get(tid, []) if s <= int(f)]
            tot += 1
            agree += names.get(f"{tid}@{max(c)}" if c else str(tid)) == r["player"]
    return tot > 0 and agree / tot > 0.98, f"{agree}/{tot} saved boxes keep their number ({report})"


if __name__ == "__main__":
    fails = 0
    for name, fn in [("swap", scenario_swap), ("plain fix", scenario_plain_fix), ("undo", scenario_undo),
                     ("autosave", scenario_autosave), ("damaged file", scenario_damaged_file), ("pose re-run", scenario_pose_rerun)]:
        ok, msg = fn()
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:12s} {msg}")
    sys.exit(1 if fails else 0)
