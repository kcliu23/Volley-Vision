#!/usr/bin/env python
"""Drive tools.review (the labeling window) with a script of keys, the way you would press them.

Runs the REAL review loop (tools.review.review) with OpenCV's window swapped for a fake one: each
scripted key is returned by cv2.waitKey, and the picture the tool shows after it is saved as a PNG.
Run it INSIDE a sandbox (scripts/sandbox.sh): the tool saves to ./data/verified/<clip>.json there.

    cd <sandbox> && PYTHONPATH=<repo> python <skill>/scripts/drive_review.py holdout/holdout2.mov \
        --keys "p P q" --evidence <dir>

Keys (space separated): one character ("p", "P", "m", "1", "a"), or
    esc  enter  space  wait:N (N idle polls; playback advances)  click:X,Y (left click at window pixels)
A final "q" (save and quit) is added if missing. Writes to --evidence:
    NNN_<key>.png       the window right after each key
    keys.txt            the keys as sent
    labels_before.json / labels_after.json, and diff.txt (contacts / not_rally / unseen changes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _labels(stem):
    p = Path("data/verified") / f"{stem}.json"
    return json.load(open(p)) if p.exists() else {}


def _diff(a, b):
    out = []
    for k in ("contacts", "not_rally"):
        A, B = a.get(k) or {}, b.get(k) or {}
        for f in sorted(set(A) | set(B), key=int):
            if A.get(f) != B.get(f):
                out.append(f"{k}[{f}]: {A.get(f)} -> {B.get(f)}")
    A, B = set(a.get("unseen") or []), set(b.get("unseen") or [])
    out += [f"unseen +{f}" for f in sorted(B - A)] + [f"unseen -{f}" for f in sorted(A - B)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--keys", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if not Path(".vv-repo").exists():
        sys.exit("run me inside a sandbox made by scripts/sandbox.sh (cwd has no .vv-repo)")
    repo = Path(".vv-repo").read_text().strip()
    if Path("data/verified").resolve() == (Path(repo) / "data/verified").resolve():
        sys.exit("data/verified here is the real one: refusing")
    ev = Path(a.evidence)
    ev.mkdir(parents=True, exist_ok=True)
    keys = a.keys.split()
    if not keys or keys[-1] != "q":
        keys.append("q")
    (ev / "keys.txt").write_text("\n".join(keys) + "\n")
    stem = Path(a.video).stem
    before = _labels(stem)
    json.dump(before, open(ev / "labels_before.json", "w"))

    import cv2
    state = {"queue": list(keys), "last": "start", "n": 0, "pending": None, "mouse": None, "idle": 0}

    def imshow(_win, img):
        if state["pending"] is not None:
            cv2.imwrite(str(ev / f"{state['n']:03d}_{state['pending']}.png"), img)
            state["pending"] = None
        elif state["n"] == 0:
            cv2.imwrite(str(ev / "000_start.png"), img)
            state["n"] = 1

    def waitKey(_ms=0):
        if state["idle"] > 0:
            state["idle"] -= 1
            if state["idle"] == 0:
                state["pending"] = state["wait_name"]          # snapshot once the wait is over
            return -1
        if not state["queue"]:
            return ord("q")
        k = state["queue"].pop(0)
        state["n"] += 1
        safe = {"esc": "esc", "enter": "enter", "space": "space"}.get(k, k if k.isalnum() else f"x{ord(k[0]):02x}")
        state["pending"] = safe.replace(":", "-").replace(",", "_")
        if k.startswith("wait:"):
            state["idle"], state["wait_name"], state["pending"] = int(k[5:]), state["pending"], None
            return -1
        if k.startswith("click:"):
            x, y = (int(v) for v in k[6:].split(","))
            if state["mouse"]:
                state["mouse"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
            return -1
        return {"esc": 27, "enter": 13, "space": 32}.get(k, ord(k[0]))

    cv2.namedWindow = lambda *x, **y: None
    cv2.setMouseCallback = lambda _w, cb, *x: state.__setitem__("mouse", cb)
    cv2.imshow, cv2.waitKey = imshow, waitKey
    cv2.destroyAllWindows = lambda *x: None

    sys.path.insert(0, repo)
    from tools.review import review
    review(a.video, a.test)

    after = _labels(stem)
    json.dump(after, open(ev / "labels_after.json", "w"))
    d = _diff(before, after)
    (ev / "diff.txt").write_text("\n".join(d) + ("\n" if d else "no label changes\n"))
    print(f"[drive_review] {len(keys)} keys sent, {len(d)} label changes -> {ev}")
    for line in d:
        print("  " + line)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
