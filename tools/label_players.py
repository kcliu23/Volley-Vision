"""Label who is who: give every player on court a number for the whole clip (ground truth for re-ID).

    python -m tools.pose videos/test76.mov             # first, once per clip (person boxes + poses)
    python -m tools.label_players videos/test76.mov

Numbers 1-6 = the team on the NEAR side (the camera's side of the net), 7-12 = the FAR side, so a
number also says the side. You don't label boxes frame by frame: the tracker has already followed
each person for a while (a "piece"), and one key press names the whole piece. A grey box with
"?7" means "probably 7" (the re-ID guess, or the same piece-group as one you named): press a to
take all the guesses in this frame, then fix the wrong ones.

  click a box, then a key:
    1-9, 0, -, =   player 1..9, 10, 11, 12 (1-6 near side, 7-12 far side) - for the whole piece
    x              not a player (bench, referee, spectator, someone walking by)
    u              un-name this piece
    k              the tracker swapped people here: cut the piece at this frame (then name each part)
  a                take every "?n" guess shown in this frame
  n / Tab          jump to the next frame with an un-named person on court
  -> / ]   <- / [  0.5 s forward / back          . ,  one frame forward / back
  Enter            save (data/player_ids/<clip>.json)      q / Esc  quit (asks to save if changed)

Aim: everyone on court named in every frame (the top bar shows how much is done). A clip of
10-20 s takes about 3-5 minutes.

Saved: which piece is which player (tied to output/<clip>/pose.json), and the boxes per frame
with their player number (so the labels survive a new pose run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402

OUT_DIR = Path("data/player_ids")
WIN = "VolleyVision - who is who"
MAX_W, MAX_H, TOP_H = 1500, 900, 70
KEYS = {ord(str(i)): i for i in range(1, 10)} | {ord("0"): 10, ord("-"): 11, ord("="): 12}
LEFT_KEYS = {63234, 65361, 2424832}
RIGHT_KEYS = {63235, 65363, 2555904}
NEAR, FAR = (0, 200, 255), (255, 140, 0)          # BGR: near side orange-yellow, far side blue


def frame_cache(video: str, stem: str, scale: float):
    """Every frame, decoded in order (seeking in phone video can land on the wrong frame), saved
    small as JPEG once: output/<clip>/cache/label_frames/."""
    import cv2
    d = paths.cache_dir(stem) / "label_frames"
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    done = d.exists() and len(list(d.glob("*.jpg"))) >= n - 2
    if not done:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[label] {stem}: preparing {n} frames (once) ...", flush=True)
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if scale != 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / f"{f:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            f += 1
        n = f
    return d, n


def main():
    import cv2
    from core.camera import Camera
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    stem = Path(a.video).stem
    pj = paths.clip_dir(stem, a.output) / "pose.json"
    if not pj.exists():
        sys.exit(f"[label] {stem}: no poses yet. Run first:\n  python -m tools.pose {a.video}")
    pose = json.load(open(pj))["frames"]
    cam_p = Path("data/cameras") / f"{stem}.json"
    W, H = int(cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FRAME_WIDTH)), int(cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(MAX_W / W, (MAX_H - TOP_H) / H, 1.0)
    fdir, n = frame_cache(a.video, stem, scale)

    # pieces: tracker id, optionally cut into parts at frames ("k"): piece key = "id" or "id@f0"
    out = OUT_DIR / f"{stem}.json"
    names: dict = {}                                        # piece key -> 1..12 or "x"
    cuts: dict = {}                                         # tracker id -> [frames where a new part starts]
    if out.exists():
        old = json.load(open(out))
        names = {k: v for k, v in old.get("pieces", {}).items()}
        cuts = {int(k): v for k, v in old.get("cuts", {}).items()}
        print(f"[label] {stem}: loaded {len(names)} named pieces from {out}")

    def piece(tid: int, f: int) -> str:
        c = [s for s in cuts.get(tid, []) if s <= f]
        return f"{tid}@{max(c)}" if c else str(tid)

    # re-ID guesses (core.reid): pieces the joiner put together
    group = {}
    on_court = {}
    if cam_p.exists():
        from core.reid import build_ids
        cam = Camera.load(cam_p)
        fps = cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FPS) or 60.0
        print(f"[label] {stem}: re-ID guesses ...", flush=True)
        ids = build_ids(pose, a.video, cam, fps)
        group = {t: v["player"] for t, v in ids.items()}
        on_court = {t: v["on_court"] for t, v in ids.items()}

    def guess(key: str, tid: int):
        """Number guessed for an un-named piece: a named piece in the same re-ID group."""
        g = group.get(tid)
        if g is None:
            return None
        for k, v in names.items():
            t = int(k.split("@")[0])
            if group.get(t) == g and v != "x":
                return v
        return None

    def todo(f: int) -> list:
        return [p for p in pose.get(str(f), []) if p["id"] >= 0 and piece(p["id"], f) not in names
                and on_court.get(p["id"], True)]

    def progress():
        tot = done = 0
        for f in range(n):
            for p in pose.get(str(f), []):
                if p["id"] < 0 or not on_court.get(p["id"], True):
                    continue
                tot += 1
                done += piece(p["id"], f) in names
        return done / max(tot, 1)

    state = {"click": None, "sel": None}
    cur, changed, prog, prog_t = 0, False, progress(), time.time()

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    while True:
        img = cv2.imread(str(fdir / f"{cur:05d}.jpg"))
        if img is None:
            img = np.zeros((int(H * scale), int(W * scale), 3), np.uint8)
        canvas = np.zeros((img.shape[0] + TOP_H, img.shape[1], 3), np.uint8)
        canvas[TOP_H:] = img
        people = [p for p in pose.get(str(cur), []) if p["id"] >= 0]
        for p in people:
            key = piece(p["id"], cur)
            x1, y1, x2, y2 = [int(v * scale) for v in p["box"]]
            y1, y2 = y1 + TOP_H, y2 + TOP_H
            v = names.get(key)
            if v == "x":
                col, label, th = (90, 90, 90), "x", 1
            elif v is not None:
                col, label, th = (NEAR if v <= 6 else FAR), str(v), 2
            else:
                gs = guess(key, p["id"])
                col, label, th = (170, 170, 170), (f"?{gs}" if gs else "?"), 1
                if not on_court.get(p["id"], True):
                    col, label = (70, 70, 70), "off?"
            if state["sel"] == (key, p["id"]):
                cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 0), 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, th)
            cv2.putText(canvas, label, (x1, max(TOP_H + 14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        if time.time() - prog_t > 1.0:
            prog, prog_t = progress(), time.time()
        left = len(todo(cur))
        top = (f"frame {cur}/{n - 1}   named: {prog:.0%} of on-court person-frames   "
               f"{left} to name here   1-6 near, 7-12 far   x not a player   k cut   a take guesses   n next   Enter save")
        cv2.putText(canvas, top, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        sel = state["sel"]
        cv2.putText(canvas, f"selected: {sel[0] if sel else '-'}  (click a box, then press its number)",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)

        raw = cv2.waitKeyEx(30)
        key = raw & 0xFF if raw != -1 else 255
        if state["click"] is not None:
            cx, cy = state["click"]
            state["click"] = None
            best = None
            for p in people:                                # smallest box under the click
                x1, y1, x2, y2 = [v * scale for v in p["box"]]
                if x1 <= cx <= x2 and y1 + TOP_H <= cy <= y2 + TOP_H:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best[0]:
                        best = (area, p)
            state["sel"] = (piece(best[1]["id"], cur), best[1]["id"]) if best else None
        if raw == -1:
            continue
        if raw in LEFT_KEYS or key == ord("["):
            cur = max(0, cur - 30)
        elif raw in RIGHT_KEYS or key == ord("]"):
            cur = min(n - 1, cur + 30)
        elif key == ord(","):
            cur = max(0, cur - 1)
        elif key == ord("."):
            cur = min(n - 1, cur + 1)
        elif key in (ord("n"), 9):
            nxt = next((g for g in range(cur + 1, n) if todo(g)), None) or next((g for g in range(0, cur) if todo(g)), None)
            if nxt is None:
                print("[label] every on-court person is named")
            else:
                cur = nxt
        elif key in KEYS and state["sel"]:
            names[state["sel"][0]] = KEYS[key]
            changed = True
        elif key == ord("x") and state["sel"]:
            names[state["sel"][0]] = "x"
            changed = True
        elif key == ord("u") and state["sel"]:
            names.pop(state["sel"][0], None)
            changed = True
        elif key == ord("k") and state["sel"]:
            k0, tid = state["sel"]
            if cur not in cuts.setdefault(tid, []):
                cuts[tid] = sorted(cuts[tid] + [cur])
                old_name = names.get(k0)                    # the earlier part keeps its name
                if old_name is not None and k0 == piece(tid, cur - 1):
                    names[k0] = old_name
                state["sel"] = (piece(tid, cur), tid)
                changed = True
        elif key == ord("a"):
            for p in people:
                k0 = piece(p["id"], cur)
                if k0 not in names:
                    gs = guess(k0, p["id"])
                    if gs:
                        names[k0] = gs
                        changed = True
        elif key in (13, 10):
            save(out, stem, pj, names, cuts, pose, piece)
            changed = False
        elif key in (ord("q"), 27):
            if changed:
                print("[label] unsaved changes: Enter to save, q again to quit without saving")
                changed = False
                continue
            break
    cv2.destroyAllWindows()


def save(out: Path, stem: str, pose_path: Path, names: dict, cuts: dict, pose: dict, piece):
    """Pieces (tied to this pose.json) + boxes per frame with their number (survive a new pose run)."""
    frames = {}
    for f, people in pose.items():
        row = []
        for p in people:
            if p["id"] < 0:
                continue
            v = names.get(piece(p["id"], int(f)))
            if v is not None:
                row.append({"box": p["box"], "player": v})
        if row:
            frames[f] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"schema": "vv-player-ids/1", "clip": stem, "pose": str(pose_path),
               "sides": "players 1-6 near side (camera side), 7-12 far side",
               "pieces": names, "cuts": {str(k): v for k, v in cuts.items()}, "frames": frames},
              open(out, "w"), separators=(",", ":"))
    print(f"[label] {stem}: saved {len(names)} named pieces, {sum(len(r) for r in frames.values())} boxes  →  {out}")


if __name__ == "__main__":
    main()
