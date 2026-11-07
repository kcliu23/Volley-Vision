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
  z                undo the last change (up to 50)
  g                after renaming someone: also rename their other pieces that still carry the old
                   number (the tracker often splits one person into several pieces)
    k              the tracker swapped people here: cut the piece at this frame (then name each part)
  a                take every "?n" guess shown in this frame
  h                boxes: thin full boxes (default) -> corners only -> numbers only
  n / Tab          jump to the next frame with an un-named person on court, or a number used twice
                   (red "7!" boxes: one of the two is wrong - see below)
  -> / ]   <- / [  0.5 s forward / back          . ,  one frame forward / back
  Enter / q        everything is saved as you go (data/player_ids/<clip>.json, after every change);
                   the labels as they were each time you opened the clip: <clip>.backup-<time>.json (last 5)

Same number on two boxes (red "7!"): one of them is wrong. Click the wrong one and:
  - a different person: press their number (renames that whole piece)
  - the tracker swapped two people at this moment (the piece was right before, wrong after):
    k to cut it here, then press the right number for the part from here on
  (Two boxes on ONE person - the detector found them twice - are fine: give both the same
  number. Boxes that mostly overlap aren't flagged red.)

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


from core import paths
from core.safe_io import backup, file_key, load_json, save_json

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
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))          # the container's count: can be a few too many
    done = d.exists() and len(list(d.glob("*.jpg"))) >= n - 5
    if done:
        n = len(list(d.glob("*.jpg")))                  # the frames that really decoded
    else:
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
    named_at: dict = {}                                     # piece key -> frame where it was last named
    cuts: dict = {}                                         # tracker id -> [frames where a new part starts]
    pose_key = file_key(pj)
    if out.exists():
        old = load_json(out, log=lambda m: print(f"[label] {stem}: {m}"))
        backup(out, keep=5, content=old)                    # the labels as loaded, one copy per session
        if old.get("pose_key") not in (None, pose_key):
            # pose.json was re-run: tracker ids mean other people now. Carry the numbers over by
            # box overlap from the saved boxes instead of by id.
            names, cuts, named_at_, report = remap_by_boxes(old.get("frames", {}), pose)
            named_at.update(named_at_)
            print(f"[label] {stem}: pose.json changed since these labels were made; carried "
                  f"{len(names)} pieces over by box overlap ({report}). Check them with g.")
        else:
            names = {k: v for k, v in old.get("pieces", {}).items()}
            cuts = {int(k): v for k, v in old.get("cuts", {}).items()}
            named_at.update(old.get("named_at", {}))
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
        if g is None or cuts.get(tid):
            return None                                      # a cut piece holds two people: no guess
        for k, v in names.items():
            if "@" in k or cuts.get(int(k)):
                continue
            if group.get(int(k)) == g and v != "x":
                return v
        return None

    def todo(f: int) -> list:
        return [p for p in pose.get(str(f), []) if p["id"] >= 0 and piece(p["id"], f) not in names
                and on_court.get(p["id"], True)]

    def clashes(f: int) -> set:
        """Numbers given to two boxes in the same frame that are NOT on the same person (boxes
        that overlap a lot are one person found twice: both may keep the number)."""
        byv = {}
        for p in pose.get(str(f), []):
            v = names.get(piece(p["id"], f)) if p["id"] >= 0 else None
            if v not in (None, "x"):
                byv.setdefault(v, []).append(p["box"])
        out = set()
        for v, bs in byv.items():
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    a, b = bs[i], bs[j]
                    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                    if ix < 0.6 * max(small, 1.0):
                        out.add(v)
        return out

    def progress():
        tot = done = 0
        for f in range(n):
            for p in pose.get(str(f), []):
                if p["id"] < 0 or not on_court.get(p["id"], True):
                    continue
                tot += 1
                done += piece(p["id"], f) in names
        return done / max(tot, 1)

    state = {"click": None, "sel": None, "boxes": "full",      # h: full -> corners -> numbers only
             "msg": None}                                    # (text, BGR colour, until time): feedback in the window
    history = []                                             # undo: (names, cuts) before each change

    def say(text, colour=(200, 255, 200), secs=3.0):
        state["msg"] = (text, colour, time.time() + secs)

    def remember():
        history.append(({k: v for k, v in names.items()}, {k: list(v) for k, v in cuts.items()}, dict(named_at)))
        del history[:-50]
        state["recount"] = True

    def piece_len(key0):
        tid = int(key0.split("@")[0])
        return sum(1 for f, ps in pose.items() if any(p["id"] == tid for p in ps) and piece(tid, int(f)) == key0)
    cur, changed, prog, prog_t = 0, False, progress(), time.time()

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    prev_frame = cur
    while True:
        if cur != prev_frame and state["sel"]:          # a new frame: keep the selection only if that piece is here
            here = {piece(p["id"], cur) for p in pose.get(str(cur), []) if p["id"] >= 0}
            if state["sel"][0] not in here:
                state["sel"] = None
        prev_frame = cur
        fp = fdir / f"{cur:05d}.jpg"
        img = cv2.imread(str(fp)) if fp.exists() else None
        if img is None:
            img = np.zeros((int(H * scale), int(W * scale), 3), np.uint8)
        canvas = np.zeros((img.shape[0] + TOP_H, img.shape[1], 3), np.uint8)
        canvas[TOP_H:] = img
        people = [p for p in pose.get(str(cur), []) if p["id"] >= 0]
        clash = clashes(cur)
        for p in people:
            key = piece(p["id"], cur)
            x1, y1, x2, y2 = [int(v * scale) for v in p["box"]]
            y1, y2 = y1 + TOP_H, y2 + TOP_H
            v = names.get(key)
            if v == "x":
                col, label, th = (90, 90, 90), "x", 1
            elif v is not None and v in clash:
                col, label, th = (0, 0, 255), f"{v}! used twice", 2   # same number twice in this frame
            elif v is not None:
                col, label, th = (NEAR if v <= 6 else FAR), str(v), 1
            else:
                gs = guess(key, p["id"])
                col, label, th = (170, 170, 170), (f"?{gs}" if gs else "?"), 1
                if not on_court.get(p["id"], True):
                    col, label = (70, 70, 70), "off?"
            sel_here = state["sel"] == (key, p["id"])
            if sel_here:
                col, th = (0, 255, 0), 1
            mode = state["boxes"]
            if mode == "full" or sel_here:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), col, th)
            elif mode == "corners":                         # just the corners: the player stays visible
                c = max(4, min(12, (x2 - x1) // 5))
                for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                    cv2.line(canvas, (px, py), (px + dx * c, py), col, th)
                    cv2.line(canvas, (px, py), (px, py + dy * c), col, th)
            # the number, small, just above the head (a clash: white on a red tag, easy to spot)
            ty = max(TOP_H + 12, y1 - 4)
            if label.endswith("used twice"):
                (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (x1 - 2, ty - tht - 4), (x1 + tw + 2, ty + 3), (0, 0, 220), -1)
                cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if state.pop("recount", False) or time.time() - prog_t > 1.0:
            prog, prog_t = progress(), time.time()
        left = len(todo(cur))
        top = (f"frame {cur}/{n - 1}   named: {prog:.0%} of on-court person-frames   "
               f"{left} to name here" + (f"   RED = number used twice: {sorted(clash)}" if clash else "") +
               "   1-6 near, 7-12 far   x not a player   k cut   a guesses   n next   h boxes   Enter save")
        cv2.putText(canvas, top, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        sel = state["sel"]
        msg = state["msg"]
        if msg and time.time() < msg[2]:                    # feedback for the last action
            cv2.putText(canvas, msg[0], (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, msg[1], 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, (f"selected: {sel[0]} - press its number (1-6 near, 7-12 far), x, u, k   z undo"
                                 if sel else "click a player's box to select it   z undo"),
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)

        raw = cv2.waitKeyEx(30)
        key = raw & 0xFF if raw != -1 else 255
        if raw != -1 or state["click"] is not None:      # a log of every key and click, to debug the tool
            try:
                with open(OUT_DIR / f"{stem}.log", "a") as lf:
                    lf.write(json.dumps({"t": round(time.time(), 2), "pid": __import__("os").getpid(), "frame": cur,
                                         "raw": raw, "click": state["click"], "sel": state["sel"],
                                         "n_named": len(names), "sel_name": names.get(state["sel"][0]) if state["sel"] else None}) + "\n")
            except OSError:
                pass
        if state["click"] is not None:
            cx, cy = state["click"]
            state["click"] = None
            # every box under the click, the one whose centre is closest first; clicking the same
            # spot again goes to the next one (people standing behind each other)
            under = []
            for p in people:
                x1, y1, x2, y2 = [v * scale for v in p["box"]]
                if x1 <= cx <= x2 and y1 + TOP_H <= cy <= y2 + TOP_H:
                    d = np.hypot((cx - (x1 + x2) / 2) / max(x2 - x1, 1), (cy - TOP_H - (y1 + y2) / 2) / max(y2 - y1, 1))
                    under.append((d, p))
            under.sort(key=lambda t: t[0])
            best = None
            if under:
                again = state.get("last_click") and np.hypot(cx - state["last_click"][0], cy - state["last_click"][1]) < 6 \
                    and state.get("last_frame") == cur
                i = (state.get("cycle", -1) + 1) % len(under) if again else 0
                state["cycle"] = i
                best = under[i]
                if len(under) > 1:
                    say(f"{len(under)} boxes here: click again on the same spot for the next one", (200, 230, 255))
            else:                                           # small far players: the nearest box within 15 px
                for p in people:
                    x1, y1, x2, y2 = [v * scale for v in p["box"]]
                    d = np.hypot(max(x1 - cx, 0, cx - x2), max(y1 + TOP_H - cy, 0, cy - y2 - TOP_H))
                    if d <= 15 and (best is None or d < best[0]):
                        best = (d, p)
            state["last_click"], state["last_frame"] = (cx, cy), cur
            state["sel"] = (piece(best[1]["id"], cur), best[1]["id"]) if best else None
            if best is None and cy >= TOP_H:
                say("no player there - click inside a box", (120, 200, 255))
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
            def needs(g):
                return bool(todo(g)) or bool(clashes(g))
            nxt = next((g for g in range(cur + 1, n) if needs(g)), None) or next((g for g in range(0, cur) if needs(g)), None)
            if nxt is None:
                say("Done: every on-court person is named, no number used twice.", (120, 255, 120), 5)
            else:
                wrapped = nxt < cur
                cur = nxt
                t_, c_ = len(todo(cur)), sorted(clashes(cur))
                why = ", ".join(([f"{t_} un-named (grey ?)"] if t_ else []) + ([f"number {c_} used twice (red !)"] if c_ else []))
                say(("back at the start: " if wrapped else "") + f"frame {cur} needs: {why}", (200, 230, 255), 5)
        elif (key in KEYS or key in (ord("x"), ord("u"), ord("k"))) and not state["sel"]:
            say("click a player's box first, then the key", (120, 200, 255))
        elif key in KEYS:
            remember()
            k0, tid = state["sel"]
            before = names.get(k0)
            v = KEYS[key]
            f0 = named_at.get(k0)
            handled = False
            fixing_clash = before in clashes(cur)          # this piece wears a red "used twice" tag here
            if (before not in (None, "x") and before != v and f0 is not None and f0 != cur and fixing_clash
                    and piece(tid, f0) == k0):
                # this piece was named `before` at frame f0, and here (where that number now clashes)
                # it is someone else: the tracker swapped people in between. (A plain correction, with
                # no clash here, just renames the piece.) Cut where its box jumps the most, keep the old
                # number on f0's side and the new one on this side.
                lo, hi = sorted((f0, cur))
                track = {}
                for g in range(lo, hi + 1):
                    for p in pose.get(str(g), []):
                        if p["id"] == tid:
                            track[g] = p["box"]
                fs = sorted(track)
                best_s, best_j = None, -1.0
                for a_, b_ in zip(fs, fs[1:]):
                    A, B = track[a_], track[b_]
                    h = max(A[3] - A[1], B[3] - B[1], 1.0)
                    j = np.hypot((A[0] + A[2] - B[0] - B[2]) / 2, (A[1] + A[3] - B[1] - B[3]) / 2) / h \
                        + abs((A[3] - A[1]) - (B[3] - B[1])) / h
                    if j > best_j:
                        best_s, best_j = b_, j
                if best_s is not None and best_s not in cuts.get(tid, []):
                    cuts[tid] = sorted(cuts.get(tid, []) + [best_s])
                    names[piece(tid, f0)] = before
                    named_at[piece(tid, f0)] = f0
                    k0 = piece(tid, cur)
                    state["sel"] = (k0, tid)
                    names[k0] = v
                    named_at[k0] = cur
                    changed = True
                    state["also"] = None
                    say(f"the tracker swapped people around frame {best_s}: cut there - before it {before if f0 < cur else v}, "
                        f"from it on {v if f0 < cur else before}   (check with , and .   z undo)", (120, 255, 255), 8)
                    handled = True
            if not handled:
                names[k0] = v
                named_at[k0] = cur
                changed = True
            # the same person is often several pieces (the tracker loses them now and then): other
            # pieces in the same re-ID group still carrying the OLD number are probably the same
            # mistake - offer to fix them all with g
            same = [] if handled or cuts.get(tid) else [
                k for k, nv in names.items() if before is not None and nv == before and k != k0 and "@" not in k
                and not cuts.get(int(k)) and group.get(int(k)) is not None and group.get(int(k)) == group.get(tid)]
            state["also"] = (same, v) if same else None
            if not handled:
                say(f"named {v} ({'near' if v <= 6 else 'far'} side) for {piece_len(k0)} frames" +
                    (f"   {len(same)} more piece(s) of this person still say {before}: g to change them too" if same else "") +
                    "   z undo", (120, 255, 255) if same else (200, 255, 200), 6 if same else 3)
        elif key == ord("x"):
            remember()
            names[state["sel"][0]] = "x"
            say(f"not a player, for {piece_len(state['sel'][0])} frames   z undo")
            changed = True
        elif key == ord("u"):
            remember()
            names.pop(state["sel"][0], None)
            say("un-named   z undo")
            changed = True
        elif key == ord("z"):
            if history:
                old_n, old_c, old_a = history.pop()
                names.clear(); names.update(old_n)
                cuts.clear(); cuts.update(old_c)
                named_at.clear(); named_at.update(old_a)
                state["sel"], state["also"] = None, None
                say("undone (selection cleared)")
                state["recount"] = True
                changed = True
            else:
                say("nothing to undo", (120, 200, 255))
        elif key == ord("k"):
            k0, tid = state["sel"]
            remember()
            if cur not in cuts.setdefault(tid, []):
                cuts[tid] = sorted(cuts[tid] + [cur])
                old_name = names.get(k0)                    # the earlier part keeps its name
                if old_name is not None and k0 == piece(tid, cur - 1):
                    names[k0] = old_name
                state["sel"] = (piece(tid, cur), tid)
                say("cut here: now name the part from this frame on   z undo")
                changed = True
        elif key == ord("g"):
            also = state.get("also")
            if also:
                remember()
                for k in also[0]:
                    names[k] = also[1]
                say(f"changed {len(also[0])} more piece(s) to {also[1]}   z undo")
                state["also"] = None
                changed = True
            else:
                say("nothing to change with g (it follows a renaming)", (120, 200, 255))
        elif key == ord("h"):
            state["boxes"] = {"full": "corners", "corners": "off", "off": "full"}[state["boxes"]]
        elif key == ord("a"):
            remember()
            took = 0
            for p in people:
                k0 = piece(p["id"], cur)
                if k0 not in names:
                    gs = guess(k0, p["id"])
                    if gs:
                        names[k0] = gs
                        changed = True
                        took += 1
            say(f"took {took} guess{'es' if took != 1 else ''}   z undo" if took else "no guesses in this frame",
                (200, 255, 200) if took else (120, 200, 255))
        elif key in (13, 10):
            save(out, stem, pj, names, cuts, pose, piece, named_at=named_at, pose_key=pose_key)
            say(f"Saved {len(names)} named pieces  ->  {out}", (120, 255, 120))
            changed = False
        elif key in (ord("q"), 27):
            if changed:
                save(out, stem, pj, names, cuts, pose, piece, named_at=named_at, pose_key=pose_key)
            break
        if changed:                                         # autosave after every change: nothing gets lost
            save(out, stem, pj, names, cuts, pose, piece, quiet=True, named_at=named_at, pose_key=pose_key)
            changed = False
    cv2.destroyAllWindows()


def save(out: Path, stem: str, pose_path: Path, names: dict, cuts: dict, pose: dict, piece, quiet: bool = False,
         named_at: dict | None = None, pose_key: str | None = None):
    """Pieces (tied to this pose.json, by pose_key) + boxes per frame with their number (survive a new pose run)."""
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
    save_json(out, {"schema": "vv-player-ids/1", "clip": stem, "pose": str(pose_path), "pose_key": pose_key,
                    "sides": "players 1-6 near side (camera side), 7-12 far side",
                    "pieces": names, "cuts": {str(k): v for k, v in cuts.items()}, "named_at": named_at or {},
                    "frames": frames}, separators=(",", ":"))   # temp file then swap: never half-written
    if not quiet:
        print(f"[label] {stem}: saved {len(names)} named pieces, {sum(len(r) for r in frames.values())} boxes  →  {out}")

def _iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def remap_by_boxes(saved_frames: dict, pose: dict, min_iou: float = 0.6):
    """Saved numbered boxes -> names for the pieces of a NEW pose run. Each new box takes the number
    of the saved box it overlaps best (IoU >= min_iou); a tracker id whose matched number changes
    over time is cut where it changes, so one id can carry several numbers, like a hand cut."""
    seq = {}                                              # tid -> [(frame, number)]
    for f, row in saved_frames.items():
        for p in pose.get(str(f), pose.get(f, [])) if isinstance(pose, dict) else []:
            if p["id"] < 0:
                continue
            best = max(row, key=lambda r: _iou(r["box"], p["box"]), default=None)
            if best is not None and _iou(best["box"], p["box"]) >= min_iou:
                seq.setdefault(p["id"], []).append((int(f), best["player"]))
    names, cuts, named_at, n_cut = {}, {}, {}, 0
    for tid, fs in seq.items():
        fs.sort()
        runs = []                                         # [number, first frame, frames]
        for f, v in fs:
            if runs and runs[-1][0] == v:
                runs[-1][2] += 1
            else:
                runs.append([v, f, 1])
        runs = [r for r in runs if r[2] >= 5] or [max(runs, key=lambda r: r[2])]   # drop flickers
        merged = [runs[0]]
        for r in runs[1:]:
            if r[0] == merged[-1][0]:
                merged[-1][2] += r[2]
            else:
                merged.append(r)
        for i, (v, f, _) in enumerate(merged):
            key = str(tid) if i == 0 else f"{tid}@{f}"
            if i:
                cuts.setdefault(tid, []).append(f)
                n_cut += 1
            names[key], named_at[key] = v, f
    return names, cuts, named_at, f"{len(seq)} tracker ids, {n_cut} new cuts"


if __name__ == "__main__":
    main()
