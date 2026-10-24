"""Check autolabel proposals rally by rally; what you accept becomes training (or test) data.

    python -m tools.review tests/test40.mov          # needs data/proposals/test40.json (tools.autolabel)
    python -m tools.review tests/test40.mov --test   # mark this clip as a test clip (never trained on)

Watch each rally with the proposed ball drawn on it. Where the circle is on the ball, do
nothing; where it isn't, pause and fix those frames. Then press Enter to accept the rally:
every frame you didn't touch takes the proposal (circle = ball visible there, no circle =
no ball visible). Saves after every action to data/verified/<clip>.json — quit any time,
it resumes where you left off.

  play           space  play / pause          w / s  faster / slower (¼× … 4×)
                 d / a  next / previous frame  D / A  ±10 frames
  fix a frame    click  the ball is here       n      no ball visible
                 h      hidden: in play but covered (behind a player, …)
                 x      can't tell (skip)      c      clear my edit (back to the proposal)
                 i      fill the frames between your last two clicks (straight line)
  move           Enter  accept this rally and go to the next unchecked one
                 f      next flagged frame     j / k  previous / next segment
  other          u      undo                   t      toggle train / test clip
                 ?      help                   q/Esc  save and quit

Red ticks on the timeline are frames autolabel flagged for a closer look (the reason shows
at the top); green = accepted, yellow = your edits. Ball labels: data/verified/<clip>.json
  {"labels": {"<frame>": [x, y] | "none" | "hidden" | "skip"}, "boxes": {"<frame>": [w, h]},
   "src": {"<frame>": "auto" | "fixed" | "interp"}, "role": "train" | "test", "reviewed": [[f0, f1], ...]}
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

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
MAX_W, MAX_H = 1500, 860
TIMELINE_H = 46
TOP_H = 58          # info bar above the picture (never covers the ball)


# ── review state (no GUI: testable on its own) ───────────────────────────────

class Review:
    def __init__(self, proposals: dict, verified: dict | None = None):
        self.P = proposals
        self.n = proposals["n"]
        v = verified or {}
        self.labels: dict[int, object] = {int(k): x for k, x in v.get("labels", {}).items()}
        self.boxes: dict[int, list] = {int(k): x for k, x in v.get("boxes", {}).items()}
        self.src: dict[int, str] = {int(k): x for k, x in v.get("src", {}).items()}
        self.reviewed: list[list[int]] = [list(r) for r in v.get("reviewed", [])]
        self.role = v.get("role", "train")
        self.clicks: list[int] = []              # frames clicked (for interpolate)
        self.undo_stack: list[tuple] = []

    # proposals
    def proposal(self, f: int):
        xy = self.P["frames"][f]["xy"]
        return list(xy) if xy is not None else "none"

    def proposal_box(self, f: int):
        return list(self.P["frames"][f]["box"])

    def doubt(self, f: int) -> list[str]:
        return self.P["frames"][f]["doubt"]

    # labels
    def is_edited(self, f: int) -> bool:
        return self.src.get(f) in ("fixed", "interp")

    def label(self, f: int):
        """What this frame will be saved as: your edit, an accepted proposal, or (unchecked) the proposal."""
        return self.labels.get(f, self.proposal(f))

    def is_reviewed(self, f: int) -> bool:
        return any(a <= f <= b for a, b in self.reviewed)

    def _push(self, frames):
        self.undo_stack.append(([(f, self.labels.get(f), self.boxes.get(f), self.src.get(f)) for f in frames],
                                [list(r) for r in self.reviewed], list(self.clicks)))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def _set(self, f, value, src, box=None):
        self.labels[f] = value
        self.src[f] = src
        if isinstance(value, list):
            self.boxes[f] = box or self.boxes.get(f) or self.proposal_box(f)
        else:
            self.boxes.pop(f, None)

    def set_ball(self, f: int, xy):
        self._push([f])
        self._set(f, [round(float(xy[0]), 1), round(float(xy[1]), 1)], "fixed")
        self.clicks = (self.clicks + [f])[-2:]

    def set_mark(self, f: int, value: str):
        assert value in ("none", "hidden", "skip")
        self._push([f])
        self._set(f, value, "fixed")

    def clear(self, f: int):
        self._push([f])
        for d in (self.labels, self.boxes, self.src):
            d.pop(f, None)

    def interpolate(self) -> int:
        """Straight line between the last two clicked frames; frames in between that you
        didn't edit yourself get it. Returns the number of frames filled."""
        if len(self.clicks) < 2:
            return 0
        a, b = sorted(self.clicks)
        pa, pb = self.labels.get(a), self.labels.get(b)
        if not (isinstance(pa, list) and isinstance(pb, list)) or b - a < 2:
            return 0
        todo = [f for f in range(a + 1, b) if self.src.get(f) != "fixed"]
        self._push(todo)
        for f in todo:
            w = (f - a) / (b - a)
            self._set(f, [round(pa[0] + w * (pb[0] - pa[0]), 1), round(pa[1] + w * (pb[1] - pa[1]), 1)], "interp")
        return len(todo)

    def accept(self, f0: int, f1: int) -> int:
        """Accept frames f0..f1: frames you didn't edit take the proposal."""
        frames = [f for f in range(f0, f1 + 1) if not self.is_edited(f)]
        self._push(frames)
        for f in frames:
            self._set(f, self.proposal(f), "auto", self.proposal_box(f))
        self.reviewed.append([f0, f1])
        self.reviewed = _merge(self.reviewed)
        return len(frames)

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        saved, reviewed, clicks = self.undo_stack.pop()
        for f, lab, box, src in saved:
            for d, v in ((self.labels, lab), (self.boxes, box), (self.src, src)):
                if v is None:
                    d.pop(f, None)
                else:
                    d[f] = v
        self.reviewed, self.clicks = reviewed, clicks
        return True

    # navigation helpers
    def segment_of(self, f: int) -> int:
        for i, s in enumerate(self.P["segments"]):
            if s["f0"] <= f <= s["f1"]:
                return i
        return 0

    def next_unreviewed_segment(self, after: int) -> int | None:
        segs = self.P["segments"]
        for i in list(range(after + 1, len(segs))) + list(range(0, after + 1)):
            s = segs[i]
            if not all(self.is_reviewed(f) for f in (s["f0"], s["f1"], (s["f0"] + s["f1"]) // 2)):
                return i
        return None

    def next_doubt(self, f: int) -> int | None:
        seg = self.P["segments"][self.segment_of(f)]
        for g in range(f + 1, seg["f1"] + 1):
            if self.doubt(g) and not self.is_edited(g):
                return g
        return None

    def progress(self) -> tuple[int, int]:
        done = sum(b - a + 1 for a, b in self.reviewed)
        return done, self.n

    def to_json(self, video: str) -> dict:
        P = self.P
        keep = {f for a, b in self.reviewed for f in range(a, b + 1)} | {f for f in self.labels if self.is_edited(f)}
        labels = {str(f): self.labels[f] for f in sorted(keep) if f in self.labels}
        return {"schema": "vv-verified/1", "video": str(video), "fps": P["fps"], "width": P["width"],
                "height": P["height"], "n": self.n, "every": 1, "role": self.role,
                "labels": labels,
                "boxes": {str(f): self.boxes[f] for f in sorted(keep) if f in self.boxes},
                "src": {str(f): self.src[f] for f in sorted(keep) if f in self.src},
                "reviewed": self.reviewed,
                "proposals": {"pipeline": P.get("pipeline"), "file": str(paths.proposals(Path(video).stem))}}


def _merge(rs):
    out = []
    for a, b in sorted(rs):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


# ── frames ───────────────────────────────────────────────────────────────────

class FrameSource:
    """Display-size frames, decoded in order (seeking is unreliable on phone video).
    Keeps the current segment in memory as JPEG; going backwards re-opens the file."""

    def __init__(self, video: str, scale: float):
        import cv2
        self.cv2, self.video, self.scale = cv2, video, scale
        self.cap, self.pos = None, 0
        self.cache: dict[int, bytes] = {}

    def _open(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.pos = self.cv2.VideoCapture(self.video), 0

    def load(self, f0: int, f1: int, pad: int = 30):
        cv2 = self.cv2
        f0 = max(0, f0 - pad)
        need = [f for f in range(f0, f1 + pad + 1) if f not in self.cache]
        if not need:
            return
        if self.cap is None or need[0] < self.pos:
            self._open()
        self.cache = {f: v for f, v in self.cache.items() if f0 <= f <= f1 + pad}
        while self.pos < need[0]:
            if not self.cap.grab():
                return
            self.pos += 1
        while self.pos <= need[-1]:
            ok, img = self.cap.read()
            if not ok:
                return
            if self.scale != 1.0:
                img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
            self.cache[self.pos] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            self.pos += 1

    def get(self, f: int):
        b = self.cache.get(f)
        if b is None:
            return None
        return self.cv2.imdecode(np.frombuffer(b, np.uint8), self.cv2.IMREAD_COLOR)


# ── drawing ──────────────────────────────────────────────────────────────────

def draw(cv2, img, R: Review, f: int, scale: float, seg_i: int, speed: float, playing: bool, mouse, msg: str):
    """img: canvas = info bar (TOP_H) + picture + timeline (TIMELINE_H)."""
    H, W = img.shape[:2]
    T = TOP_H
    lab = R.label(f)
    src = R.src.get(f, "proposal")
    col = {"fixed": (60, 220, 60), "interp": (150, 230, 150), "auto": (255, 220, 0), "proposal": (255, 220, 0)}[src]
    ball = R.P.get("ball_px", 30) * scale
    rad = max(10, int(ball * 0.9))
    if isinstance(lab, list):
        cv2.circle(img, (int(lab[0] * scale), T + int(lab[1] * scale)), rad, col, 2, cv2.LINE_AA)
    else:
        txt = {"none": "NO BALL", "hidden": "HIDDEN", "skip": "SKIP"}[lab]
        cv2.putText(img, txt, (W // 2 - 60, T + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
        p = R.proposal(f)
        if lab == "hidden" and isinstance(p, list):      # where the ball would be
            cv2.circle(img, (int(p[0] * scale), T + int(p[1] * scale)), rad, (120, 120, 120), 1, cv2.LINE_AA)
    # magnifier
    if mouse is not None and T <= mouse[1] < H - TIMELINE_H:
        mx, my = mouse
        r = 40
        x0, y0 = max(0, mx - r), max(T, my - r)
        crop = img[y0:min(my + r, H - TIMELINE_H), x0:mx + r].copy()
        if crop.size:
            z = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            zh, zw = z.shape[:2]
            ox = 8 if mx > 300 else W - zw - 8                     # keep it away from the cursor
            img[T + 8:T + 8 + zh, ox:ox + zw] = z
            cv2.rectangle(img, (ox, T + 8), (ox + zw, T + 8 + zh), (255, 255, 255), 1)
    # top bar
    seg = R.P["segments"][seg_i]
    done, tot = R.progress()
    top = (f"{Path(R.P['video']).stem}  frame {f}/{R.n - 1}   segment {seg_i + 1}/{len(R.P['segments'])} "
           f"({seg['kind']})   {'PLAY' if playing else 'pause'} {speed:g}x   checked {done / max(tot, 1):.0%}"
           f"   [{R.role}]")
    cv2.rectangle(img, (0, 0), (W, T), (0, 0, 0), -1)
    cv2.putText(img, top, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    d = R.doubt(f)
    lab_txt = {"fixed": "your label", "interp": "filled (line)", "auto": "accepted proposal",
               "proposal": "proposal"}[src]
    note = msg or (("check: " + ", ".join(d) + "   ") if d and not R.is_edited(f) else "") + lab_txt
    cv2.putText(img, note, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if msg else ((90, 90, 255) if d and not R.is_edited(f) else (180, 180, 180)),
                1, cv2.LINE_AA)
    # timeline
    y0 = H - TIMELINE_H
    cv2.rectangle(img, (0, y0), (W, H), (25, 25, 25), -1)
    xs = lambda g: int(g / max(R.n - 1, 1) * (W - 1))
    for s in R.P["segments"]:
        cv2.rectangle(img, (xs(s["f0"]), y0 + 6), (xs(s["f1"]), y0 + 22),
                      (110, 60, 20) if s["kind"] == "rally" else (70, 70, 70), -1)
    for a, b in R.reviewed:
        cv2.rectangle(img, (xs(a), y0 + 24), (xs(b), y0 + 30), (60, 200, 60), -1)
    for g in range(R.n):
        if R.doubt(g) and not R.is_edited(g):
            cv2.line(img, (xs(g), y0 + 32), (xs(g), y0 + 42), (60, 60, 255), 1)
        elif R.is_edited(g):
            cv2.line(img, (xs(g), y0 + 32), (xs(g), y0 + 42), (0, 220, 255), 1)
    cv2.line(img, (xs(f), y0), (xs(f), H), (255, 255, 255), 2)
    return img


# ── main loop ────────────────────────────────────────────────────────────────

def review(video: str, test: bool = False):
    import cv2
    stem = Path(video).stem
    pp = paths.proposals(stem)
    if not pp.exists():
        sys.exit(f"no proposals for {stem}: run  python -m tools.autolabel {video}")
    P = json.load(open(pp))
    vp = paths.verified(stem)
    V = json.load(open(vp)) if vp.exists() else None
    R = Review(P, V)
    if test:
        R.role = "test"
    scale = min(MAX_W / P["width"], (MAX_H - TIMELINE_H - TOP_H) / P["height"], 1.0)
    src = FrameSource(video, scale)

    def save():
        json.dump(R.to_json(video), open(paths.ensure(vp), "w"), separators=(",", ":"))

    seg_i = R.next_unreviewed_segment(-1)
    if seg_i is None:
        print(f"[review] {stem}: every segment is already checked (j/k to revisit)")
        seg_i = 0
    f = P["segments"][seg_i]["f0"]
    playing, speed_i = False, SPEEDS.index(1.0)
    state = {"mouse": None, "click": None}
    msg, msg_t = "", 0.0

    WIN = "VolleyVision review"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, x, y, flags, param):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
    cv2.setMouseCallback(WIN, on_mouse)

    last_step = time.time()
    while True:
        seg = P["segments"][seg_i]
        src.load(seg["f0"], seg["f1"])
        img = src.get(f)
        if img is None:
            img = np.zeros((int(P["height"] * scale), int(P["width"] * scale), 3), np.uint8)
        canvas = np.zeros((TOP_H + img.shape[0] + TIMELINE_H, img.shape[1], 3), np.uint8)
        canvas[TOP_H:TOP_H + img.shape[0]] = img
        show_msg = msg if time.time() - msg_t < 2.5 else ""
        cv2.imshow(WIN, draw(cv2, canvas, R, f, scale, seg_i, SPEEDS[speed_i], playing, state["mouse"], show_msg))
        key = cv2.waitKey(5) & 0xFF

        if state["click"] is not None:
            x, y = state["click"]
            state["click"] = None
            if TOP_H <= y < TOP_H + img.shape[0]:
                R.set_ball(f, (x / scale, (y - TOP_H) / scale)); save(); playing = False
            elif y >= TOP_H + img.shape[0]:                    # click on the timeline: jump there
                f = int(round(x / max(canvas.shape[1] - 1, 1) * (R.n - 1)))
                seg_i = R.segment_of(f); playing = False
        if key == 255:
            if playing and time.time() - last_step >= 1.0 / (P["fps"] * SPEEDS[speed_i]):
                last_step = time.time()
                if f < seg["f1"]:
                    f += 1
                else:
                    playing = False
            continue
        k = chr(key) if key < 128 else ""
        if k == " ":
            playing = not playing; last_step = time.time()
        elif k in ("d", "."):
            f = min(R.n - 1, f + 1); playing = False
        elif k in ("a", ","):
            f = max(0, f - 1); playing = False
        elif k == "D":
            f = min(R.n - 1, f + 10); playing = False
        elif k == "A":
            f = max(0, f - 10); playing = False
        elif k == "w":
            speed_i = min(len(SPEEDS) - 1, speed_i + 1)
        elif k == "s":
            speed_i = max(0, speed_i - 1)
        elif k == "n":
            R.set_mark(f, "none"); save()
        elif k == "h":
            R.set_mark(f, "hidden"); save()
        elif k == "x":
            R.set_mark(f, "skip"); save()
        elif k == "c":
            R.clear(f); save()
        elif k == "i":
            m = R.interpolate(); save()
            msg, msg_t = (f"filled {m} frames" if m else "click the ball in two frames first"), time.time()
        elif key in (13, 10):
            m = R.accept(seg["f0"], seg["f1"]); save()
            nxt = R.next_unreviewed_segment(seg_i)
            msg, msg_t = f"accepted segment {seg_i + 1} ({m} frames from the proposal)", time.time()
            if nxt is None:
                msg = "all segments checked - q to quit"
            else:
                seg_i = nxt; f = P["segments"][seg_i]["f0"]
            playing = False
        elif k == "f":
            g = R.next_doubt(f)
            if g is None:
                msg, msg_t = "no more flagged frames in this segment", time.time()
            else:
                f = g
            playing = False
        elif k == "k":
            seg_i = min(len(P["segments"]) - 1, seg_i + 1); f = P["segments"][seg_i]["f0"]; playing = False
        elif k == "j":
            seg_i = max(0, seg_i - 1); f = P["segments"][seg_i]["f0"]; playing = False
        elif k == "u":
            msg, msg_t = ("undone" if R.undo() else "nothing to undo"), time.time(); save()
        elif k == "t":
            R.role = "test" if R.role == "train" else "train"; save()
            msg, msg_t = f"clip role: {R.role}", time.time()
        elif k == "?":
            print(__doc__)
            msg, msg_t = "help printed in the terminal", time.time()
        elif k == "q" or key == 27:
            save(); break
        seg_i = R.segment_of(f) if not (P["segments"][seg_i]["f0"] <= f <= P["segments"][seg_i]["f1"]) else seg_i
    cv2.destroyAllWindows()
    done, tot = R.progress()
    print(f"[review] {stem}: checked {done}/{tot} frames, {sum(1 for f in R.src if R.is_edited(f))} edited  →  {vp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--test", action="store_true", help="mark as a test clip (kept out of training)")
    a = ap.parse_args()
    for v in a.videos:
        review(v, a.test)


if __name__ == "__main__":
    main()
