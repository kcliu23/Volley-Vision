"""Check autolabel proposals rally by rally; what you accept becomes training (or test) data.

    python -m tools.review videos/test40.mov          # needs data/proposals/test40.json (tools.autolabel)
    python -m tools.review videos/test40.mov --test   # mark this clip as a test clip (never trained on)

Press space and watch the clip with the proposed ball drawn on it (it plays through the
whole video). Where the circle is on the ball, do nothing; where it isn't, pause and fix
those frames. Accept with Enter (the segment you're in) or y (the whole clip): every frame
you didn't touch takes the proposal (circle = ball visible there, no circle = no ball). Saves after every action to data/verified/<clip>.json — quit any time,
it resumes where you left off.

  play           space  play / pause (through the whole clip)   w / s  faster / slower (¼× … 4×)
                 d / a  next / previous frame  D / A  ±10 frames
  fix a frame    click  the ball is here: snaps to the centre of a YOLO box near your click if
                        there is one (more exact than a click); z  toggle snapping off / on
                        n      no ball visible
                 N      no ball here AND in the following frames while the proposal stays on
                        that same object (a wall sign, a fan, a head: one key, not 30); stops
                        where the proposal jumps somewhere else
                 h      hidden: in play but covered (behind a player, …)
                 x      can't tell (skip)      c      clear my edit (back to the proposal)
                 i      fill the frames between your last two clicks (straight line; each
                        filled frame snaps to a YOLO box near the line, like a click)
                 I      fill between ALL your clicks in this segment (held / slow ball:
                        click it every 10-20 frames, then press I once)
  accept         Enter  accept the segment you're in and go to the next unchecked one
                 y      accept the whole clip (after watching it all)
  move           f      next flagged frame     j / k  previous / next segment
                 b      next frame to re-check in YOUR labels (purple ticks): a ball that jumps, sits
                        off its arc, blinks out for one frame, a long fill at a touch, a touch with a
                        gap or two types, a rally without its serve / rally over mark
                        (core.common.label_checks); the reason shows at the top
                 click the timeline to jump there
  other balls    mark EVERY volleyball that isn't the game ball (spare balls, one on the sideline,
                 one held by someone): WASB is then never taught that a real ball is "nothing".
                 orange = other balls you marked; grey = YOLO suggestions (often a clock: ignore)
                 shift+click a ball     marks it. If it sits still, it's marked in every frame it
                                        stays there (checked by its look, a few seconds)
                 moving ball            shift+click it every time it changes direction, then g:
                                        straight lines between all those clicks
                 shift+click orange     wrong box: removed from this frame to the end of its stretch
                                        (option+shift+click: back to its start)
                 r / R                  remove the orange box under the mouse / every orange box,
                                        in THIS frame only          u  undo
                 o                      next frame with a grey suggestion
                 (F: old crop following with YOLO, slow; off by default)
                 Every action goes to data/verified/.logs/<clip>.log.
  contacts       m      mark a touch at this frame, then its type: 1 serve, 2 pass/receive,
                        (ball held a few frames, e.g. a set: mark every frame; m on a frame next
                        to a marked touch joins it: same type and toucher, no questions)
                        3 set, 4 hit (anything sent over that isn't a serve or block), 5 block,
                        6 dig, 0 other (m on a marked frame removes it)
                        then, optional, who touched it: the numbered players are drawn (your
                        numbers from tools.label_players) and the one nearest the ball is
                        suggested in yellow: Enter accepts it, click another player's box, or
                        type a number + Enter (a player with no box: hidden, not detected);
                        any other key skips. Saved: touched_by (number) + touched_box (their
                        pose box, so the touch is tied to that person in the pose data)
                        9 rally over: the ball is dead (not a touch). Serve -> rally over marks
                        which frames are IN PLAY: the scores report those on their own
                 v      on a marked touch: NOT SEEN (toggle). The touch happened but you didn't
                        see it: hidden behind a player, or a serve hit off-screen / before the
                        video starts (mark 1 serve at the first frame you see the ball, then v).
                        Mark it at your best-guess frame; grey tick. Not scored for timing or
                        reach, and a not-seen serve isn't used as "the serve starts far back"
                 M      jump to the next marked touch
  other          u      undo                   t      toggle train / test clip
                 ?      help                   q/Esc  save and quit

Red ticks on the timeline are frames autolabel flagged for a closer look (the reason shows
at the top); green = accepted, yellow = your edits. Ball labels: data/verified/<clip>.json
  {"labels": {"<frame>": [x, y] | "none" | "hidden" | "skip"}, "boxes": {"<frame>": [w, h]},
   "src": {"<frame>": "auto" | "fixed" | "interp"}, "role": "train" | "test", "reviewed": [[f0, f1], ...],
   "others": {"<frame>": [[x, y, w, h], ...]},   other balls you confirmed (only these are exported)
   "contacts": {"<frame>": "serve" | "pass" | "set" | "hit" | "block" | "dig" | "other"},   touches you marked
   "touched_by": {"<frame>": 7}, "touched_box": {"<frame>": [x0, y0, x1, y1]},  who touched it, when you said
   "unseen": [<frame>, ...]}   touches marked but not seen (v): frame is a best guess
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from core.common import paths
from core.common.safe_io import backup, load_json, save_json

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
MAX_W, MAX_H = 1500, 860
TIMELINE_H = 46
TOP_H = 58          # info bar above the picture (never covers the ball)


# ── review state (no GUI: testable on its own) ───────────────────────────────

class Review:
    def __init__(self, proposals: dict, verified: dict | None = None, dets_frames: list | None = None):
        self.P = proposals
        self.n = proposals["n"]
        v = verified or {}
        self.dets = dets_frames                  # YOLO boxes per frame (for the other-ball suggestions)
        sizes = sorted(max(fr["box"]) for fr in proposals["frames"] if fr.get("xy") is not None and fr.get("box"))
        self.ball = sizes[len(sizes) // 2] if sizes else 30.0
        self.others_edit: dict[int, list] = {int(k): x for k, x in v.get("others", {}).items()}
        self.contacts: dict[int, str] = {int(k): {"attack": "hit"}.get(x, x)       # "attack" was the old name
                                         for k, x in v.get("contacts", {}).items()}
        self.touched_by: dict[int, int] = {int(k): int(x) for k, x in v.get("touched_by", {}).items()}
        self.touched_box: dict[int, list] = {int(k): x for k, x in v.get("touched_box", {}).items()}
        self.unseen: set[int] = {int(k) for k in v.get("unseen", [])}   # touches marked but not seen (v)
        self.box_at = None                       # (frame, number, box) -> that player's box at frame (or None)
        self._checks = None                      # label_checks flags, rebuilt after an edit
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

    # other balls
    def game_xy(self, f: int):
        lab = self.label(f)
        if isinstance(lab, list):
            return lab
        p = self.proposal(f)
        return p if isinstance(p, list) else None

    def others(self, f: int) -> list:
        """Other balls you confirmed in this frame (these go into the training labels)."""
        return self.others_edit.get(f, [])

    def suggestions(self, f: int) -> list:
        """Confident YOLO boxes away from the game ball that you haven't confirmed."""
        if not self.dets or f >= len(self.dets):
            return []
        from core.ball.other_balls import auto
        mine = self.others(f)
        return [o for o in auto(self.dets[f], self.game_xy(f), self.ball)
                if not any(((o[0] - m[0]) ** 2 + (o[1] - m[1]) ** 2) ** 0.5 <= self.ball for m in mine)]

    @staticmethod
    def _nearest(boxes, xy, r: float):
        best = None
        for i, o in enumerate(boxes):
            d = ((o[0] - xy[0]) ** 2 + (o[1] - xy[1]) ** 2) ** 0.5
            if d <= r and (best is None or d < best[1]):
                best = (i, d)
        return None if best is None else best[0]

    FOLLOW_CONF = 0.1        # once you confirmed a ball, weaker YOLO boxes at that spot count too
    FOLLOW_GAP_S = 1.0       # ... and gaps up to this long (YOLO lost it) are bridged
    STILL_GAP_S = 5.0        # ... or up to this long if it reappears at the same spot (resting ball)

    def _follow(self, f: int, spot, boxes_at, direction: int, with_spot: bool = False) -> dict:
        """Follow one object from frame f in one direction through the whole clip: frame -> box.
        boxes_at(g) gives candidate boxes in frame g. Gaps where YOLO lost it are bridged by
        interpolating between the sightings on both sides: up to FOLLOW_GAP_S, or up to
        STILL_GAP_S when it reappears at the same spot (a resting ball doesn't go anywhere)."""
        fps = self.P.get("fps", 30)
        max_gap = max(1, int(round(self.FOLLOW_GAP_S * fps)))
        max_still = max(max_gap, int(round(self.STILL_GAP_S * fps)))
        hits, last = {}, f
        while True:
            found = None
            # past max_gap only a ball at the same spot counts; with crop searches (with_spot: one
            # YOLO call per look) check every 4th frame there, the bridge fills the rest
            steps = list(range(1, max_gap + 1)) + list(range(max_gap + 1, max_still + 1, 4 if with_spot else 1))
            for g in (last + direction * k for k in steps):
                if not 0 <= g < self.n:
                    break
                cand = boxes_at(g, spot) if with_spot else boxes_at(g)
                j = self._nearest(cand, spot, self.ball)
                if j is not None:
                    box = cand[j]
                    still = ((box[0] - spot[0]) ** 2 + (box[1] - spot[1]) ** 2) ** 0.5 <= 0.5 * self.ball
                    if abs(g - last) <= max_gap or still:
                        found = (g, box)
                    break
            if found is None:
                return hits
            g, box = found
            hits[g] = box
            pa = hits.get(last) or self._origin
            for h in range(last + direction, g, direction):              # bridge the gap
                w = (h - last) / (g - last)
                hits[h] = [round(pa[k] + w * (box[k] - pa[k]), 1) for k in range(4)]
            last, spot = g, box[:2]

    def _boxes_near_play(self, g: int) -> list:
        """YOLO boxes in frame g (>= FOLLOW_CONF) that aren't the game ball."""
        if not self.dets or g >= len(self.dets):
            return []
        game = self.game_xy(g)
        out = []
        for b in self.dets[g]:
            if b[4] < self.FOLLOW_CONF:
                continue
            if game is not None and ((b[0] - game[0]) ** 2 + (b[1] - game[1]) ** 2) ** 0.5 <= 1.5 * self.ball:
                continue
            out.append([round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)])
        return out

    crop_find = None         # set by review(): (frame, spot) -> YOLO boxes on a full-res crop there
    still_find = None        # set by review(): (frame, box) -> (a, b) frames a resting ball stays put

    def _add_still(self, f: int, box, frames: dict) -> dict:
        """frames + every frame the clicked ball rests at the same spot (checked by its look)."""
        if self.still_find is None:
            return frames
        a, b = self.still_find(f, box)
        for g in range(a, b + 1):
            frames.setdefault(g, [round(float(box[0]), 1), round(float(box[1]), 1),
                                  round(float(box[2]), 1), round(float(box[3]), 1)])
        return frames
    start_follow = None      # set by review(): (frame, box) -> remember what this ball looks like

    def _boxes_or_crop(self, g: int, spot) -> list:
        """Cached YOLO boxes near the spot; where the full-frame pass missed the ball (small,
        far, resting balls often are), YOLO on a crop around the spot instead."""
        cached = self._boxes_near_play(g)
        if self._nearest(cached, spot, self.ball) is not None or self.crop_find is None:
            return cached
        game = self.game_xy(g)
        return [b for b in self.crop_find(g, spot)
                if game is None or ((b[0] - game[0]) ** 2 + (b[1] - game[1]) ** 2) ** 0.5 > 1.5 * self.ball]

    def other_click_target(self, f: int, xy):
        """For a shift+click away from confirmed balls and suggestions: a weaker YOLO box near it
        (not the game ball) to snap to and follow, or None."""
        r = max(self.ball, 12.0)
        if self._nearest(self.others(f), xy, r) is not None or self._nearest(self.suggestions(f), xy, r) is not None:
            return None
        boxes = self._boxes_near_play(f)
        j = self._nearest(boxes, xy, self.SNAP_BW * self.ball)
        return boxes[j] if j is not None else None

    def remove_other_here(self, f: int, xy) -> str:
        """Remove the orange box nearest xy in THIS frame only (a wrong box in a crowd)."""
        i = self._nearest(self.others(f), xy, 2.0 * max(self.ball, 12.0))
        if i is None:
            return "no orange box near the mouse in this frame"
        self._push([f])
        cur = self.others(f)[:i] + self.others(f)[i + 1:]
        if cur:
            self.others_edit[f] = cur
        else:
            self.others_edit.pop(f, None)
        return f"removed 1 orange box in frame {f} ({len(cur)} left here)"

    def clear_others_here(self, f: int) -> str:
        """Remove every orange box in THIS frame."""
        n = len(self.others(f))
        if not n:
            return "no orange boxes in this frame"
        self._push([f])
        self.others_edit.pop(f, None)
        return f"removed all {n} orange boxes in frame {f}"

    def _remove_other(self, f: int, i: int, backward: bool = False) -> str:
        """Remove the clicked orange ball from this frame to the end of its track (forward), or
        to its start (backward): a follow that went wrong at frame f is cut there, and what came
        before stays. From its first frame forward = the whole track."""
        box = self.others(f)[i]
        self._origin = box
        frames = {f: box}
        frames.update(self._follow(f, box[:2], self.others, -1 if backward else +1))
        self._push(list(frames))
        for g, b in frames.items():
            cur = [o for o in self.others(g) if ((o[0] - b[0]) ** 2 + (o[1] - b[1]) ** 2) ** 0.5 > self.ball]
            if cur:
                self.others_edit[g] = cur
            else:
                self.others_edit.pop(g, None)
        self.other_clicks = [c for c in self.other_clicks if c[0] not in frames]
        lo, hi = min(frames), max(frames)
        other = "shift+click" if backward else "option+shift+click"
        return (f"removed the other ball from {len(frames)} frame{'s' if len(frames) != 1 else ''} ({lo}-{hi}, "
                f"{'back to its start' if backward else 'to its end'}; {other} cuts the other way; u undoes)")

    def toggle_other(self, f: int, xy, until: int | None = None, crop_box=None, backward: bool = False) -> str:
        """On a confirmed ball: remove it, here and everywhere it continues (both directions).
        On a suggestion, or (snapping on) a weaker YOLO box near the click: confirm it, here and
        everywhere YOLO keeps seeing it at that spot (both directions, whole clip, weaker boxes
        and gaps up to a second included). Elsewhere: a ball in this frame only, at crop_box
        (a box YOLO found on a crop around the click) if given, else at the click."""
        r = max(self.ball, 12.0)
        i = self._nearest(self.others(f), xy, r)
        if not self.follow:
            # default: mark the ball here; a still ball everywhere it stays (by its look); a moving
            # ball by keyframes (g draws straight lines between them). No following through YOLO's
            # boxes: YOLO rarely sees spare balls, and in a crowd it drifts onto heads and knees.
            if i is not None:
                return self._remove_other(f, i, backward)
            k = self._nearest(self.suggestions(f), xy, r)
            box = (self.suggestions(f)[k] if k is not None
                   else self.other_click_target(f, xy) if self.snap else None)        # snap the position only
            if box is None:
                box = list(crop_box) if crop_box is not None else \
                    [round(float(xy[0]), 1), round(float(xy[1]), 1), self.ball, self.ball]
            box = [round(float(v), 1) for v in box[:4]]
            frames = self._add_still(f, box, {f: box})
            new = {g: b for g, b in frames.items()
                   if not any(((o[0] - b[0]) ** 2 + (o[1] - b[1]) ** 2) ** 0.5 <= self.ball for o in self.others(g))}
            self._push(list(new) or [f])
            for g, b in new.items():
                self.others_edit[g] = self.others(g) + [b]
            if len(frames) > 1:
                return f"still ball: marked in {len(frames)} frames ({min(frames)}-{max(frames)})"
            self.other_clicks = [c for c in self.other_clicks if c[0] != f] + [(f, box)]
            n = len(self.other_clicks)
            return (f"marked (keyframe {n})" + ("   g: straight lines between the keyframes" if n > 1
                                                 else "   moving? click it again a few frames on, then g"))
        if i is not None:                                  # an orange ball: cut it from here (as with F off)
            return self._remove_other(f, i, backward)
        i = self._nearest(self.suggestions(f), xy, r)
        weak = self.other_click_target(f, xy) if (i is None and self.snap) else None
        if i is None and weak is None and self.crop_find is not None:
            # found on a crop, or (nothing YOLO can see) the clicked spot: follow it with crops,
            # and where YOLO stays blind, by what the ball looks like at the click
            weak = list(crop_box) if crop_box is not None else \
                [round(float(xy[0]), 1), round(float(xy[1]), 1), self.ball, self.ball]
        if i is not None or weak is not None:              # confirm a suggestion / weak box, following it
            box = self.suggestions(f)[i] if i is not None else weak
            self._origin = box
            print("[review] following the other ball ...", flush=True)
            if self.start_follow is not None:
                self.start_follow(f, box)
            frames = {f: box}
            for d in (+1, -1):
                if self.start_follow is not None:
                    self.start_follow(f, box)
                frames.update(self._follow(f, box[:2], self._boxes_or_crop, d, with_spot=True))
            new = {g: b for g, b in frames.items()
                   if not any(((o[0] - b[0]) ** 2 + (o[1] - b[1]) ** 2) ** 0.5 <= self.ball for o in self.others(g))}
            self._push(list(new))
            for g, b in new.items():
                self.others_edit[g] = self.others(g) + [b]
            lo, hi = min(frames), max(frames)
            return f"confirmed another ball in {len(frames)} frames ({lo}-{hi})"
        self._push([f])
        if crop_box is not None:
            self.others_edit[f] = self.others(f) + [list(crop_box)]
            return "added another ball (this frame, snapped to a YOLO box on a crop)"
        self.others_edit[f] = self.others(f) + [[round(float(xy[0]), 1), round(float(xy[1]), 1), self.ball, self.ball]]
        return "added another ball (this frame, at your click)"

    def fill_others(self) -> int:
        """Straight lines between all shift+click keyframes since the last g (one moving ball),
        in frame order; frames in between get it unless they already have a ball there. The
        keyframes are then cleared, ready for the next ball. Returns the number of frames filled."""
        keys = sorted(self.other_clicks)
        if len(keys) < 2:
            return 0
        todo = {}
        for (a, pa), (b, pb) in zip(keys, keys[1:]):
            for g in range(a + 1, b):
                w = (g - a) / (b - a)
                box = [round(pa[k] + w * (pb[k] - pa[k]), 1) for k in range(4)]
                if not any(((o[0] - box[0]) ** 2 + (o[1] - box[1]) ** 2) ** 0.5 <= self.ball for o in self.others(g)):
                    todo[g] = box
        if todo:
            self._push(list(todo))
            for g, box in todo.items():
                self.others_edit[g] = self.others(g) + [box]
        self.other_clicks = []
        return len(todo)

    def next_suggestion(self, f: int) -> int | None:
        """Next frame with an unconfirmed suggestion, skipping the rest of the current run of them."""
        g = f + 1
        while g < self.n and self.suggestions(g) and self.suggestions(f):
            g += 1
        for h in range(g, self.n):
            if self.suggestions(h):
                return h
        return None

    CONTACT_TYPES = {"1": "serve", "2": "pass", "3": "set", "4": "hit", "5": "block", "6": "dig", "0": "other",
                     "9": "dead"}                      # 9: not a touch: the rally is over (ball dead)

    def set_contact(self, f: int, kind: str | None):
        """Mark a touch at frame f (kind from CONTACT_TYPES values), or remove it (None)."""
        self._push([f])
        if kind is None:
            self.contacts.pop(f, None)
            self.touched_by.pop(f, None)
            self.touched_box.pop(f, None)
            self.unseen.discard(f)
        else:
            self.contacts[f] = kind

    def touch_run(self, f: int) -> list[int]:
        """The frames of the touch at f: consecutive marked frames of the same type (the ball held
        on the hands for a few frames is one touch)."""
        if f not in self.contacts:
            return []
        kind, a, b = self.contacts[f], f, f
        while self.contacts.get(a - 1) == kind:
            a -= 1
        while self.contacts.get(b + 1) == kind:
            b += 1
        return list(range(a, b + 1))

    def toggle_unseen(self, f: int) -> bool | None:
        """v: the touch at f (all its frames) marked not seen, or back to seen. Returns the new state,
        or None (no touch at f, or a rally-over mark)."""
        run = self.touch_run(f)
        if not run or self.contacts[f] == "dead":
            return None
        self._push(run)
        on = not any(g in self.unseen for g in run)
        for g in run:
            (self.unseen.add if on else self.unseen.discard)(g)
        return on

    def extend_touch(self, f: int) -> list[int] | None:
        """m on a frame right next to a marked touch: f joins that touch (same type, same toucher),
        no questions. Returns the touch's frames, or None (no touch next to f)."""
        for g in (f - 1, f + 1):
            kind = self.contacts.get(g)
            if kind is None or kind == "dead" or f in self.contacts:
                continue
            run = self.touch_run(g)
            self._push([f])
            self.contacts[f] = kind
            if any(h in self.unseen for h in run):
                self.unseen.add(f)
            who = next((self.touched_by[h] for h in run if h in self.touched_by), None)
            if who is not None:
                self.touched_by[f] = who
                h = min((h for h in run if h in self.touched_box), key=lambda h: abs(h - f), default=None)
                if h is not None:
                    box = self._toucher_box(f, who, self.touched_box[h])
                    if box is not None:
                        self.touched_box[f] = box
            return self.touch_run(f)
        return None

    def set_toucher(self, f: int, number: int | None, box: list | None = None):
        """Who touched the ball at frame f (a player number, and their pose box when they were
        detected: it ties the touch to that person in the pose data), or not known (None). Applies
        to every frame of that touch (touch_run)."""
        run = self.touch_run(f) or [f]
        self._push(run)
        for g in run:
            if number is None:
                self.touched_by.pop(g, None)
            else:
                self.touched_by[g] = int(number)
            b = None
            if box is not None:
                b = [round(float(v), 1) for v in box] if g == f else self._toucher_box(g, number, box)
            if b is None:
                self.touched_box.pop(g, None)
            else:
                self.touched_box[g] = b

    def _toucher_box(self, g: int, number, box) -> list | None:
        """That player's box at frame g (they move during a touch): from the pose players via
        box_at; without poses, the box as it was."""
        b = self.box_at(g, number, box) if self.box_at else box
        return None if b is None else [round(float(v), 1) for v in b]

    def checks(self) -> dict[int, str]:
        """Frames to re-check in your labels (core.common.label_checks), cached until an edit."""
        if self._checks is None:
            from core.common.label_checks import all_flags
            self._checks = all_flags({"labels": self.labels, "contacts": self.contacts, "src": self.src})
        return self._checks

    def next_check(self, f: int) -> int | None:
        C = sorted(self.checks())
        later = [g for g in C if g > f]
        return later[0] if later else (C[0] if C else None)

    def next_contact(self, f: int) -> int | None:
        later = [g for g in sorted(self.contacts) if g > f]
        return later[0] if later else None

    def _push(self, frames):
        self._checks = None
        self.undo_stack.append(([(f, self.labels.get(f), self.boxes.get(f), self.src.get(f), self.others_edit.get(f),
                                  self.contacts.get(f), self.touched_by.get(f), self.touched_box.get(f),
                                  f in self.unseen) for f in frames],
                                [list(r) for r in self.reviewed], list(self.clicks), list(self.other_clicks)))
        if len(self.undo_stack) > 200:
            self.undo_stack.pop(0)

    def _set(self, f, value, src, box=None):
        self.labels[f] = value
        self.src[f] = src
        if isinstance(value, list):
            self.boxes[f] = box or self.boxes.get(f) or self.proposal_box(f)
        else:
            self.boxes.pop(f, None)

    SNAP_BW = 0.75           # a click snaps to a YOLO box whose centre is within this many ball-widths

    def snap_target(self, f: int, xy):
        """The YOLO box (from autolabel's full-frame pass) nearest to a click, if its centre is
        within SNAP_BW ball-widths: [cx, cy, w, h], else None. Hand clicks wobble ~2.4x more
        than accepted proposals; a box centre is pixel-accurate."""
        if not self.dets or f >= len(self.dets):
            return None
        best = None
        for b in self.dets[f]:
            if b[4] < 0.05:
                continue
            d = ((b[0] - xy[0]) ** 2 + (b[1] - xy[1]) ** 2) ** 0.5
            if d <= self.SNAP_BW * self.ball and (best is None or d < best[0]):
                best = (d, [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)])
        return None if best is None else best[1]

    def set_ball(self, f: int, xy, box=None):
        """Game ball at xy in frame f; box = [cx, cy, w, h] of a YOLO box it was snapped to."""
        self._push([f])
        if box is not None:
            xy = box[:2]
        self._set(f, [round(float(xy[0]), 1), round(float(xy[1]), 1)], "fixed",
                  box=[box[2], box[3]] if box is not None else None)
        self.clicks = ([c for c in self.clicks if c != f] + [f])[-2:]   # re-clicking a frame just updates it

    def set_mark(self, f: int, value: str):
        assert value in ("none", "hidden", "skip")
        self._push([f])
        self._set(f, value, "fixed")

    def set_none_run(self, f: int) -> int:
        """"No ball" for f and the following frames where the proposal stays on the same object
        (each step within 1.5 ball-widths; it may blink off for up to 0.5 s), whatever its state:
        a wall object, a fan, a head. Stops where the proposal jumps elsewhere (maybe the real
        ball) or at your own edits. Returns the count."""
        last = self.proposal(f)
        todo, pending = [f], []
        max_gap = max(1, int(round(0.5 * self.P.get("fps", 30))))
        for g in range(f + 1, self.n):
            if self.src.get(g) == "fixed":
                break
            p = self.proposal(g)
            if not isinstance(p, list):                    # no circle here: fine, keep going a little
                pending.append(g)
                if len(pending) > max_gap:
                    break
                continue
            if isinstance(last, list) and ((p[0] - last[0]) ** 2 + (p[1] - last[1]) ** 2) ** 0.5 > 1.5 * self.ball:
                break                                      # jumped to something else
            todo += pending + [g]
            pending, last = [], p
        self._push(todo)
        for g in todo:
            self._set(g, "none", "fixed")
        return len(todo)

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
            self._set_interp(f, [pa[0] + w * (pb[0] - pa[0]), pa[1] + w * (pb[1] - pa[1])])
        return len(todo)

    snap = True               # the review loop's z key turns snapping off for clicks and fills
    follow = False            # F: shift+click follows the ball through the clip (slow where YOLO can't see it)
    other_clicks: list = []   # (frame, box) shift+click keyframes since the last g (one moving ball)

    def _set_interp(self, f: int, xy):
        """A filled frame: on the straight line, or on a YOLO box near it (snapping on)."""
        box = self.snap_target(f, xy) if self.snap else None
        if box is not None:
            self._set(f, box[:2], "interp", box=[box[2], box[3]])
        else:
            self._set(f, [round(xy[0], 1), round(xy[1], 1)], "interp")

    def interpolate_all(self, f0: int, f1: int, max_gap: int = 60) -> int:
        """Fill between every pair of consecutive frames you clicked in f0..f1 (straight line),
        for gaps up to max_gap frames. For a held or slow ball: click every 10-20 frames, then
        fill once. Frames you set yourself (click / n / h / x) are never overwritten."""
        keys = [f for f in range(f0, f1 + 1) if self.src.get(f) == "fixed" and isinstance(self.labels.get(f), list)]
        todo = []
        for a, b in zip(keys, keys[1:]):
            if 1 < b - a <= max_gap:
                todo += [(f, a, b) for f in range(a + 1, b) if self.src.get(f) != "fixed"]
        if not todo:
            return 0
        self._push([f for f, _, _ in todo])
        for f, a, b in todo:
            pa, pb, w = self.labels[a], self.labels[b], (f - a) / (b - a)
            self._set_interp(f, [pa[0] + w * (pb[0] - pa[0]), pa[1] + w * (pb[1] - pa[1])])
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
        saved, reviewed, clicks, other_clicks = self.undo_stack.pop()
        self._checks = None
        for f, lab, box, src, oth, con, who, wbox, uns in saved:
            (self.unseen.add if uns else self.unseen.discard)(f)
            for d, v in ((self.labels, lab), (self.boxes, box), (self.src, src), (self.others_edit, oth),
                         (self.contacts, con), (self.touched_by, who), (self.touched_box, wbox)):
                if v is None:
                    d.pop(f, None)
                else:
                    d[f] = v
        self.reviewed, self.clicks, self.other_clicks = reviewed, clicks, other_clicks
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
                "others": {str(f): self.others_edit[f] for f in sorted(self.others_edit)},
                "contacts": {str(f): self.contacts[f] for f in sorted(self.contacts)},
                "touched_by": {str(f): self.touched_by[f] for f in sorted(self.touched_by)},
                "touched_box": {str(f): self.touched_box[f] for f in sorted(self.touched_box)},
                "unseen": sorted(f for f in self.unseen if f in self.contacts),
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


def still_span(video: str, f: int, box, ball: float, fps: float, n: int, step_s: float = 0.25,
               min_corr: float = 0.7, max_hidden_s: float = 4.0, log=print):
    """Frames (a, b) around f where a resting ball stays at box's spot. One pass over the video,
    looking at that spot every step_s seconds: its picture there vs at frame f (normalized
    correlation, no YOLO). Out of sight for up to max_hidden_s (a player walking in front) and
    back at the spot = still there. Returns (f, f) when it isn't resting (it moved away)."""
    import cv2
    step = max(1, int(round(step_s * fps)))
    r = int(max(8, 0.6 * max(box[2], box[3], ball)))           # patch half-size
    pad = int(0.3 * ball) + 1                                   # allowed wobble (jitter, compression)
    x, y = int(round(box[0])), int(round(box[1]))
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = x - r - pad, y - r - pad, x + r + pad + 1, y + r + pad + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        cap.release()
        return f, f                                             # too close to the edge to compare
    win, g = {}, 0
    while g < n:
        if not cap.grab():
            break
        if (g - f) % step == 0:
            ok, img = cap.retrieve()
            if ok:
                win[g] = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g += 1
    cap.release()
    if f not in win:
        return f, f
    ref = win[f][pad:-pad, pad:-pad]
    if float(ref.std()) < 2.0:                                  # a flat patch: nothing to recognise
        return f, f

    def there(g):
        return float(cv2.matchTemplate(win[g], ref, cv2.TM_CCOEFF_NORMED).max()) >= min_corr

    hidden = int(round(max_hidden_s * fps))
    ends = []
    for d in (+1, -1):
        last, g = f, f + d * step
        while g in win and abs(g - last) <= hidden + step:
            if there(g):
                last = g
            g += d * step
        ends.append(last)
    log(f"[review] resting ball at ({x}, {y}): frames {ends[1]}-{ends[0]}")
    return ends[1], ends[0]


class FullFrames:
    """Full-resolution frames for crop searches, decoded in order (seeking is unreliable on phone
    video) and kept as JPEG, up to KEEP chunks. Going backwards means re-opening the file and
    reading from the start, so that pass also keeps the BACK chunks before the one asked for."""
    CHUNK, KEEP, BACK = 60, 12, 4

    def __init__(self, video: str):
        import cv2
        self.cv2, self.video, self.cap, self.pos, self.cache = cv2, video, None, 0, {}

    def get(self, g: int):
        cv2 = self.cv2
        if g not in self.cache:
            a = (g // self.CHUNK) * self.CHUNK
            lo = a
            if self.cap is None or a < self.pos:
                if self.cap is not None:
                    self.cap.release()
                self.cap, self.pos = cv2.VideoCapture(self.video), 0
                lo = max(0, a - self.BACK * self.CHUNK)    # we pass these anyway: keep them too
            while self.pos < lo:
                if not self.cap.grab():
                    return None
                self.pos += 1
            while self.pos < a + self.CHUNK:
                ok, img = self.cap.read()
                if not ok:
                    break
                self.cache[self.pos] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                self.pos += 1
            chunks = sorted({k // self.CHUNK for k in self.cache}, key=lambda c: abs(c - a // self.CHUNK))
            keep = set(chunks[:self.KEEP])
            self.cache = {k: v for k, v in self.cache.items() if k // self.CHUNK in keep}
        b = self.cache.get(g)
        return None if b is None else cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


class Follower:
    """Finds a confirmed other ball in later frames where the cached full-frame YOLO pass has no
    box: YOLO on a full-res crop around it, else the clicked ball's own image patch (normalized
    correlation), for tiny / resting / blurred balls YOLO can't see at all."""
    PATCH_MIN = 0.75

    def __init__(self, video: str, R, yolo: dict):
        import cv2
        self.cv2, self.R, self.yolo = cv2, R, yolo
        self.full = FullFrames(video)
        self.patch, self.wh, self.calls, self.by_patch, self.origin = None, None, 0, False, None

    def attach(self):
        self.R.crop_find, self.R.start_follow = self.find, self.start
        return self

    def _model(self):
        if "m" not in self.yolo:
            from ultralytics import YOLO
            self.yolo["m"] = YOLO("models/best4.pt") if Path("models/best4.pt").exists() else None
        return self.yolo["m"]

    def start(self, g: int, box):
        """Remember the clicked ball; its look (a square patch at its centre) is cut out only
        if it's needed (decoding a full-res frame is the slow part)."""
        self.patch, self.by_patch, self.origin = None, False, (g, list(box))

    def _cut(self):
        g, box = self.origin
        self.origin = None
        img = self.full.get(g)
        if img is None:
            return
        h = int(max(8, 0.7 * max(box[2], box[3], 0.6 * self.R.ball)))
        x, y = int(round(box[0])), int(round(box[1]))
        p = img[max(0, y - h):y + h, max(0, x - h):x + h]
        if p.size and p.shape[0] == p.shape[1] == 2 * h:
            self.patch = p.copy()
        self.wh = [round(float(box[2]), 1), round(float(box[3]), 1)]

    def crop(self, g: int, spot) -> list:
        try:
            m, img = self._model(), self.full.get(g)
            if m is None or img is None:
                return []
            from core.ball.refine import zoom_detect
            self.calls += 1
            if self.calls % 100 == 0:
                print(f"[review] following the other ball: frame {g}", flush=True)
            return [b[:4] for b in zoom_detect(m, img, spot, int(max(48, 3.0 * self.R.ball)), conf=0.05)]
        except Exception as e:                               # a convenience: never block labelling
            print(f"[review] crop follow: {type(e).__name__}: {e}")
            return []

    def match(self, g: int, spot):
        """The remembered patch near spot in frame g, or None."""
        if self.origin is not None:
            self._cut()
        cv2, p, img = self.cv2, self.patch, self.full.get(g)
        if p is None or img is None:
            return None
        h = p.shape[0] // 2
        r = int(1.5 * self.R.ball) + h
        x, y = int(round(spot[0])), int(round(spot[1]))
        x0, y0 = max(0, x - r), max(0, y - r)
        win = img[y0:y + r, x0:x + r]
        if win.shape[0] <= p.shape[0] or win.shape[1] <= p.shape[1]:
            return None
        _, best, _, (bx, by) = cv2.minMaxLoc(cv2.matchTemplate(win, p, cv2.TM_CCOEFF_NORMED))
        if best < self.PATCH_MIN:
            return None
        return [round(x0 + bx + h, 1), round(y0 + by + h, 1)] + self.wh

    def find(self, g: int, spot) -> list:
        if self.by_patch:          # YOLO couldn't see this ball: keep to the patch (cheap) for this follow
            hit = self.match(g, spot)
            return [hit] if hit else []
        boxes = self.crop(g, spot)
        if any(((b[0] - spot[0]) ** 2 + (b[1] - spot[1]) ** 2) ** 0.5 <= self.R.ball for b in boxes):
            return boxes
        hit = self.match(g, spot)
        self.by_patch = hit is not None
        return boxes + ([hit] if hit else [])


# ── drawing ──────────────────────────────────────────────────────────────────

def draw(cv2, img, R: Review, f: int, scale: float, seg_i: int, speed: float, playing: bool, mouse, msg: str,
         people=None, pick=None):
    """img: canvas = info bar (TOP_H) + picture + timeline (TIMELINE_H). people: the players at
    this frame (core.people.touchers.Players.at), drawn on touch frames; pick: the suggested toucher."""
    H, W = img.shape[:2]
    T = TOP_H
    for p in people or []:                                    # numbered players: cyan; the suggestion: yellow
        x0, y0, x1, y1 = (int(v * scale) for v in p["box"])
        sug = pick is not None and p["id"] == pick["id"]
        mine = f in R.touched_box and R.touched_box[f] == [round(float(v), 1) for v in p["box"]]
        if p["number"] is None and not sug:
            cv2.rectangle(img, (x0, T + y0), (x1, T + y1), (110, 110, 110), 1, cv2.LINE_AA)
            continue
        col_ = (0, 230, 255) if sug else ((60, 220, 60) if mine else (255, 200, 0))
        cv2.rectangle(img, (x0, T + y0), (x1, T + y1), col_, 2 if (sug or mine) else 1, cv2.LINE_AA)
        if p["number"] is not None:
            cv2.putText(img, f"#{p['number']}", (x0, T + max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col_, 1, cv2.LINE_AA)
    lab = R.label(f)
    src = R.src.get(f, "proposal")
    col = {"fixed": (60, 220, 60), "interp": (150, 230, 150), "auto": (255, 220, 0), "proposal": (255, 220, 0)}[src]
    box = R.boxes.get(f) or R.proposal_box(f)                  # this frame's ball size (from nearby YOLO boxes)
    rad = max(6, int(0.5 * max(box) * scale) + 3)             # just outside the ball
    if isinstance(lab, list):
        cx, cy = int(lab[0] * scale), T + int(lab[1] * scale)
        cv2.circle(img, (cx, cy), rad, col, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 2, col, -1, cv2.LINE_AA)       # exact labelled point
    else:
        txt = {"none": "NO BALL", "hidden": "HIDDEN", "skip": "SKIP"}[lab]
        cv2.putText(img, txt, (W // 2 - 60, T + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3, cv2.LINE_AA)
        p = R.proposal(f)
        if lab == "hidden" and isinstance(p, list):      # where the ball would be
            cv2.circle(img, (int(p[0] * scale), T + int(p[1] * scale)), rad, (120, 120, 120), 1, cv2.LINE_AA)
    for boxes, col_, th in ((R.suggestions(f), (150, 150, 150), 1), (R.others(f), (0, 150, 255), 2)):
        for o in boxes:                                        # grey = suggestion, orange = confirmed other ball
            ox_, oy_, hw = int(o[0] * scale), T + int(o[1] * scale), max(5, int(0.5 * max(o[2], o[3]) * scale) + 3)
            cv2.rectangle(img, (ox_ - hw, oy_ - hw), (ox_ + hw, oy_ + hw), col_, th, cv2.LINE_AA)
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
            # where a click would land: crosshair with a gap, so the pixel under it stays visible
            cx, cy = ox + (mx - x0) * 3 + 1, T + 8 + (my - y0) * 3 + 1
            for col_, w_ in (((0, 0, 0), 3), ((0, 255, 255), 1)):          # dark outline, bright line
                for (a1, b1), (a2, b2) in (((cx - 18, cy), (cx - 5, cy)), ((cx + 5, cy), (cx + 18, cy)),
                                          ((cx, cy - 18), (cx, cy - 5)), ((cx, cy + 5), (cx, cy + 18))):
                    cv2.line(img, (a1, b1), (a2, b2), col_, w_, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 1, (0, 255, 255), -1)
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
    shown = {"tracknet": "WASB only"}               # old name in the proposal files: finder point, no YOLO box
    note = msg or (("check: " + ", ".join(shown.get(x, x) for x in d) + "   ") if d and not R.is_edited(f) else "") + lab_txt \
        + (f"   TOUCH: {R.contacts[f]}" + (" (NOT SEEN)" if f in R.unseen else "")
           + (f" by #{R.touched_by[f]}" if f in R.touched_by else "") if f in R.contacts else "")
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
    for g, _kind in R.contacts.items():                      # marked touches: white ticks above the timeline
        cv2.line(img, (xs(g), y0 + 1), (xs(g), y0 + 8), (130, 130, 130) if g in R.unseen else (255, 255, 255), 2)
    for g in R.checks():                                     # to re-check in your labels (b): purple
        cv2.line(img, (xs(g), y0 + 22), (xs(g), y0 + 42), (249, 121, 232), 2)
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
    V = load_json(vp, log=lambda m: print(f"[review] {m}")) if vp.exists() else None
    if vp.exists():
        backup(vp, keep=5, content=V)                       # the labels as opened, one copy per session
    dp = paths.dets(stem)
    R = Review(P, V, json.load(open(dp))["frames"] if dp.exists() else None)
    if test or Path(video).resolve().parent.name == paths.HOLDOUT.name:   # holdout/ clips are always test clips
        R.role = "test"
    from core.people.touchers import Players
    players = Players.load(stem, paths.clip_dir(stem) / "pose.json", log=lambda m: print(f"[review] {m}"))
    if players is None:
        print(f"[review] {stem}: no poses yet (tools.pose): who touched it is typed as a number")
    if players:
        def box_at(g, number, ref):
            """The toucher at frame g: the same number, else the box overlapping theirs most."""
            p = players.with_number(g, number) if number is not None else None
            if p is None:
                from core.people.player_labels import _iou
                c = [(_iou(ref, q["box"]), q) for q in players.at(g)]
                c = [x for x in c if x[0] > 0.3]
                p = max(c, key=lambda x: x[0])[1] if c else None
            return p["box"] if p else None
        R.box_at = box_at
    cam3d, ball3d = None, {}                  # a 3D from an earlier run: the suggestion also checks depth
    b3 = paths.clip_dir(stem) / "ball3d.json"
    cp = Path("data/cameras") / f"{stem}.json"
    if players and b3.exists() and cp.exists():
        from core.geometry.camera import Camera
        cam3d, ball3d = Camera.load(cp), json.load(open(b3)).get("frames", {})
    scale = min(MAX_W / P["width"], (MAX_H - TIMELINE_H - TOP_H) / P["height"], 1.0)
    src = FrameSource(video, scale)

    def save():
        save_json(vp, R.to_json(video), separators=(",", ":"))     # temp file then swap: never half-written

    seg_i = R.next_unreviewed_segment(-1)
    if seg_i is None:
        print(f"[review] {stem}: every segment is already checked (j/k to revisit)")
        seg_i = 0
    f = P["segments"][seg_i]["f0"]
    playing, speed_i = False, SPEEDS.index(1.0)
    state = {"mouse": None, "click": None, "extra": False, "contact": False, "snap": True, "who": None, "who_txt": "",
             "who_sug": None}
    yolo = {}

    def snap_by_crop(img, xy, ball, scale):
        """YOLO on a crop of the displayed frame around a click; the nearest box within SNAP_BW
        ball-widths, in video coordinates, or None. The model loads on first use."""
        try:
            if "m" not in yolo:
                from ultralytics import YOLO
                yolo["m"] = YOLO("models/best4.pt") if Path("models/best4.pt").exists() else None
            if yolo["m"] is None:
                return None
            from core.ball.refine import zoom_detect
            c = (xy[0] * scale, xy[1] * scale)
            half = int(max(48, 5.0 * ball * scale))
            boxes = zoom_detect(yolo["m"], img, c, half, conf=0.05)
        except Exception as e:                               # snapping is a convenience: never block labelling
            print(f"[review] snap: {type(e).__name__}: {e}")
            return None
        best = None
        for b in boxes:
            d = ((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2) ** 0.5 / scale
            if d <= Review.SNAP_BW * ball and (best is None or d < best[0]):
                best = (d, [round(b[0] / scale, 1), round(b[1] / scale, 1), round(b[2] / scale, 1), round(b[3] / scale, 1)])
        return None if best is None else best[1]
    Follower(video, R, yolo).attach()

    def find_still(g, box):
        """A quick look first (frames already in memory, ~1/3 s on each side): if the ball isn't
        at the same spot there, it's moving: no full scan (that takes a few seconds)."""
        a = src.get(g)
        h = int(max(5, 0.6 * max(box[2], box[3], R.ball) * scale))
        x, y = int(round(box[0] * scale)), int(round(box[1] * scale))
        pad = max(2, int(0.3 * R.ball * scale))
        if a is not None and h + pad < x < a.shape[1] - h - pad and h + pad < y < a.shape[0] - h - pad:
            ref = cv2.cvtColor(a[y - h:y + h, x - h:x + h], cv2.COLOR_BGR2GRAY)
            looks = []
            for k in (-20, -10, -5, 5, 10, 20):               # any match = maybe still: do the full scan
                b = src.get(g + k) if 0 <= g + k < R.n else None
                if b is not None and ref.std() >= 2.0:
                    win = cv2.cvtColor(b[y - h - pad:y + h + pad, x - h - pad:x + h + pad], cv2.COLOR_BGR2GRAY)
                    looks.append(float(cv2.matchTemplate(win, ref, cv2.TM_CCOEFF_NORMED).max()) >= 0.7)
            if looks and not any(looks):
                return g, g
        print("[review] checking where this ball rests (a few seconds) ...", flush=True)
        return still_span(video, g, box, R.ball, P["fps"], R.n)
    R.still_find = find_still

    log_path = vp.parent / ".logs" / f"{stem}.log"            # what happened, for tracing problems
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logged = {"msg": None}

    def log(text):
        with open(log_path, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  frame {f:5d}  {text}\n")
    log(f"opened ({sum(1 for v in R.others_edit.values() if v)} frames with other balls)")
    msg, msg_t = "", 0.0

    WIN = "VolleyVision review"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, x, y, flags, param):
        state["mouse"] = (x, y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y, bool(flags & cv2.EVENT_FLAG_SHIFTKEY), bool(flags & cv2.EVENT_FLAG_ALTKEY))
    cv2.setMouseCallback(WIN, on_mouse)

    last_step = time.time()
    while True:
        seg = P["segments"][seg_i]
        src.load(max(0, f - 60), min(R.n - 1, f + 90), pad=0)    # a window around the current frame
        img = src.get(f)
        if img is None:
            img = np.zeros((int(P["height"] * scale), int(P["width"] * scale), 3), np.uint8)
        canvas = np.zeros((TOP_H + img.shape[0] + TIMELINE_H, img.shape[1], 3), np.uint8)
        canvas[TOP_H:TOP_H + img.shape[0]] = img
        if msg and msg != logged["msg"]:
            log(msg)
            logged["msg"] = msg
        show_msg = msg if time.time() - msg_t < 2.5 else ""
        people = players.at(f) if players and (f in R.contacts or state["who"] == f) else None
        cv2.imshow(WIN, draw(cv2, canvas, R, f, scale, seg_i, SPEEDS[speed_i], playing, state["mouse"], show_msg,
                             people, state["who_sug"] if state["who"] == f else None))
        key = cv2.waitKey(5) & 0xFF

        if state["click"] is not None and state["who"] is not None and players:
            x, y, shift, alt = state["click"]                  # who touched it: the player you click
            state["click"] = None
            g = state["who"]
            p = players.under(g, (x / scale, (y - TOP_H) / scale)) if TOP_H <= y < TOP_H + img.shape[0] else None
            if p is None:
                msg, msg_t = f"who touched it at {g}? click inside a player's box", time.time()
            elif p["number"] is None:
                msg, msg_t = (f"who touched it at {g}? that player has no number yet (step 3, tools.label_players);"
                              f" type their number + Enter, or click another"), time.time()
            else:
                R.set_toucher(g, p["number"], p["box"]); save()
                state["who"], state["who_txt"], state["who_sug"] = None, "", None
                msg, msg_t = f"touch at frame {g}: {R.contacts.get(g)} by #{p['number']}", time.time()
        if state["click"] is not None:
            x, y, shift, alt = state["click"]
            state["click"] = None
            if TOP_H <= y < TOP_H + img.shape[0]:
                if shift or state["extra"]:
                    xy = (x / scale, (y - TOP_H) / scale)
                    cb = None
                    rr = max(R.ball, 12.0)
                    if (state["snap"] and R._nearest(R.others(f), xy, rr) is None
                            and R._nearest(R.suggestions(f), xy, rr) is None and R.other_click_target(f, xy) is None):
                        cb = snap_by_crop(img, xy, R.ball, scale)    # nothing cached near the click: try a crop
                    m = R.toggle_other(f, xy, seg["f1"], crop_box=cb, backward=alt); save()
                    msg, msg_t, state["extra"] = m, time.time(), False
                else:
                    xy = (x / scale, (y - TOP_H) / scale)
                    box, how = None, ""
                    if state["snap"]:
                        box = R.snap_target(f, xy)
                        if box is None:                      # no cached box: YOLO on a crop around the click
                            box = snap_by_crop(img, xy, R.ball, scale)
                        if box is not None:
                            how = f"snapped to the ball ({((box[0] - xy[0]) ** 2 + (box[1] - xy[1]) ** 2) ** 0.5:.0f} px from your click)"
                        else:
                            how = "your click (no YOLO box near it)"
                    R.set_ball(f, xy, box); save()
                    if how:
                        msg, msg_t = how, time.time()
                playing = False
            elif y >= TOP_H + img.shape[0]:                    # click on the timeline: jump there
                f = int(round(x / max(canvas.shape[1] - 1, 1) * (R.n - 1)))
                seg_i = R.segment_of(f); playing = False
        if key == 255:
            if playing and time.time() - last_step >= 1.0 / (P["fps"] * SPEEDS[speed_i]):
                last_step = time.time()
                if f < R.n - 1:
                    f += 1
                    seg_i = R.segment_of(f)
                else:
                    playing = False
                    msg, msg_t = "end of clip - Enter / y to accept, q to quit", time.time()
            continue
        k = chr(key) if key < 128 else ""
        if state["who"] is not None:                            # optional: who touched it (number + Enter)
            g = state["who"]
            if k.isdigit() and len(state["who_txt"]) < 2:
                state["who_txt"] += k
                msg, msg_t = f"who touched it at {g}? #{state['who_txt']}  (Enter to keep, Backspace, any other key: skip)", time.time()
                continue
            if key in (8, 127) and state["who_txt"]:
                state["who_txt"] = state["who_txt"][:-1]
                msg, msg_t = f"who touched it at {g}? #{state['who_txt']}", time.time()
                continue
            txt, sug = state["who_txt"], state["who_sug"]
            state["who"], state["who_txt"], state["who_sug"] = None, "", None
            if key in (13, 10) and txt and 1 <= int(txt) <= 99:
                p = players.with_number(g, int(txt)) if players else None
                R.set_toucher(g, int(txt), p["box"] if p else None); save()
                msg, msg_t = (f"touch at frame {g}: {R.contacts.get(g)} by #{int(txt)}"
                              + ("" if p or not players else "  (not in the player boxes here: kept as typed)")), time.time()
                continue
            if key in (13, 10) and not txt and sug is not None:     # Enter: the suggested player
                R.set_toucher(g, sug["number"], sug["box"]); save()
                msg, msg_t = f"touch at frame {g}: {R.contacts.get(g)} by #{sug['number']}", time.time()
                continue
            msg, msg_t = f"touch at frame {g}: {R.contacts.get(g)} (who: not given)", time.time()
            if key in (13, 10):
                continue
        if state["contact"]:                                    # waiting for the touch type after "m"
            state["contact"] = False
            if k in Review.CONTACT_TYPES:
                R.set_contact(f, Review.CONTACT_TYPES[k]); save()
                msg, msg_t = f"touch at frame {f}: {Review.CONTACT_TYPES[k]}", time.time()
                if Review.CONTACT_TYPES[k] != "dead":
                    state["who"], state["who_txt"] = f, ""
                    lab = R.label(f)
                    xy = lab if isinstance(lab, list) else R.proposal(f)
                    X3 = (ball3d.get(str(f)) or {}).get("xyz")
                    sug = players.suggest(f, xy, cam3d if X3 else None, X3) if players and isinstance(xy, list) else None
                    state["who_sug"] = sug
                    msg = (f"touch at frame {f}: {Review.CONTACT_TYPES[k]} - who touched it? "
                           + (f"#{sug['number']} (yellow): Enter accepts; " if sug else "")
                           + ("click a player, or " if players else "") + "number + Enter; any other key: skip")
            else:
                msg, msg_t = "no touch marked (type keys: 1 serve 2 pass 3 set 4 hit 5 block 6 dig 0 other 9 rally over)", time.time()
            continue
        if k == "m":
            playing = False
            if f in R.contacts:
                R.set_contact(f, None); save()
                msg, msg_t = f"touch at frame {f} removed", time.time()
            elif (run := R.extend_touch(f)) is not None:         # next to a marked touch: it joins it
                save()
                who = R.touched_by.get(f)
                msg, msg_t = (f"touch {R.contacts[f]} now frames {run[0]}-{run[-1]}"
                              + (f" by #{who}" if who is not None else "")), time.time()
            else:
                state["contact"] = True
                msg, msg_t = "touch type? 1 serve 2 pass 3 set 4 hit 5 block 6 dig 0 other 9 rally over", time.time()
            continue
        if k == "v":
            playing = False
            on = R.toggle_unseen(f)
            if on is None:
                msg, msg_t = "v: no touch here (mark it with m first, then v: not seen)", time.time()
            else:
                save()
                msg, msg_t = (f"touch at {f}: {R.contacts[f]}" + (" NOT SEEN (best-guess frame)" if on else " seen")), time.time()
            continue
        if k == "b":
            g = R.next_check(f); playing = False
            if g is None:
                msg, msg_t = "nothing to re-check in your labels", time.time()
            else:
                f = g; seg_i = R.segment_of(f)
                msg, msg_t = f"re-check {f}: {R.checks()[g]}  ({len(R.checks())} to look at)", time.time() + 3
            continue
        if k == "M":
            g = R.next_contact(f); playing = False
            if g is None:
                msg, msg_t = "no more marked touches", time.time()
            else:
                f = g; seg_i = R.segment_of(f)
            continue
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
        elif k == "N":
            m = R.set_none_run(f); save(); playing = False
            msg, msg_t = f"no ball in {m} frame{'s' if m != 1 else ''} (this run of {R.P['frames'][f]['state']} proposals)", time.time()
        elif k == "h":
            R.set_mark(f, "hidden"); save()
        elif k == "x":
            R.set_mark(f, "skip"); save()
        elif k == "c":
            R.clear(f); save()
        elif k == "o":
            g = R.next_suggestion(f); playing = False
            if g is None:
                msg, msg_t = "no more grey suggestions in this clip", time.time()
            else:
                f = g; seg_i = R.segment_of(f)
        elif k == "z":
            state["snap"] = R.snap = not state["snap"]
            msg, msg_t = ("clicks snap to the ball (YOLO box)" if state["snap"] else "clicks stay exactly where you click"), time.time()
        elif k == "e":
            state["extra"] = not state["extra"]; playing = False
            msg, msg_t = ("next click: another ball (on a square: remove it)" if state["extra"] else "back to normal clicks"), time.time()
        elif k == "r":                                         # the orange box under the mouse, this frame only
            if state["mouse"] is not None and TOP_H <= state["mouse"][1] < TOP_H + img.shape[0]:
                mx, my = state["mouse"]
                m = R.remove_other_here(f, (mx / scale, (my - TOP_H) / scale)); save()
            else:
                m = "point the mouse at an orange box, then press r"
            msg, msg_t, playing = m, time.time(), False
        elif k == "R":                                         # every orange box, this frame only
            m = R.clear_others_here(f); save()
            msg, msg_t, playing = m, time.time(), False
        elif k == "g":
            m = R.fill_others(); save()
            msg, msg_t = (f"other ball: filled {m} frames between the keyframes" if m
                          else "shift+click the moving ball in two or more frames first"), time.time()
        elif k == "F":
            R.follow = not R.follow
            msg, msg_t = ("F on: shift+click also follows with YOLO on crops where YOLO lost the ball (slow)"
                          if R.follow else "F off: shift+click follows YOLO's saved boxes + resting balls (fast)"), time.time()
        elif k == "i":
            m = R.interpolate(); save()
            msg, msg_t = (f"filled {m} frames" if m else "click the ball in two frames first"), time.time()
        elif k == "I":
            m = R.interpolate_all(seg["f0"], seg["f1"]); save()
            msg, msg_t = (f"filled {m} frames between your clicks" if m else "click the ball in at least two frames first"), time.time()
        elif key in (13, 10):
            m = R.accept(seg["f0"], seg["f1"]); save()
            nxt = R.next_unreviewed_segment(seg_i)
            msg, msg_t = f"accepted segment {seg_i + 1} ({m} frames from the proposal)", time.time()
            if nxt is None:
                msg = "all segments checked - q to quit"; playing = False
            elif nxt != seg_i + 1 or not playing:          # keep playing straight on into the next segment
                seg_i = nxt; f = P["segments"][seg_i]["f0"]
        elif k == "y":
            m = sum(R.accept(s["f0"], s["f1"]) for s in P["segments"]); save()
            msg, msg_t, playing = f"accepted the whole clip ({m} frames from the proposal) - q to quit", time.time(), False
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
            if Path(video).resolve().parent.name != paths.HOLDOUT.name:   # holdout/ clips stay test
                R.role = "test" if R.role == "train" else "train"; save()
            msg, msg_t = f"clip role: {R.role}", time.time()
        elif k == "?":
            print(__doc__)
            msg, msg_t = "help printed in the terminal", time.time()
        elif k == "q" or key == 27:
            save()
            log(f"saved and quit ({sum(1 for v in R.others_edit.values() if v)} frames with other balls)")
            break
        seg_i = R.segment_of(f) if not (P["segments"][seg_i]["f0"] <= f <= P["segments"][seg_i]["f1"]) else seg_i
    cv2.destroyAllWindows()
    done, tot = R.progress()
    print(f"[review] {stem}: checked {done}/{tot} frames, {sum(1 for f in R.src if R.is_edited(f))} edited  →  {vp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--test", action="store_true", help="mark as a test clip (kept out of training)")
    a = ap.parse_args()
    from core.common import paths
    from tools.todo import old_data_clips
    old = old_data_clips(paths.YOLO_DATASETS / "volleyball_combined")
    for v in a.videos:
        stem = Path(v).stem
        if stem in old and not (Path("data/verified") / f"{stem}.json").exists():
            ans = input(f"[review] {stem} is already in the old training data (volleyball_combined); "
                        f"labeling it again counts it twice. Review anyway? [y/N] ")
            if ans.strip().lower() != "y":
                continue
        review(v, a.test)


if __name__ == "__main__":
    main()
