"""Who touched the ball: ONE rule, used by the pipeline and by review.

Players come from the pose tracking (output/<clip>/pose.json) with your numbers
(data/player_ids/<clip>.json). The toucher is the player whose wrist (or box) is nearest the ball
at the touch (+-2 frames), within NEAR_BOX_H of their height; with the 3D, a player much nearer or
farther from the camera than the ball can't be the one (DEPTH_TOL).

  pipeline   tools.ball3d_run: rank() at every 3D touch -> output/<clip>/touches.json
  review     tools.review: suggest() -> the yellow suggestion you confirm or correct; what you
             confirm (touched_by + touched_box) scores the pipeline (evaluation.whotouched)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WRISTS = (9, 10)            # COCO body points
KP_MIN = 0.3                # a wrist this sure is used; otherwise the box
NEAR_BOX_H = 0.6            # the toucher's wrist (or box) within this many of their box heights
AROUND = 2                  # frames each side of the touch
DEPTH_TOL = 0.3             # with the 3D: the toucher's distance from the camera within 30% of the ball's
DEPTH_TOL_M = 2.5           # ... or this many metres, whichever is larger
BRIDGE_MAX = 30             # frames (0.5 s at 60 fps): a numbered player missed for at most this long ...
BRIDGE_MOVE = 2.0           # ... who moved at most this many box heights meanwhile is bridged across


def _box_dist(b, xy) -> float:
    dx = max(b[0] - xy[0], 0.0, xy[0] - b[2])
    dy = max(b[1] - xy[1], 0.0, xy[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


class Players:
    def __init__(self, frames: dict, labels=None):
        self.frames = {int(k): v for k, v in frames.items()}
        self.labels = labels                      # core.people.player_labels.PlayerLabels or None
        self.bridged = self._bridge() if labels else {}

    def _bridge(self) -> dict:
        """People don't teleport: a numbered player the detector misses for a moment (hidden behind
        someone, blurred) is put in between where they were seen just before and just after, if
        that's at most BRIDGE_MAX frames and BRIDGE_MOVE box heights apart. frame -> [player]."""
        seen: dict = {}
        for f in sorted(self.frames):
            nums = [self.labels.number(p["id"], f) for p in self.frames[f]]
            for p, n in zip(self.frames[f], nums):
                if isinstance(n, int) and nums.count(n) == 1:      # the same number twice: a label slip
                    seen.setdefault(n, []).append((f, p))
        out: dict = {}
        for n, L in seen.items():
            for (a, pa), (b, pb) in zip(L[:-1], L[1:]):
                if b - a <= 1 or b - a > BRIDGE_MAX:
                    continue
                A, B = pa["box"], pb["box"]
                h = 0.5 * ((A[3] - A[1]) + (B[3] - B[1]))
                move = (((A[0] + A[2]) - (B[0] + B[2])) ** 2 + ((A[1] + A[3]) - (B[1] + B[3])) ** 2) ** 0.5 / 2
                if move > BRIDGE_MOVE * max(h, 1.0):
                    continue
                for g in range(a + 1, b):
                    u = (g - a) / (b - a)
                    box = [A[i] + u * (B[i] - A[i]) for i in range(4)]
                    out.setdefault(g, []).append({"id": pa["id"], "box": box, "kp": None, "number": n,
                                                  "bridged": True})
        return out

    @classmethod
    def load(cls, stem: str, pose_path, log=print) -> Players | None:
        if not Path(pose_path).exists():
            return None
        P = json.load(open(pose_path))
        from core.people.player_labels import PlayerLabels
        return cls(P.get("frames", P), PlayerLabels.load(stem, pose_path, log=log))

    def at(self, f: int) -> list[dict]:
        """[{"id", "box", "kp", "number"}] at frame f (number None when you haven't numbered them);
        numbered players missed for a moment are filled in (bridged: True; no body points)."""
        out = []
        for p in self.frames.get(int(f), []):
            n = self.labels.number(p["id"], f) if self.labels else None
            out.append({"id": p["id"], "box": p["box"], "kp": p.get("kp"),
                        "number": n if isinstance(n, int) else None})
        have = {p["number"] for p in out}
        out += [p for p in self.bridged.get(int(f), []) if p["number"] not in have]
        return out

    def dist(self, p: dict, xy) -> float:
        """Pixels from the ball to this player's nearest sure wrist (else their box)."""
        ws = [k for i, k in enumerate(p.get("kp") or []) if i in WRISTS and k[2] >= KP_MIN]
        if ws:
            return min(((k[0] - xy[0]) ** 2 + (k[1] - xy[1]) ** 2) ** 0.5 for k in ws)
        return _box_dist(p["box"], xy)

    def rank(self, f: int, xy, cam=None, ball_xyz=None, numbered: bool = False) -> list[dict]:
        """Players who could have touched the ball at (x, y) at frame f, nearest first: each player's
        closest approach over f +- AROUND, in their box heights ("d"), kept when <= NEAR_BOX_H. With
        cam + ball_xyz (the 3D ball), players whose feet are much nearer or farther from the camera
        than the ball are dropped. numbered: only players you numbered."""
        best: dict = {}
        for g in range(f - AROUND, f + AROUND + 1):
            for p in self.at(g):
                if numbered and p["number"] is None:
                    continue
                h = max(1.0, p["box"][3] - p["box"][1])
                d = self.dist(p, xy) / h
                key = p["number"] if p["number"] is not None else ("id", p["id"])
                if d <= NEAR_BOX_H and (key not in best or d < best[key]["d"]):
                    best[key] = {**p, "d": round(d, 3), "frame": g}
        out = sorted(best.values(), key=lambda p: p["d"])
        if cam is not None and ball_xyz is not None:
            out = self._depth_ok(out, cam, ball_xyz)
        if any(not p.get("bridged") for p in out):          # a bridged player only when nobody seen is there
            out = [p for p in out if not p.get("bridged")]
        return out

    def _depth_ok(self, out, cam, ball_xyz):
        """Players whose feet are about as far from the camera as the ball (+ where they stand)."""
        zb = float(cam.depth(np.asarray([ball_xyz], float))[0])
        keep = []
        for p in out:
            b = p["box"]
            foot = cam.floor_point([((b[0] + b[2]) / 2, b[3])], 0.0)[0]
            zp = float(cam.depth(np.asarray([[foot[0], foot[1], 0.0]]))[0])
            if abs(zp - zb) <= max(DEPTH_TOL * zb, DEPTH_TOL_M):
                keep.append({**p, "feet": [round(float(foot[0]), 2), round(float(foot[1]), 2)]})
        return keep

    def suggest(self, f: int, xy, cam=None, ball_xyz=None) -> dict | None:
        """The numbered player most likely to have touched the ball at (x, y), or None."""
        r = self.rank(f, xy, cam, ball_xyz, numbered=True)
        return r[0] if r else None

    def under(self, f: int, xy) -> dict | None:
        """The player whose box is under a click (the smallest box, when boxes overlap)."""
        hit = [p for p in self.at(f) if p["box"][0] <= xy[0] <= p["box"][2] and p["box"][1] <= xy[1] <= p["box"][3]]
        return min(hit, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])) if hit else None

    def with_number(self, f: int, n: int) -> dict | None:
        return next((p for p in self.at(f) if p["number"] == n), None)
