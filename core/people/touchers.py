"""Who touched the ball, from the pose tracking and your player numbers (for tools.review).

The numbered players at a frame (output/<clip>/pose.json + data/player_ids/<clip>.json), and the
one most likely to have touched the ball: the numbered player whose wrist (or box) is nearest the
ball. You confirm it in review, so "who touched it" always names someone in the player labels.
"""
from __future__ import annotations

import json
from pathlib import Path

WRISTS = (9, 10)            # COCO body points
KP_MIN = 0.3                # a wrist this sure is used; otherwise the box
NEAR_BOX_H = 0.6            # the toucher's wrist (or box) within this many of their box heights


def _box_dist(b, xy) -> float:
    dx = max(b[0] - xy[0], 0.0, xy[0] - b[2])
    dy = max(b[1] - xy[1], 0.0, xy[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


class Players:
    def __init__(self, frames: dict, labels=None):
        self.frames = {int(k): v for k, v in frames.items()}
        self.labels = labels                      # core.people.player_labels.PlayerLabels or None

    @classmethod
    def load(cls, stem: str, pose_path, log=print) -> Players | None:
        if not Path(pose_path).exists():
            return None
        P = json.load(open(pose_path))
        from core.people.player_labels import PlayerLabels
        return cls(P.get("frames", P), PlayerLabels.load(stem, pose_path, log=log))

    def at(self, f: int) -> list[dict]:
        """[{"id", "box", "kp", "number"}] at frame f (number None when you haven't numbered them)."""
        out = []
        for p in self.frames.get(int(f), []):
            n = self.labels.number(p["id"], f) if self.labels else None
            out.append({"id": p["id"], "box": p["box"], "kp": p.get("kp"),
                        "number": n if isinstance(n, int) else None})
        return out

    def dist(self, p: dict, xy) -> float:
        """Pixels from the ball to this player's nearest sure wrist (else their box)."""
        ws = [k for i, k in enumerate(p.get("kp") or []) if i in WRISTS and k[2] >= KP_MIN]
        if ws:
            return min(((k[0] - xy[0]) ** 2 + (k[1] - xy[1]) ** 2) ** 0.5 for k in ws)
        return _box_dist(p["box"], xy)

    def suggest(self, f: int, xy) -> dict | None:
        """The numbered player most likely to have touched the ball at (x, y), or None."""
        best = None
        for p in self.at(f):
            if p["number"] is None:
                continue
            h = max(1.0, p["box"][3] - p["box"][1])
            d = self.dist(p, xy) / h
            if d <= NEAR_BOX_H and (best is None or d < best[0]):
                best = (d, p)
        return best[1] if best else None

    def under(self, f: int, xy) -> dict | None:
        """The player whose box is under a click (the smallest box, when boxes overlap)."""
        hit = [p for p in self.at(f) if p["box"][0] <= xy[0] <= p["box"][2] and p["box"][1] <= xy[1] <= p["box"][3]]
        return min(hit, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])) if hit else None

    def with_number(self, f: int, n: int) -> dict | None:
        return next((p for p in self.at(f) if p["number"] == n), None)
