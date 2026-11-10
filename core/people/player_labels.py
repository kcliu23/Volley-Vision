"""Your hand player numbers (tools.label_players) as a lookup for the rest of the pipeline.

    ids = PlayerLabels.load(stem, pose_path)     # None if the clip has no labels, or they were made
                                                 # on a different pose.json run (ids mean other people)
    ids.number(tracker_id, frame)  -> 1..12, "x" (not a player), or None (not named)
    ids.side(tracker_id, frame)    -> -1 near (1-6), +1 far (7-12), or None

Numbers 1-6 are the near side (camera side), 7-12 the far side, as in the labeling tool. Until
re-ID names players by itself, the 3D step uses these on clips that have them.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.common.safe_io import file_key

LABELS = Path("data/player_ids")


class PlayerLabels:
    def __init__(self, pieces: dict, cuts: dict):
        self.pieces = pieces
        self.cuts = {int(k): sorted(v) for k, v in cuts.items()}

    @classmethod
    def load(cls, stem: str, pose_path, log=print) -> PlayerLabels | None:
        p = LABELS / f"{stem}.json"
        if not p.exists():
            return None
        L = json.load(open(p))
        if L.get("pose_key") not in (None, file_key(pose_path)):
            log(f"player labels for {stem} were made on another pose.json run: not used "
                f"(open tools.label_players once to carry them over)")
            return None
        return cls(L.get("pieces", {}), L.get("cuts", {}))

    def piece(self, tid: int, f: int) -> str:              # the same rule as tools.label_players
        c = [s for s in self.cuts.get(int(tid), []) if s <= f]
        return f"{tid}@{max(c)}" if c else str(tid)

    def number(self, tid: int, f: int):
        return self.pieces.get(self.piece(tid, f))

    def side(self, tid: int, f: int) -> int | None:
        n = self.number(tid, f)
        if isinstance(n, int) and 1 <= n <= 12:
            return -1 if n <= 6 else 1
        return None
