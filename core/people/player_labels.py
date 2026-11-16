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
            # made on another pose.json run (tracker ids mean other people now): carry the numbers
            # over by box overlap, as tools.label_players does when you open it (the file on disk
            # is not changed; open the label tool to save the carried-over numbers)
            pose = json.load(open(pose_path))
            pose = pose.get("frames", pose)
            names, cuts, _, report = remap_by_boxes(L.get("frames", {}), pose)
            log(f"player labels for {stem}: made on another pose run, carried over by box overlap ({report})")
            return cls(names, cuts)
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
