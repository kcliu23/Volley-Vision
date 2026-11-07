"""How good is re-ID? Scores core.reid's player groups against your labels (tools.label_players).

    python -m tools.reid_eval                      # every clip with labels in data/player_ids/
    python -m tools.reid_eval --clips test76

For every person-frame you named (on-court players only):
  purity       of the frames re-ID puts in one group, the share that really is one player (the
               group's majority): low = it joins different people (the dangerous mistake)
  pieces       how many groups each real player ends up split into: 1 is perfect
  side right   the group's side of the net matches the player's (1-6 near, 7-12 far): this is
               what the 3D uses at touches
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


from core import paths
from core.safe_io import file_key

LABELS = Path("data/player_ids")


def score(stem: str, video: str, output: str):
    from core.camera import Camera
    from core.reid import build_ids
    import cv2
    L = json.load(open(LABELS / f"{stem}.json"))
    pj = paths.clip_dir(stem, output) / "pose.json"
    if L.get("pose_key") not in (None, file_key(pj)):
        print(f"  {stem}: pose.json changed since labeling; open tools.label_players once to carry the labels over")
        return None
    pose = json.load(open(pj))["frames"]
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS) or 60.0
    ids = build_ids(pose, video, cam, fps)
    cuts = {int(k): v for k, v in L.get("cuts", {}).items()}

    def piece(tid, f):
        c = [s for s in cuts.get(tid, []) if s <= f]
        return f"{tid}@{max(c)}" if c else str(tid)

    pairs = []                                             # (group, true player, group side)
    for f, people in pose.items():
        for p in people:
            if p["id"] < 0:
                continue
            v = L["pieces"].get(piece(p["id"], int(f)))
            if v is None or v == "x" or p["id"] not in ids:
                continue
            pairs.append((ids[p["id"]]["player"], v, ids[p["id"]]["side"]))
    if not pairs:
        return None
    by_group = defaultdict(Counter)
    by_player = defaultdict(set)
    side_ok = 0
    for g, v, s in pairs:
        by_group[g][v] += 1
        by_player[v].add(g)
        side_ok += s == (-1 if v <= 6 else 1)
    purity = sum(c.most_common(1)[0][1] for c in by_group.values()) / len(pairs)
    return {"frames": len(pairs), "players": len(by_player), "purity": purity,
            "pieces": float(np.mean([len(g) for g in by_player.values()])), "side": side_ok / len(pairs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    stems = sorted(p.stem for p in LABELS.glob("*.json") if ".backup" not in p.name and not p.name.endswith(".tmp") and (not a.clips or p.stem in a.clips))
    for stem in stems:
        video = next((str(p) for d in ("videos", "holdout") for p in [Path(d) / f"{stem}.mov"] if p.exists()), None)
        r = score(stem, video, a.output) if video else None
        if r:
            print(f"{stem:10s} {r['players']:2d} players, {r['frames']:5d} person-frames | purity {r['purity']:.1%} | "
                  f"pieces per player {r['pieces']:.1f} | side right {r['side']:.1%}")


if __name__ == "__main__":
    main()
