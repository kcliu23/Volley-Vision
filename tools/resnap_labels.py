"""Snap hand-clicked game-ball labels to the centre of a nearby YOLO box (more exact than a click).

    python -m tools.resnap_labels                    # report only: how many clicks would move, how far
    python -m tools.resnap_labels --apply            # write the snapped labels (a backup is kept)
    python -m tools.resnap_labels --apply test61 test62

Only frames you clicked or filled yourself (src "fixed" / "interp") are touched; accepted proposals
already sit on YOLO boxes. A click snaps to the nearest YOLO box (autolabel's full-frame pass,
output/<clip>/cache/dets.json) whose centre is within --max-bw ball-widths (default 0.4); otherwise it stays.
Before writing, each clip's file is copied to data/verified/backup/<clip>.<time>.json.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time

from core import paths
from core.safe_io import save_json


def resnap(V: dict, dets_frames: list, max_bw: float):
    labels, src = V["labels"], V.get("src", {})
    boxes = V.setdefault("boxes", {})
    sizes = sorted(max(b) for b in boxes.values()) or [V.get("ball_px", 30)]
    ball = sizes[len(sizes) // 2]
    moved = []
    for k, v in labels.items():
        if not isinstance(v, list) or src.get(k) not in ("fixed", "interp"):
            continue
        f = int(k)
        if f >= len(dets_frames):
            continue
        best = None
        for b in dets_frames[f]:
            if b[4] < 0.05:
                continue
            d = math.hypot(b[0] - v[0], b[1] - v[1])
            if d <= max_bw * ball and (best is None or d < best[0]):
                best = (d, b)
        if best is not None and best[0] > 0.5:
            d, b = best
            labels[k] = [round(b[0], 1), round(b[1], 1)]
            boxes[k] = [round(b[2], 1), round(b[3], 1)]
            moved.append(d)
    return moved, ball


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="*", help="clip names (default: every checked clip)")
    ap.add_argument("--apply", action="store_true", help="write the changes (default: report only)")
    ap.add_argument("--max-bw", type=float, default=0.4,
                    help="snap distance limit, ball-widths (0.4 ~ 13 px: click wobble, not a different object)")
    a = ap.parse_args()
    stems = a.clips or [p.stem for p in sorted(paths.VERIFIED.glob("*.json"))]
    tot_n = tot_clicks = 0
    for s in stems:
        vp, dp = paths.verified(s), paths.dets(s)
        if not vp.exists() or not dp.exists():
            print(f"  {s}: {'no labels' if not vp.exists() else 'no YOLO cache (run tools.autolabel)'}, skipped")
            continue
        V = json.load(open(vp))
        clicks = sum(1 for k, v in V["labels"].items() if isinstance(v, list) and V.get("src", {}).get(k) in ("fixed", "interp"))
        moved, ball = resnap(V, json.load(open(dp))["frames"], a.max_bw)
        tot_n += len(moved); tot_clicks += clicks
        if moved:
            ms = sorted(moved)
            print(f"  {s}: {len(moved)}/{clicks} hand-placed frames snap, median {ms[len(ms) // 2]:.1f} px "
                  f"(max {ms[-1]:.1f}; ball {ball:.0f} px)")
            if a.apply:
                bk = paths.ensure(paths.VERIFIED / "backup" / f"{s}.{time.strftime('%Y%m%d-%H%M%S')}.json")
                shutil.copy(vp, bk)
                save_json(vp, V, separators=(",", ":"))
    print(f"{tot_n}/{tot_clicks} hand-placed frames {'snapped' if a.apply else 'would snap'} to a YOLO box"
          + ("" if a.apply else "  (report only: --apply to write; backups are kept)"))


if __name__ == "__main__":
    main()
