"""Which clips still need labeling? One line per clip, grouped by status.

    python -m tools.todo            # summary + the clips to review next
    python -m tools.todo --all      # every clip with its status

Status of each clip in videos/ and holdout/:
  verified    checked in tools.review (data/verified/<clip>.json)
  old data    already in the old training data (volleyball_combined / tracknet_v1). Skip it:
              re-labeling would count it twice. (export_training skips those frames anyway.)
  dev set     has data/gt labels: the old dev scoring set, kept out of training
  to review   has autolabel proposals, not checked yet  <- your queue, most-flagged first
  new         no proposals yet: run tools.autolabel first
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


from core import paths

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
OLD_RE = re.compile(r"^((?:test|holdout)\d*)_\d+_")


def old_data_clips(combined: Path) -> set:
    """Clip names in the old YOLO data (Roboflow names: <clip>_<frame>_jpg.rf.<hash>.jpg)."""
    out = set()
    for split in ("train", "valid", "test"):
        d = combined / split / "images"
        if d.is_dir():
            for p in d.iterdir():
                m = OLD_RE.match(p.name)
                if m:
                    out.add(m.group(1))
    return out


def natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def status() -> dict:
    root = Path(".")
    vids = {}
    for folder in ("videos", "holdout"):
        for p in sorted((root / folder).glob("*")):
            if p.suffix.lower() in VIDEO_EXT:
                vids[p.stem] = p
    old = old_data_clips(paths.YOLO_DATASETS / "volleyball_combined")
    gt = {p.name.replace("_gt.json", "") for p in Path("data/gt").glob("*_gt.json")}
    out = {}
    for stem in sorted(vids, key=natural):
        prop = Path("data/proposals") / f"{stem}.json"
        if (Path("data/verified") / f"{stem}.json").exists():
            st, info = "verified", ""
        elif stem in old:
            st, info = "old data", ""
        elif stem in gt:
            st, info = "dev set", ""
        elif prop.exists():
            p = json.load(open(prop))
            n = max(p.get("n") or len(p["frames"]), 1)
            flagged = sum(1 for fr in p["frames"] if fr["doubt"])
            st, info = "to review", f"{flagged / n:.0%} flagged, {n} frames"
        else:
            st, info = "new", ""
        out[stem] = (st, info, str(vids[stem]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="list every clip with its status")
    a = ap.parse_args()
    s = status()
    groups = {}
    for stem, (st, info, path) in s.items():
        groups.setdefault(st, []).append((stem, info, path))
    for st in ("verified", "old data", "dev set", "to review", "new"):
        g = groups.get(st, [])
        print(f"{st:10s} {len(g):3d}  " + (" ".join(x[0] for x in g) if st != "to review" else ""))
    q = sorted(groups.get("to review", []), key=lambda x: -float(x[1].split("%")[0]))
    if q:
        print("\nTo review, most flagged first (hard clips teach the most):")
        for stem, info, path in (q if a.all else q[:10]):
            print(f"  python -m tools.review {path:28s} # {info}")
        if len(q) > 10 and not a.all:
            print(f"  ... {len(q) - 10} more (--all)")
    if groups.get("new"):
        print("\nNo proposals yet:  python -m tools.autolabel " + " ".join(x[2] for x in groups["new"]))


if __name__ == "__main__":
    main()
