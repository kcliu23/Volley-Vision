"""Move output/ from the old flat layout into per-clip folders (see core/paths.py).

    python -m tools.organize_output            # show what would move (nothing changes)
    python -m tools.organize_output --apply    # do it

    output/detections/<clip>_dets.json         -> output/<clip>/cache/dets.json
    output/detections/<clip>_tracknet*.json    -> output/<clip>/cache/tracknet*.json
    output/<clip>_track.json (vv-track/2)      -> output/<clip>/track.json
    output/<clip>_contacts.json                -> output/<clip>/contacts.json
    output/<clip>_tracked.mp4                  -> output/<clip>/tracked.mp4  (if the clip has a current track)
    output/dataset_labels.json                 -> training/tracknet/dataset_labels.json
    output/tracknet_data/                      -> training/tracknet/datasets/tracknet_v1/
    output/dev/sync*                           -> output/archive/
    everything else from older versions        -> output/archive/   (touches, link, clip.json, posters, ...)

Nothing is deleted and nothing is overwritten: if a destination exists, the file goes
to output/archive/ instead.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402

CLIP = r"(test\w*?|[A-Za-z0-9][\w\-]*?)"


def plan(root: Path) -> list[tuple[Path, Path, str]]:
    moves: list[tuple[Path, Path, str]] = []
    arch = paths.archive(root)

    def to_archive(p: Path, why: str):
        moves.append((p, arch / p.relative_to(root), why))

    det = root / "detections"
    if det.is_dir():
        for p in sorted(det.iterdir()):
            m = re.fullmatch(r"(.+)_dets\.json", p.name)
            t = re.fullmatch(r"(.+?)_tracknet(?:_(\w+))?\.json", p.name)
            if m:
                moves.append((p, paths.dets(m.group(1), root), "cache"))
            elif t:
                moves.append((p, paths.tracknet(t.group(1), root, t.group(2)), "cache"))
            else:
                to_archive(p, "unknown cache file")

    current = set()
    for p in root.glob("*_track.json"):
        try:
            if json.load(open(p)).get("schema") == "vv-track/2":
                current.add(p.name[:-len("_track.json")])
        except Exception:
            pass

    keep_dirs = {"detections", "dev", "archive"}
    for p in sorted(root.iterdir()):
        if p.is_dir():
            if p.name in keep_dirs or (p / "cache").is_dir() or (p / "track.json").exists():
                continue
            if p.name == "tracknet_data":
                moves.append((p, paths.TRACKNET_DATASETS / "tracknet_v1", "training"))
                continue
            to_archive(p, "old folder")
            continue
        n = p.name
        if n.endswith("_track.json"):
            stem = n[:-len("_track.json")]
            if stem in current:
                moves.append((p, paths.track(stem, root), "result"))
            else:
                to_archive(p, "old track format")
        elif n.endswith("_contacts.json"):
            moves.append((p, paths.contacts(n[:-len("_contacts.json")], root), "result"))
        elif n.endswith("_tracked.mp4"):
            stem = n[:-len("_tracked.mp4")]
            if stem in current:
                moves.append((p, paths.video(stem, root), "result"))
            else:
                to_archive(p, "render from an older pipeline")
        elif n == "dataset_labels.json":
            moves.append((p, paths.TRAINING / "tracknet" / "dataset_labels.json", "training"))
        else:
            to_archive(p, "older version")

    dev = paths.dev(root)
    if dev.is_dir():
        for p in sorted(dev.glob("sync*")):
            moves.append((p, arch / p.name, "file-transfer folder"))
    # destinations that already exist go to the archive instead (never overwrite)
    out = []
    for src, dst, why in moves:
        if dst.exists():
            dst = arch / "duplicates" / src.relative_to(root)
            why += " (destination existed)"
        out.append((src, dst, why))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(paths.OUTPUT))
    ap.add_argument("--apply", action="store_true", help="move the files (default: only show the plan)")
    ap.add_argument("--verbose", action="store_true", help="list every file")
    a = ap.parse_args()
    root = Path(a.root)
    moves = plan(root)
    from collections import Counter
    by = Counter(w for _, _, w in moves)
    clips = sorted({d.relative_to(root).parts[0] for _, d, w in moves
                    if d.is_relative_to(root) and not d.is_relative_to(paths.archive(root))})
    print(f"{len(moves)} items to move: " + ", ".join(f"{v} {k}" for k, v in by.most_common()))
    print(f"clip folders: {len(clips)}  ({', '.join(clips[:12])}{' ...' if len(clips) > 12 else ''})")
    for s, d, w in moves if a.verbose else moves[:15]:
        print(f"  {s}  ->  {d}   [{w}]")
    if not a.verbose and len(moves) > 15:
        print(f"  ... ({len(moves) - 15} more; --verbose lists all)")
    if not a.apply:
        print("\nnothing moved: run with --apply to do it")
        return
    for s, d, _ in moves:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    for leftover in (root / "detections",):
        try:
            leftover.rmdir()
        except OSError:
            pass
    print(f"moved {len(moves)} items")


if __name__ == "__main__":
    main()
