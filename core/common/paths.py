"""Where every file lives. All code builds paths through here, so the layout can change in one place.

    output/<clip>/tracked.mp4          rendered video
    output/<clip>/track.json           ball position per frame          (vv-track/2)
    output/<clip>/contacts.json        flights + contacts               (vv-contacts/1)
    output/<clip>/cache/dets.json      YOLO candidates, zoom finds, profile  (rebuildable)
    output/<clip>/cache/tracknet.json  TrackNet candidates               (rebuildable)
    output/dev/                        comparison videos, contact sheets, experiments
    output/archive/                    files from older versions of the pipeline

    videos/                            your clip pool: everything you label for TRAINING
    holdout/                           test clips: never trained on, only scored (name them
                                       holdout1, holdout2, ... so they never clash with testN)

    data/gt/<clip>_gt.json             old test labels (every 5th frame; the 10 dev clips)
    data/proposals/<clip>.json         autolabel proposals, waiting for review
    data/verified/<clip>.json          human-checked labels (every frame)

    training/datasets/<name>/          YOLO datasets (Roboflow layout: train|valid|test/images|labels)
    training/tracknet/datasets/<name>/ TrackNet datasets
"""
from __future__ import annotations

from pathlib import Path

OUTPUT = Path("output")
DATA = Path("data")
TRAINING = Path("training")


def stem_of(video: str | Path) -> str:
    return Path(video).stem


def clip_dir(stem: str, root: str | Path = OUTPUT) -> Path:
    return Path(root) / stem


def cache_dir(stem: str, root: str | Path = OUTPUT) -> Path:
    return clip_dir(stem, root) / "cache"


def dets(stem: str, root: str | Path = OUTPUT) -> Path:
    return cache_dir(stem, root) / "dets.json"


def tracknet(stem: str, root: str | Path = OUTPUT, tag: str | None = None) -> Path:
    return cache_dir(stem, root) / (f"tracknet_{tag}.json" if tag else "tracknet.json")


def track(stem: str, root: str | Path = OUTPUT) -> Path:
    return clip_dir(stem, root) / "track.json"


def contacts(stem: str, root: str | Path = OUTPUT) -> Path:
    return clip_dir(stem, root) / "contacts.json"


def video(stem: str, root: str | Path = OUTPUT) -> Path:
    return clip_dir(stem, root) / "tracked.mp4"


def dev(root: str | Path = OUTPUT) -> Path:
    return Path(root) / "dev"


def archive(root: str | Path = OUTPUT) -> Path:
    return Path(root) / "archive"


GT = DATA / "gt"
PROPOSALS = DATA / "proposals"
VERIFIED = DATA / "verified"
YOLO_DATASETS = TRAINING / "datasets"
TRACKNET_DATASETS = TRAINING / "tracknet" / "datasets"


def gt(stem: str) -> Path:
    return GT / f"{stem}_gt.json"


def proposals(stem: str) -> Path:
    return PROPOSALS / f"{stem}.json"


def verified(stem: str) -> Path:
    return VERIFIED / f"{stem}.json"


def gt_stems() -> list[str]:
    return sorted(p.name.replace("_gt.json", "") for p in GT.glob("*_gt.json"))


VIDEOS = Path("videos")
HOLDOUT = Path("holdout")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def holdout_stems() -> list[str]:
    """Test clips: every video in holdout/, plus any checked clip marked "test" in the review tool."""
    out = {p.stem for p in HOLDOUT.glob("*") if p.suffix.lower() in VIDEO_EXT}
    import json
    for p in VERIFIED.glob("*.json"):
        try:
            if json.load(open(p)).get("role") == "test":
                out.add(p.stem)
        except (OSError, ValueError):
            pass
    return sorted(out)


def is_test(stem: str) -> bool:
    """Never train on this clip: it's in holdout/, marked test, or has old data/gt labels."""
    return stem in holdout_stems() or stem in gt_stems()


def ensure(p: Path) -> Path:
    """Create the parent folder of a file path; returns the path."""
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
