"""Multi-frame ball finders behind one interface: WASB (default) and TrackNet (classic pipeline).

The pipeline has one "finder" slot (historically called TrackNet: the TrackNet stage, the
"tracknet" source in the linkers). Which model fills it is decided by the checkpoint:

    *.pth.tar or a path containing "wasb"  -> core.wasb       (WASB, HRNet)
    anything else                          -> core.tracknet   (TrackNetV3)
(VballNet was tried and dropped: core/archive/vballnet.py.)

Each finder caches its candidates per clip (output/<clip>/cache/tracknet[_tag].json) and gets
its own score calibration (models/score_calibration.json, one model per finder name), because
a peak of 0.5 means different things for different models.
"""
from __future__ import annotations

from pathlib import Path

from core import paths

TRACKNET_DEFAULT = "models/tracknet/FT_TrackNet_best.pt"


def kind(ckpt: str | None) -> str | None:
    if not ckpt:
        return None
    p = str(ckpt).lower()
    if "wasb" in p or p.endswith(".pth.tar"):
        return "wasb"
    return "tracknet"


def tag(ckpt: str) -> str | None:
    """Cache tag for this checkpoint: None (tracknet.json) for TrackNet, as before; WASB's
    released weights -> "wasb"; any other checkpoint -> "<kind>_<file stem>"."""
    k = kind(ckpt)
    if k == "tracknet":
        return None
    from core.wasb import DEFAULT_CKPT as WASB_DEFAULT
    if k == "wasb" and Path(ckpt).name == Path(WASB_DEFAULT).name:
        return "wasb"
    stem = Path(ckpt).name.split('.')[0]
    return stem if stem.startswith(k) else f"{k}_{stem}"


def calib_name(ckpt: str) -> str:
    """Which calibration model scores this finder's peaks ("tracknet", "wasb", "wasb_<stem>", ...)."""
    t = tag(ckpt)
    return "tracknet" if t is None else t


def cache_path(stem: str, ckpt: str, root=paths.OUTPUT) -> Path:
    return paths.tracknet(stem, root, tag(ckpt))


def candidates(video: str, dets: dict, ckpt: str, cache_path=None, log=print, redo: bool = False):
    """Per-frame candidates [[x, y, size, size, peak], ...] from whichever finder ckpt is."""
    k = kind(ckpt)
    cp = cache_path or paths.tracknet(Path(video).stem, tag=tag(ckpt))
    if k == "wasb":
        from core.wasb import candidates as c
    else:
        from core.tracknet import candidates as c
    return c(video, dets, ckpt, cache_path=cp, log=log, redo=redo)


def load_cached(stem: str, ckpt: str, root=paths.OUTPUT):
    """Cached candidates for a clip, or None."""
    import json
    p = paths.tracknet(stem, root, tag(ckpt))
    return json.load(open(p))["frames"] if p.exists() else None


def place_on_arc(ckpt: str | None) -> bool:
    """Should the flights stage move this finder's points onto the fitted arc? Yes for TrackNet
    (coarse, 0.35 bw: the arc places it better, 0.33 -> 0.16 bw on the dev clips); no for WASB
    (0.17 bw: on the holdout clips the arc made it worse, 12.3% -> 16.6% of points > 0.25 bw off)."""
    return kind(ckpt) in (None, "tracknet")


def correct(frames, ball_px: float, ckpt: str):
    """Model-specific position fix: TrackNet's measured per-checkpoint offset; none for the others."""
    if kind(ckpt) != "tracknet":
        return frames
    from core.tracknet import correct as c
    return c(frames, ball_px, ckpt=ckpt)
