"""Finder-first tracking: the default pipeline since 2026-10-27.

    1. sparse YOLO   full-frame YOLO on every 10th frame only: ball size and clip profile
    2. finder        WASB on every frame: where the ball might be (candidates + peak)
    3. crops         YOLO on a small crop around the finder's top candidate, if its calibrated
                     P(ball) >= 0.6 (strict, the default); options: up to N candidates with
                     P >= q (--crop-top / --crop-p), or top 2 with no threshold (loose)
    4. joint linker  picks the ball's path over the whole clip in one pass: every candidate
                     weighed by its calibrated probability + physics (speed, size, gravity arcs)

No zoom-search rounds, no rule-linker cascade. (A tracklet layer with rival / fixture removal
used to sit before the joint linker; removed 2026-10-28 after tools.ablate showed the joint
linker alone scores the same on holdout and dev, and better on the checked clips:
core/archive/tracklet_linker.py.) Holdout numbers: README.md.

Caches (output/<clip>/cache/): sample_dets.json (sparse YOLO + profile), tracknet_<finder>.json
(finder candidates), crops_<finder>[_strict0.6 | _top2_p0.1].json (crop finds; no suffix = loose).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from core import paths

DEFAULT_FINDER = "models/wasb/wasb_ft2.pth.tar"
CROP_TOP, CROP_P = 1, 0.6      # default crop rule (strict): the top candidate, if calibrated P(ball) >= 0.6.
                               # (2, 0.1) scores +0.7 on-ball on holdout but breaks regression case test21
SAMPLE_EVERY = 10              # sparse full-frame YOLO pass: every Nth frame
P_MIN, TOP = 0.1, 2            # loose crop rule: candidates with peak >= P_MIN, at most TOP per frame


def sparse_dets(video: str, model, out_path: Path, every: int = SAMPLE_EVERY, conf: float = 0.05,
                imgsz: int = 1280, log=print) -> dict:
    """YOLO on every `every`-th full frame (ball size, profile); repeated-frame flags for all
    frames. Same format as the full detection cache, with empty lists for unsampled frames,
    plus "every". Cached."""
    out_path = Path(out_path)
    if out_path.exists():
        d = json.load(open(out_path))
        if d.get("every") == every and d.get("imgsz") == imgsz:
            return d
    from core.refine import small_gray, frame_changed_pixels, DUP_MAX_CHANGED
    from core.device import predict, BATCH
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, dup, prev, batch, t0 = [], [], None, [], time.time()

    def flush():
        for (f, _), r in zip(batch, predict(model, [im for _, im in batch], conf=conf, imgsz=imgsz, batch=len(batch))):
            frames[f] = [[round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1), round(x2 - x1, 1), round(y2 - y1, 1),
                          round(float(b.conf[0]), 3)] for b in r.boxes for x1, y1, x2, y2 in [map(float, b.xyxy[0])]]
        batch.clear()

    while True:
        ok, img = cap.read()
        if not ok:
            break
        f = len(frames)
        frames.append([])
        g = small_gray(img)
        dup.append(prev is not None and frame_changed_pixels(prev, g) < DUP_MAX_CHANGED)
        prev = g
        if f % every == 0:
            batch.append((f, img))
            if len(batch) == BATCH:
                flush()
    if batch:
        flush()
    cap.release()
    d = {"schema": "vv-dets/1", "video": video, "fps": fps, "width": w, "height": h, "imgsz": imgsz,
         "conf": conf, "every": every, "frames": frames, "dup": dup}
    json.dump(d, open(paths.ensure(out_path), "w"), separators=(",", ":"))
    if log:
        log(f"[VolleyVision] sparse YOLO pass: {len(frames[::every])} of {len(frames)} frames "
            f"({time.time() - t0:.0f}s)")
    return d


def ball_px(dets: dict) -> float:
    prof = dets.get("profile") or {}
    if prof.get("ball_px"):
        return float(prof["ball_px"])
    s = [max(c[2], c[3]) for f in dets["frames"] for c in f if c[4] >= 0.4]
    return float(np.median(s)) if s else 30.0


def crop_targets(row, prob, top: int, q: float | None):
    """Which of a frame's finder candidates get a YOLO crop: the `top` strongest with peak >=
    P_MIN, and of those only the ones with calibrated P(ball) >= q (q=None: all of them)."""
    t = sorted([c for c in row if c[4] >= P_MIN], key=lambda c: -c[4])[:top]
    return t if q is None else [c for c in t if prob(c[4]) >= q]


def subset_finds(found: dict, cands, prob, ball: float, top: int, q: float | None):
    """Simulate a stricter crop rule from the loose crop cache (top 2, peak >= P_MIN) without
    re-running YOLO: keep the boxes found around the candidates the rule would crop.
    Returns (found, number of crops)."""
    out, ncrops = {}, 0
    for f, row in enumerate(cands):
        tg = crop_targets(row, prob, top, q)
        ncrops += len(tg)
        v = found.get(str(f))
        if not tg or not v:
            continue
        boxes = [b for b in v["cands"] if any(abs(b[0] - c[0]) <= 1.5 * ball and abs(b[1] - c[1]) <= 1.5 * ball for c in tg)]
        if boxes:
            out[str(f)] = {"c": tg[0][:2], "cands": boxes}
    return out, ncrops


def crop_suffix(top: int, q: float | None) -> str:
    """Cache-name suffix of a crop rule (kept compatible with the older strict/loose names)."""
    if top == TOP and q is None:
        return ""                                  # loose
    if top == 1:
        return f"_strict{q:g}"
    return f"_top{top}_p{q:g}"


def crop_finds(stem: str, video: str, cands, ball: float, ftag: str, model, prob, top: int = CROP_TOP,
               q: float | None = CROP_P, root=paths.OUTPUT) -> dict:
    """YOLO on crops around the finder's candidates chosen by crop_targets (cached). If the loose
    crop cache exists, a stricter rule is taken from it instead of running YOLO again."""
    from core.refine import zoom_detect_many
    cp = paths.cache_dir(stem, root) / f"crops_{ftag}{crop_suffix(top, q)}.json"
    if cp.exists():
        return json.load(open(cp))
    loose = paths.cache_dir(stem, root) / f"crops_{ftag}.json"
    if loose.exists() and top <= TOP:
        found, n = subset_finds(json.load(open(loose))["found"], cands, prob, ball, top, q)
        out = {"found": found, "crops": n, "frames": len(cands), "secs": 0.0, "from": loose.name}
        json.dump(out, open(paths.ensure(cp), "w"))
        return out
    half = int(max(64, 5.0 * ball))
    want = {f: crop_targets(row, prob, top, q) for f, row in enumerate(cands)}
    want = {f: v for f, v in want.items() if v}
    cap = cv2.VideoCapture(video)
    found, f, ncrops, t0 = {}, 0, 0, time.time()
    last = max(want) if want else -1
    while f <= last:
        ok, img = cap.read()
        if not ok:
            break
        if f in want:
            jobs = [(img, (c[0], c[1]), half) for c in want[f]]
            ncrops += len(jobs)
            res = zoom_detect_many(model, jobs)
            boxes = [b for c, r in zip(want[f], res) for b in r
                     if abs(b[0] - c[0]) <= 1.5 * ball and abs(b[1] - c[1]) <= 1.5 * ball]
            if boxes:
                found[str(f)] = {"c": want[f][0][:2], "cands": boxes}
        f += 1
    cap.release()
    out = {"found": found, "crops": ncrops, "frames": len(cands), "secs": round(time.time() - t0, 1)}
    json.dump(out, open(paths.ensure(cp), "w"))
    return out


def track(video: str, model, finder: str = DEFAULT_FINDER, cal=None, crop_top: int = None,
          crop_p: float | None = "default", root=paths.OUTPUT, log=print):
    """Run stages 1-4 on a clip. Returns (track, info). Crop rule: up to crop_top candidates per
    frame with calibrated P(ball) >= crop_p (crop_p=None: no threshold, the loose rule)."""
    crop_top = CROP_TOP if crop_top is None else crop_top
    crop_p = CROP_P if crop_p == "default" else crop_p
    from core import finders
    from core.calibration import Calibration
    from core.profile import get as get_profile, link_args
    from core.joint_linker import link as joint_link
    from core.joint_linker import JointParams

    stem = Path(video).stem
    d = sparse_dets(video, model, paths.cache_dir(stem, root) / "sample_dets.json", log=log)
    before = d.get("profile")
    get_profile(video, d)                          # frame times, repeated frames, content box, ball size
    if d.get("profile") is not before:
        json.dump(d, open(paths.cache_dir(stem, root) / "sample_dets.json", "w"), separators=(",", ":"))
    ball = ball_px(d)
    cands = finders.candidates(video, d, finder, cache_path=finders.cache_path(stem, finder, root), log=log)
    if cands is None:
        raise FileNotFoundError(f"ball finder weights not found: {finder}")
    cands = finders.correct(cands, ball, finder)
    cal = (cal or Calibration.load()).for_finder(finders.calib_name(finder))
    ftag = finders.tag(finder) or "tracknet"
    cf = crop_finds(stem, video, cands, ball, ftag, model, lambda pk: float(cal.prob("tracknet", [pk])[0]),
                    crop_top, crop_p, root=root)
    a = link_args(d)
    d2 = dict(d)
    d2["frames"] = [[] for _ in d["frames"]]      # the sparse boxes only set ball size / profile
    d2["zoom"] = cf["found"]
    t = joint_link(d2, cands, cal, a["fps"], d["width"], d["height"], ball, JointParams(tn_only_bias=1.0),
                   dup_flags=a["dup_flags"], content=a["content"], times=a["times"])
    info = {"ball_px": ball, "crops": cf["crops"], "frames": len(cands), "crop_frames": len(cf["found"]),
            "fps": a["fps"], "width": d["width"], "height": d["height"]}
    if log:
        log(f"[VolleyVision] finder-first: {cf['crops']} YOLO crops ({cf['crops'] / max(len(cands), 1):.2f} per frame), "
            f"ball confirmed by YOLO in {len(cf['found'])} frames")
    return t, info
