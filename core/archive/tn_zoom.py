"""TrackNet points, YOLO pinpoints: confirm TrackNet-only finds with YOLO on a zoomed crop.

TrackNet sees motion (blurred, faint, fast balls) but places the ball coarsely (it works at
512x288, ~0.35 ball-widths median error on your clips). YOLO is precise but misses those
balls in a full, busy frame. So wherever TrackNet sees a ball and YOLO has nothing nearby,
crop the full-resolution frame around TrackNet's point, enlarge it, and run YOLO on just the
crop (the same crop-and-detect as the zoom search, core.refine.zoom_detect). A YOLO box near
TrackNet's point becomes a precise candidate (source "tnzoom"); where YOLO finds nothing,
TrackNet's own point stays as it was.

This runs before linking, so every linker gets the better candidates. Results are cached in
the dets file under "tn_zoom" (keyed by the TrackNet cache and these settings).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

TN_ZOOM_VERSION = 1


def targets(dets: dict, tn_frames: list, ball: float, p_min: float = 0.25, yolo_near_bw: float = 1.0,
            yolo_conf: float = 0.25, zoom: dict | None = None) -> dict[int, list[tuple]]:
    """{frame: [(x, y, peak), ...]}: TrackNet peaks with no YOLO (or zoom-search) box nearby."""
    out = {}
    zoom = zoom or {}
    for f, fr in enumerate(tn_frames[:len(dets["frames"])]):
        boxes = [c for c in dets["frames"][f] if c[4] >= yolo_conf]
        boxes += [c for c in (zoom.get(str(f)) or {}).get("cands", []) if c[4] >= yolo_conf]
        for c in fr:
            if c[4] < p_min:
                continue
            if any(math.hypot(b[0] - c[0], b[1] - c[1]) <= yolo_near_bw * ball for b in boxes):
                continue
            out.setdefault(f, []).append((c[0], c[1], c[4]))
    return out


def run(dets: dict, tn_frames: list, video: str, model, ball: float, half_mult: float = 5.0,
        min_half: int = 64, keep_bw: float = 1.5, cache_key: str = "", log=None, **target_kw) -> dict:
    """{str(frame): {"cands": [[cx, cy, w, h, conf], ...]}} — YOLO boxes found on crops around
    TrackNet-only points, within keep_bw ball-widths of the point. Cached in dets["tn_zoom"]."""
    from core.refine import zoom_detect
    key = f"v{TN_ZOOM_VERSION}:{cache_key}:{half_mult}:{keep_bw}:{sorted(target_kw.items())}"
    cached = dets.get("tn_zoom")
    if cached and cached.get("key") == key:
        return cached["found"]
    tg = targets(dets, tn_frames, ball, zoom=dets.get("zoom"), **target_kw)
    half = int(max(min_half, half_mult * ball))
    found = {}
    if tg:
        cap = cv2.VideoCapture(video)
        f, last = 0, max(tg)
        while f <= last:
            ok, img = cap.read()
            if not ok:
                break
            for x, y, p in tg.get(f, ()):
                near = [c for c in zoom_detect(model, img, (x, y), half)
                        if math.hypot(c[0] - x, c[1] - y) <= keep_bw * ball]
                if near:
                    found.setdefault(str(f), {"cands": []})["cands"].append(max(near, key=lambda c: c[4]))
            f += 1
        cap.release()
    dets["tn_zoom"] = {"key": key, "found": found, "targets": sum(len(v) for v in tg.values())}
    if log:
        log(f"[VolleyVision] TrackNet-guided zoom: {sum(len(v) for v in tg.values())} TrackNet-only points, "
            f"YOLO confirmed {sum(len(v['cands']) for v in found.values())}")
    return found
