"""Classify contacts (from core.flights) as touch / bounce / other, using where the people are.

    touch   the ball is at a person's upper body or arms at the contact moment
            (inside a person box horizontally, in the upper ~45% of it or just above it)
    bounce  not a touch; falling in and rising out more slowly (a floor bounce loses energy),
            and in the lowest part of this clip's ball positions
    other   anything else: net, wall, antenna, or unclear

People come from a standard COCO person detector (YOLOv8n), run only on contact frames.
Its boxes are cached in the detections file, so re-runs don't redo it.
"""
from __future__ import annotations

import cv2
import numpy as np

TOUCH_MAX_REL_Y = 0.45      # contact height within the person box: 0 = top of head, 1 = feet
TOUCH_MIN_REL_Y = -0.45     # ... allow a bit above the head (sets, blocks, attacks)
TOUCH_MAX_DX = 0.10         # horizontal distance outside the box, in box widths
BOUNCE_MAX_VY_RATIO = 0.90  # rising-out vertical speed / falling-in vertical speed
BOUNCE_MIN_Y_REL = 0.90     # contact in the lowest 10% of the clip's ball y range


def person_boxes(video: str, frames: list[int], model, cache: dict | None = None, conf: float = 0.25):
    """{frame: [[x1, y1, x2, y2, conf], ...]} for the requested frames (sequential decode)."""
    cache = cache if cache is not None else {}
    need = sorted({f for f in frames if str(f) not in cache})
    if need:
        cap = cv2.VideoCapture(video)
        f, want = 0, set(need)
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if f in want:
                r = model.predict(img, classes=[0], conf=conf, imgsz=1280, verbose=False)[0]
                cache[str(f)] = [[round(float(v), 1) for v in b.xyxy[0]] + [round(float(b.conf[0]), 3)]
                                 for b in r.boxes]
                want.discard(f)
            f += 1
        cap.release()
    return {f: cache.get(str(f), []) for f in frames}


def _touch_score(c, boxes):
    """Best (rel_y, dx) of the contact against any person box, or None."""
    best = None
    for x1, y1, x2, y2, _ in boxes:
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        dx = 0.0 if x1 <= c.x <= x2 else min(abs(c.x - x1), abs(c.x - x2)) / w
        rel = (c.y - y1) / h
        if best is None or abs(rel - 0.2) + dx < abs(best[0] - 0.2) + best[1]:
            best = (rel, dx)
    return best


def classify(contacts, track, video: str, person_model, cache: dict | None = None):
    """Sets c.kind and c.extra['person'] on each contact; returns the contacts."""
    if not contacts:
        return contacts
    ys = [p[1] for p, s in zip(track.pos, track.state) if p is not None and s in ("detected", "weak", "tracknet")]
    ylo, yhi = (np.percentile(ys, 2), np.percentile(ys, 98)) if ys else (0, track.height)
    frames = [int(round(c.frame)) for c in contacts]
    boxes = person_boxes(video, frames, person_model, cache)
    for c, f in zip(contacts, frames):
        ts = _touch_score(c, boxes.get(f, []))
        c.extra["person"] = None if ts is None else {"rel_y": round(ts[0], 2), "dx": round(ts[1], 2)}
        y_rel = (c.y - ylo) / max(yhi - ylo, 1e-6)
        vy_ratio = abs(c.vy_out) / max(abs(c.vy_in), 1e-6)
        c.extra["y_rel"] = round(float(y_rel), 2)
        c.extra["vy_ratio"] = round(float(vy_ratio), 2)
        if ts is not None and ts[1] <= TOUCH_MAX_DX and TOUCH_MIN_REL_Y <= ts[0] <= TOUCH_MAX_REL_Y:
            c.kind = "touch"
        elif c.vy_in > 0 and c.vy_out < 0 and vy_ratio <= BOUNCE_MAX_VY_RATIO and y_rel >= BOUNCE_MIN_Y_REL:
            c.kind = "bounce"
        else:
            c.kind = "other"
    return contacts
