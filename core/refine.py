"""Zoom search: look again, harder, wherever the linker had to guess.

A FILLED frame means the linker interpolated the ball's position because YOLO
found nothing there at full-frame resolution. Often the ball IS visible — just
small or motion-blurred. Cropping a window around the guessed position and
running YOLO on that crop at higher zoom recovers many of these.

Found candidates are NOT accepted directly: they're added to the frame's
candidate list and the linker re-runs, so they must still fit the arc like any
other weak detection. Rounds repeat until no new frames get resolved.

Results are cached in the dets JSON under "zoom" so re-runs don't redo the search.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.linker import link, LinkParams, FILLED


class FrameReader:
    """Sequential-read cache (seeking is unreliable on some phone videos)."""

    def __init__(self, path: str):
        self.path = path
        self.cache: dict[int, np.ndarray] = {}

    def load(self, wanted: set[int]):
        need = {f for f in wanted if f not in self.cache}
        if not need:
            return
        cap = cv2.VideoCapture(self.path)
        f, last = 0, max(need)
        while f <= last:
            ok, img = cap.read()
            if not ok:
                break
            if f in need:
                self.cache[f] = img
            f += 1
        cap.release()

    def get(self, f: int) -> np.ndarray | None:
        return self.cache.get(f)


def frame_changed_pixels(prev_gray, gray, thresh: int = 12) -> int:
    return int((np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)) > thresh).sum())


def small_gray(img):
    return cv2.cvtColor(cv2.resize(img, (640, 360)), cv2.COLOR_BGR2GRAY)


DUP_MAX_CHANGED = 8   # a repeated frame changes < ~5 px at 640x360; a real one changes hundreds


def video_dup_flags(path: str) -> list[bool]:
    cap = cv2.VideoCapture(path)
    flags, prev = [], None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        g = small_gray(img)
        flags.append(prev is not None and frame_changed_pixels(prev, g) < DUP_MAX_CHANGED)
        prev = g
    cap.release()
    return flags


def zoom_detect(model, img: np.ndarray, center, half: int, imgsz: int = 640, conf: float = 0.02):
    """Run YOLO on a (2*half)^2 crop around center. Returns [[cx, cy, w, h, conf], ...] in frame coords."""
    H, W = img.shape[:2]
    half = min(half, W // 2, H // 2)
    x0 = int(np.clip(center[0] - half, 0, W - 2 * half))
    y0 = int(np.clip(center[1] - half, 0, H - 2 * half))
    crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
    r = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)[0]
    out = []
    for b in r.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        out.append([round((x1 + x2) / 2 + x0, 1), round((y1 + y2) / 2 + y0, 1),
                    round(x2 - x1, 1), round(y2 - y1, 1), round(float(b.conf[0]), 3)])
    return out


def link_with_zoom(dets: dict, video: str, model, params: LinkParams | None = None,
                   rounds: int = 3, half_mult: float = 5.0, min_half: int = 128, log=print):
    """dets: the vv-dets/1 dict (modified in place: dets['zoom'] caches searches)."""
    P = params or LinkParams()
    fps, W, H = dets["fps"], dets["width"], dets["height"]
    zoom: dict = dets.setdefault("zoom", {})          # str(frame) -> {"c": [cx, cy], "cands": [...]}
    reader = FrameReader(video)
    if "dup" not in dets:                              # older dumps: compute from the video once
        dets["dup"] = video_dup_flags(video)
    dupf = dets["dup"]

    def merged():
        fr = [list(f) for f in dets["frames"]]
        for k, v in zoom.items():
            fr[int(k)] = fr[int(k)] + v["cands"]
        return fr

    track = link(merged(), fps, W, H, P, dup_flags=dupf)
    ball = float(np.median([max(c[2], c[3]) for f in dets["frames"] for c in f if c[4] >= P.strong_conf] or [30]))
    half = int(max(min_half, half_mult * ball))
    for rnd in range(rounds):
        todo = []
        for f, (s, p) in enumerate(zip(track.state, track.pos)):
            if s != FILLED or (track.dup and track.dup[f]):
                continue
            prev = zoom.get(str(f))
            # search if never searched, or the guess moved far from where we looked last time
            if prev is None or np.hypot(p[0] - prev["c"][0], p[1] - prev["c"][1]) > half / 2:
                todo.append((f, p))
        if not todo:
            break
        reader.load({f for f, _ in todo})
        found = 0
        for f, p in todo:
            img = reader.get(f)
            if img is None:
                continue
            cands = zoom_detect(model, img, p, half)
            zoom[str(f)] = {"c": [round(p[0], 1), round(p[1], 1)], "cands": cands}
            found += bool(cands)
        before = track.summary()["filled"]
        track = link(merged(), fps, W, H, P, dup_flags=dupf)
        log(f"  zoom round {rnd + 1}: searched {len(todo)} filled frames, candidates in {found}, "
            f"filled {before} -> {track.summary()['filled']}")
    return track
