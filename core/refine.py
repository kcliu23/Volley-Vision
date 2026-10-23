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

from core.linker import link, LinkParams, FILLED, LINKER_VERSION


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


def content_box(path: str, samples: int = 40) -> list[int]:
    """Bounding box of the actual picture inside the video frame.

    Screen recordings and re-exported clips often have side/top bars (black, gray
    or a solid colour). Bars never change over time, so columns/rows whose pixels
    are constant across sampled frames at the frame's edges are excluded.
    """
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, n // samples)
    frames, f = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f % step == 0:
            frames.append(cv2.cvtColor(cv2.resize(img, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY).astype(np.float32))
        f += 1
    cap.release()
    if len(frames) < 3:
        return [0, 0, 0, 0]
    sd = np.stack(frames).std(axis=0)                  # temporal std per pixel
    h, w = sd.shape
    live_c = sd.mean(axis=0) > 0.5                     # column changes over time somewhere
    live_r = sd.mean(axis=1) > 0.5
    cols, rows = np.nonzero(live_c)[0], np.nonzero(live_r)[0]
    if len(cols) == 0 or len(rows) == 0:
        return [0, 0, w * 4, h * 4]
    return [int(cols[0]) * 4, int(rows[0]) * 4, min(w * 4, (int(cols[-1]) + 1) * 4), min(h * 4, (int(rows[-1]) + 1) * 4)]


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


def ahead_targets(track, horizon: int, fit_pts: int = 6):
    """Predicted positions just past the end (and before the start) of every tracked run.

    Extrapolates each run's local arc into the frames where the ball was lost, so
    zoom search can look for it along the route it was on. Stops at the frame edge.
    """
    from core.linker import _predict, NONE
    n = len(track.state)
    have = [s != NONE and p is not None for s, p in zip(track.state, track.pos)]
    out = {}
    for i in range(n):
        if not have[i]:
            continue
        for direction in (+1, -1):
            j = i + direction
            if not (0 <= j < n) or have[j]:
                continue                       # not the end of a run in this direction
            # support: up to fit_pts real points on this side of the boundary
            sup = []
            k = i
            while 0 <= k < n and have[k] and len(sup) < fit_pts:
                if track.state[k] != "filled" and not (track.dup and track.dup[k]):
                    sup.append(k)
                k -= direction
            if len(sup) < 3:
                continue
            xs = [track.pos[k][0] for k in sup]
            ys = [track.pos[k][1] for k in sup]
            for step in range(1, horizon + 1):
                f = i + direction * step
                if not (0 <= f < n) or have[f]:
                    break
                px, py, _ = _predict(sup, xs, ys, f)
                x0, y0, x1, y1 = track.content or (0, 0, track.width, track.height)
                if not (x0 <= px < x1 and y0 <= py < y1):
                    break
                out.setdefault(f, (px, py))
    # tentative (unconfirmed) weak chains: look inside their gaps and along both ends
    for ch in getattr(track, "tentative", []):
        fs = [c[0] for c in ch]
        xs = [c[1] for c in ch]
        ys = [c[2] for c in ch]
        lo, hi = fs[0] - horizon, fs[-1] + horizon
        for f in range(max(0, lo), min(n, hi + 1)):
            if have[f] or f in fs:
                continue
            px, py, _ = _predict(fs, xs, ys, f)
            x0, y0, x1, y1 = track.content or (0, 0, track.width, track.height)
            if x0 <= px < x1 and y0 <= py < y1:
                out.setdefault(f, (px, py))
    return out


def link_with_zoom(dets: dict, video: str, model, params: LinkParams | None = None,
                   rounds: int = 6, half_mult: float = 5.0, min_half: int = 128,
                   ahead_s: float = 0.25, log=print, linker: str = "rule"):
    """linker: "rule" (core.linker) or "cost" (core.cost_linker). Each keeps its own zoom
    cache, because zoom search only looks where that linker's track points it."""
    """dets: the vv-dets/1 dict (modified in place: dets['zoom'] caches searches)."""
    P = params or LinkParams()
    fps, W, H = dets["fps"], dets["width"], dets["height"]
    zkey = "zoom" if linker == "rule" else f"zoom_{linker}"
    if linker == "cost":
        from core.cost_linker import link as _cost_link, COST_LINKER_VERSION as ver

        def link(frames, fps, W, H, P, dup_flags=None, content=None):
            return _cost_link(frames, fps, W, H, dup_flags=dup_flags, content=content, link_params=P)
    else:
        from core.linker import link
        ver = LINKER_VERSION
    if dets.get(f"{zkey}_ver") != ver:                 # searches steered by an older linker: redo
        dets[zkey] = {}
        dets[f"{zkey}_ver"] = ver
    zoom: dict = dets.setdefault(zkey, {})            # str(frame) -> {"c": [cx, cy], "cands": [...]}
    reader = FrameReader(video)
    if len(dets.get("dup") or []) != len(dets["frames"]):   # missing/broken: compute from the video once
        dets["dup"] = video_dup_flags(video)
    dupf = dets["dup"]
    if not dets.get("content"):
        dets["content"] = content_box(video)
    content = dets["content"]

    def merged():
        fr = [list(f) for f in dets["frames"]]
        for k, v in zoom.items():
            fr[int(k)] = fr[int(k)] + v["cands"]
        return fr

    track = link(merged(), fps, W, H, P, dup_flags=dupf, content=content)
    ball = float(np.median([max(c[2], c[3]) for f in dets["frames"] for c in f if c[4] >= P.strong_conf] or [30]))
    half = int(max(min_half, half_mult * ball))
    for rnd in range(rounds):
        targets = {f: p for f, (s, p) in enumerate(zip(track.state, track.pos)) if s == FILLED}
        n_fill = len(targets)
        for f, p in ahead_targets(track, max(2, int(round(ahead_s * fps)))).items():
            targets.setdefault(f, p)
        todo = []
        for f, p in sorted(targets.items()):
            if track.dup and track.dup[f]:
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
        before = track.summary()
        track = link(merged(), fps, W, H, P, dup_flags=dupf, content=content)
        after = track.summary()
        log(f"  zoom round {rnd + 1}: searched {len(todo)} frames ({n_fill} filled + route ahead), "
            f"candidates in {found}; coverage {before['coverage']:.1%} -> {after['coverage']:.1%}, "
            f"filled {before['filled']} -> {after['filled']}")
        if after["coverage"] <= before["coverage"] and after["filled"] >= before["filled"]:
            break
    return track
