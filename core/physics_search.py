"""Physics-guided search: look for the ball where the flights say it must be.

Step 1 of the physics backbone (docs/DEVELOPMENT_LOG.md). After a first pass (candidates ->
joint linker -> flights), every flight predicts the ball's position in every frame it spans. In
frames where no YOLO box confirms the ball, YOLO checks a crop AT THE PREDICTED POSITION; the
finds are added to the candidates and the linker runs again. Plan B for WASB's blind spots
(big near-camera balls, blurred frames, a second ball WASB ranked higher).

Where it searches (each a "trigger"):
  gaps       frames inside a flight (between its first and last observed frame) without a YOLO box
  contacts   a few frames past each flight's end and before each flight's start, extrapolated
             from that flight: the fit is weakest there (fastest motion, direction change)
Frames whose existing crop finds already contain a box near the prediction are skipped.

Crop finds are cached in output/<clip>/cache/crops_guided_<finder>.json, keyed by the target list.
"""
from __future__ import annotations

import hashlib
import json
import time

import cv2

from core import paths

CONTACT_S = 0.1          # search this long past a flight's end / before its start
ACCEPT_BW = 1.5          # a box counts if within this many ball-widths of the prediction


def targets(track, flights, ball: float, zoom: dict, fps: float, contact_s: float = CONTACT_S) -> dict:
    """{frame: [(x, y, why), ...]}: where to crop. Skips frames already confirmed by YOLO near
    the prediction, positions outside the picture, and repeated frames."""
    n = len(track.pos)
    x0, y0, x1, y1 = track.content or (0, 0, track.width, track.height)
    dup = track.dup or [False] * n
    k = max(1, int(round(contact_s * fps)))
    out: dict = {}

    def add(f, xy, why):
        if not 0 <= f < n or dup[f]:
            return
        x, y = xy
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return
        if track.state[f] in ("detected", "weak") and track.pos[f] is not None and \
                abs(track.pos[f][0] - x) <= ball and abs(track.pos[f][1] - y) <= ball:
            return                                   # already a YOLO-confirmed point there
        z = (zoom or {}).get(str(f))
        if z and any(abs(b[0] - x) <= ACCEPT_BW * ball and abs(b[1] - y) <= ACCEPT_BW * ball for b in z["cands"]):
            return                                   # an existing crop already covered it
        lst = out.setdefault(f, [])
        if not any(abs(px - x) <= 0.5 * ball and abs(py - y) <= 0.5 * ball for px, py, _ in lst):
            lst.append((round(float(x), 1), round(float(y), 1), why))

    for fl in flights:
        for f in range(fl.f0, fl.f1 + 1):
            add(f, fl.at(f), "gap")
        for f in range(fl.f1 + 1, fl.f1 + 1 + k):
            add(f, fl.at(f), "contact")
        for f in range(fl.f0 - k, fl.f0):
            add(f, fl.at(f), "contact")
    return out


def search(stem: str, video: str, tg: dict, ball: float, model, ftag: str, root=paths.OUTPUT) -> dict:
    """YOLO on a crop around every target (batched on a GPU, as core.finder_first.crop_finds).
    Returns {"found": {frame: {"c": [x, y], "cands": [...]}}, "crops": n, ...} (cached)."""
    from core.refine import zoom_detect_many
    from core.device import BATCH, yolo_device
    key = hashlib.md5(json.dumps(sorted((f, v) for f, v in tg.items())).encode()).hexdigest()[:12]
    cp = paths.cache_dir(stem, root) / f"crops_guided_{ftag}.json"
    if cp.exists():
        c = json.load(open(cp))
        if c.get("key") == key:
            return c
    batch = BATCH if yolo_device() != "cpu" else 1
    half = int(max(64, 5.0 * ball))
    per_frame, queue, t0 = {}, [], time.time()

    def flush():
        res = zoom_detect_many(model, [(img, (x, y), half) for _, x, y, img in queue])
        for (g, x, y, _), r in zip(queue, res):
            per_frame.setdefault(g, []).extend(b for b in r if abs(b[0] - x) <= ACCEPT_BW * ball
                                               and abs(b[1] - y) <= ACCEPT_BW * ball)
        queue.clear()

    cap = cv2.VideoCapture(video)
    last, f = (max(tg) if tg else -1), 0
    while f <= last:
        ok, img = cap.read()
        if not ok:
            break
        for x, y, _ in tg.get(f, ()):
            queue.append((f, x, y, img))
        if len(queue) >= batch:
            flush()
        f += 1
    if queue:
        flush()
    cap.release()
    found = {str(g): {"c": list(tg[g][0][:2]), "cands": b} for g, b in sorted(per_frame.items()) if b}
    why = {}
    for v in tg.values():
        for *_, w in v:
            why[w] = why.get(w, 0) + 1
    out = {"key": key, "found": found, "crops": sum(len(v) for v in tg.values()), "by_trigger": why,
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(paths.ensure(cp), "w"))
    return out


def merge_zoom(zoom: dict, extra: dict) -> dict:
    """Crop finds from both sources, per frame."""
    out = {k: {"c": v["c"], "cands": list(v["cands"])} for k, v in (zoom or {}).items()}
    for k, v in extra.items():
        if k in out:
            out[k]["cands"] += [b for b in v["cands"] if not any(abs(b[0] - c[0]) < 1 and abs(b[1] - c[1]) < 1
                                                                 for c in out[k]["cands"])]
        else:
            out[k] = {"c": v["c"], "cands": list(v["cands"])}
    return out
