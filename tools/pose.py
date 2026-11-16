"""Player poses for one clip: a pretrained pose model (17 body points per person) with tracking.

    python -m tools.pose videos/test76.mov            # -> output/test76/pose.json
    python -m tools.pose videos/test76.mov --frames 80 200

Used by the 3D step to tell who touched the ball (a wrist at the ball), which side of the net
they are on (facing the camera = far side), and where they stand (ankles, before a jump). No
training: the COCO pose model (yolov8m-pose) as it comes. Uses the Mac's GPU when there is one;
~1-1.5 s per frame on a plain CPU.

Far court zoom (when the clip is calibrated, data/cameras/<clip>.json): people behind the far
baseline are a few dozen pixels tall and the whole-frame pass misses them, servers above all
(holdout 2, 3, 4, 5, 7, 8, 10: the server under the tossed ball had no box in any of them).
A second pass looks at the far back court only (the far half, out to 4 m past the baseline),
enlarged FAR_ZOOM times, tracked on its own; a box it finds that no whole-frame box covers is
added (ids from FAR_ID up). The 2x zoom found the server in 4 of those 7 serve frames, the
whole-frame pass in none. About 1.5x the time. --no-far-zoom to skip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths

FAR_ZOOM = 2.0
FAR_ID = 100000                      # ids of people only the far court zoom found
FAR_REGION_M = ((5.0, 13.0), (-6.5, 6.5), (0.0, 2.6))    # x (back court -> 4 m past the far baseline), y, z
TILE_W = 640                         # px of the picture per zoomed tile (x FAR_ZOOM = the model's input size)


def far_region(cam, width: int, height: int, far_side: int = 1, margin: int = 30):
    """The far back court's box in the picture: (x0, y0, x1, y1), or None if it isn't in view."""
    import numpy as np
    (a, b), (c, d), (e, g) = FAR_REGION_M
    pts = np.array([[far_side * x, y, z] for x in (a, b) for y in (c, d) for z in (e, g)])
    uv = cam.project(pts)
    x0, y0 = np.floor(uv.min(0)).astype(int) - margin
    x1, y1 = np.ceil(uv.max(0)).astype(int) + margin
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    return (int(x0), int(y0), int(x1), int(y1)) if x1 - x0 > 40 and y1 - y0 > 40 else None


def tiles(region, w: int = TILE_W, overlap: int = 80):
    """Split the far region into tiles about w px wide, overlapping, each zoomed on its own."""
    x0, y0, x1, y1 = region
    n = max(1, int(np.ceil((x1 - x0 - overlap) / (w - overlap))))
    step = (x1 - x0 - w) / max(n - 1, 1) if n > 1 else 0
    return [(int(x0 + i * step), y0, int(min(x1, x0 + i * step + w)), y1) for i in range(n)]


def _covered(b, boxes, thr: float = 0.4) -> bool:
    """b is someone a whole-frame box already has: overlapping it (IoU over thr), or centred inside
    it at a similar size. A small far player behind a near player's big box is NOT covered."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    hb = b[3] - b[1]
    for o in boxes:
        ix = max(0.0, min(b[2], o[2]) - max(b[0], o[0])) * max(0.0, min(b[3], o[3]) - max(b[1], o[1]))
        u = (b[2] - b[0]) * hb + (o[2] - o[0]) * (o[3] - o[1]) - ix
        if u > 0 and ix / u > thr:
            return True
        if o[0] <= cx <= o[2] and o[1] <= cy <= o[3] and 0.7 <= hb / max(o[3] - o[1], 1) <= 1.4:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--model", default="models/yolov8m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--frames", nargs=2, type=int, default=None, help="first and last frame (default: all)")
    ap.add_argument("--redo", action="store_true", help="run again even if pose.json exists")
    ap.add_argument("--no-far-zoom", action="store_true", help="skip the far back court zoom pass")
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    stem = Path(a.video).stem
    out = paths.clip_dir(stem, a.output) / "pose.json"
    if out.exists() and not a.redo:
        print(f"[pose] {stem}: cached  →  {out}  (--redo to run again; player labels are tied to this run)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    m = YOLO(a.model)
    import torch
    device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)          # phone clips: stored sideways + a rotation flag
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    region, m_far = None, None
    cam_file = Path("data/cameras") / f"{stem}.json"
    if not a.no_far_zoom and cam_file.exists():
        from core.geometry.camera import Camera
        cam = Camera.load(cam_file)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        far_sign = -int(np.sign(cam.position[0])) or 1
        region = far_region(cam, W, H, far_side=far_sign)
        if region:
            T = tiles(region)
            m_far = [YOLO(a.model) for _ in T]           # each tile its own tracker
            print(f"[pose] {stem}: far court zoom on {region} in {len(T)} tiles (x{FAR_ZOOM:g})")
    elif not a.no_far_zoom:
        print(f"[pose] {stem}: not calibrated: no far court zoom (calibrate first to find far players)")
    f0, f1 = a.frames or (0, n - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames = {}
    for f in range(f0, f1 + 1):
        ok, img = cap.read()
        if not ok:
            break
        r = m.track(img, persist=True, tracker="bytetrack.yaml", imgsz=a.imgsz, verbose=False, conf=0.25,
                    device=device)[0]
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        frames[f] = [{"id": i, "box": [round(v, 1) for v in b.tolist()],
                      "kp": [[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in k.tolist()]}
                     for i, b, k in zip(ids, r.boxes.xyxy, r.keypoints.data)]
        if m_far is not None:
            whole = [p["box"] for p in frames[f]]
            for ti, ((x0, y0, x1, y1), mt) in enumerate(zip(T, m_far)):
                crop = cv2.resize(img[y0:y1, x0:x1], None, fx=FAR_ZOOM, fy=FAR_ZOOM, interpolation=cv2.INTER_CUBIC)
                rf = mt.track(crop, persist=True, tracker="bytetrack.yaml", imgsz=a.imgsz, verbose=False,
                              conf=0.25, device=device)[0]
                fids = rf.boxes.id.int().tolist() if rf.boxes.id is not None else [-1] * len(rf.boxes)
                for i, b, k in zip(fids, rf.boxes.xyxy, rf.keypoints.data):
                    bb = [float(b[0]) / FAR_ZOOM + x0, float(b[1]) / FAR_ZOOM + y0,
                          float(b[2]) / FAR_ZOOM + x0, float(b[3]) / FAR_ZOOM + y0]
                    if bb[3] >= y1 - 3 or b[1] <= 2:       # cut off by the tile: a near player's part
                        continue
                    g = cam.floor_point([((bb[0] + bb[2]) / 2, bb[3])], 0.0)[0]
                    if far_sign * g[0] < FAR_REGION_M[0][0] - 1.0:   # standing on the near side: not ours
                        continue
                    if _covered(bb, whole):                # the whole-frame pass (or another tile) has them
                        continue
                    whole.append(bb)
                    frames[f].append({"id": FAR_ID * (ti + 1) + i if i >= 0 else -1,
                                      "box": [round(v, 1) for v in bb], "far": True,
                                      "kp": [[round(x / FAR_ZOOM + x0, 1), round(y / FAR_ZOOM + y0, 1), round(c, 2)]
                                             for x, y, c in k.tolist()]})
        if (f - f0) % 100 == 0:
            print(f"[pose] {stem}: frame {f}/{f1}", flush=True)
    json.dump({"schema": "vv-pose/1", "video": a.video, "model": a.model, "keypoints": "coco17",
               "far_zoom": {"region": region, "zoom": FAR_ZOOM} if region else None,
               "frames": frames}, open(out, "w"), separators=(",", ":"))
    print(f"[pose] {stem}: {len(frames)} frames  →  {out}")


if __name__ == "__main__":
    main()
