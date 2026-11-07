"""Player poses for one clip: a pretrained pose model (17 body points per person) with tracking.

    python -m tools.pose videos/test76.mov            # -> output/test76/pose.json
    python -m tools.pose videos/test76.mov --frames 80 200

Used by the 3D step to tell who touched the ball (a wrist at the ball), which side of the net
they are on (facing the camera = far side), and where they stand (ankles, before a jump). No
training: the COCO pose model (yolov8m-pose) as it comes. Uses the Mac's GPU when there is one;
~1-1.5 s per frame on a plain CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--model", default="yolov8m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--frames", nargs=2, type=int, default=None, help="first and last frame (default: all)")
    a = ap.parse_args()
    import cv2
    from ultralytics import YOLO
    stem = Path(a.video).stem
    out = paths.clip_dir(stem, a.output) / "pose.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    m = YOLO(a.model)
    import torch
    device = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else "cpu")
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        if (f - f0) % 100 == 0:
            print(f"[pose] {stem}: frame {f}/{f1}", flush=True)
    json.dump({"schema": "vv-pose/1", "video": a.video, "model": a.model, "keypoints": "coco17",
               "frames": frames}, open(out, "w"), separators=(",", ":"))
    print(f"[pose] {stem}: {len(frames)} frames  →  {out}")


if __name__ == "__main__":
    main()
