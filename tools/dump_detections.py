"""Step 1 of the two-pass tracker: run YOLO once and keep EVERY ball candidate.

No confidence floor beyond --conf, no size filter, no gating — those are
decisions for the linking step, which can then be tuned without re-running
the detector.

    python -m tools.dump_detections videos/test42.MP4
    python -m tools.dump_detections videos/*.mov videos/*.MP4 --conf 0.05

Writes output/<stem>/cache/dets.json (see core.paths):
    {"schema": "vv-dets/1", "fps", "width", "height", "imgsz", "conf",
     "frames": [[[cx, cy, w, h, conf], ...], ...],      # one list per frame
     "dup": [false, true, ...]}                          # frame repeats the previous image
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dump(video: str, model, out_path: Path, conf: float, imgsz: int) -> Path:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    from core.refine import small_gray, frame_changed_pixels, DUP_MAX_CHANGED
    frames, dup, prev = [], [], None
    t0 = time.time()
    for i, r in enumerate(model.predict(source=video, stream=True, conf=conf, imgsz=imgsz, verbose=False)):
        boxes = []
        for b in r.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            boxes.append([round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1),
                          round(x2 - x1, 1), round(y2 - y1, 1), round(float(b.conf[0]), 3)])
        frames.append(boxes)
        g = small_gray(r.orig_img)
        dup.append(prev is not None and frame_changed_pixels(prev, g) < DUP_MAX_CHANGED)
        prev = g
        if i % 100 == 0:
            print(f"  {Path(video).name}: {i}/{total}  [{time.time() - t0:.0f}s]", flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"schema": "vv-dets/1", "video": video, "fps": fps, "width": w, "height": h,
                   "imgsz": imgsz, "conf": conf, "frames": frames, "dup": dup}, fh, separators=(",", ":"))
    n = sum(len(b) for b in frames)
    print(f"[dump] {Path(video).name}: {len(frames)} frames, {n} candidates  →  {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="models/best4.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--output", default="output", help="output root (files go to <root>/<clip>/cache/dets.json)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    from core import paths
    for v in args.videos:
        out = paths.dets(Path(v).stem, args.output)
        if args.skip_existing and out.exists():
            continue
        try:
            dump(v, model, out, args.conf, args.imgsz)
        except Exception as e:  # keep going on a bad file
            print(f"[dump] {v}: FAILED {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
