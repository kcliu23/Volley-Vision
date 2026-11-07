"""Detect and track the people in a clip (for the 3D step: where players stand, when they jump,
who touches the ball).

    python -m tools.players videos/test74.mov [--model yolov8m.pt] [--imgsz 1280]

A COCO person detector (Ultralytics) with its ByteTrack tracker, every frame. Writes
output/<clip>/players.json:
  {"frames": [[[id, x1, y1, x2, y2, conf], ...], ...]}      one list per frame (id -1 = untracked)
Cached: delete the file to redo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402


def detect(video: str, out: Path, model: str = "yolov8m.pt", imgsz: int = 1280, conf: float = 0.25) -> Path:
    from ultralytics import YOLO
    from core.device import yolo_device
    m = YOLO(model)
    frames, t0 = [], time.time()
    for i, r in enumerate(m.track(source=video, stream=True, persist=True, classes=[0], conf=conf, imgsz=imgsz,
                                  tracker="bytetrack.yaml", verbose=False, device=yolo_device())):
        row = []
        ids = r.boxes.id.tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
        for b, tid in zip(r.boxes, ids):
            x1, y1, x2, y2 = (round(float(v), 1) for v in b.xyxy[0])
            row.append([int(tid), x1, y1, x2, y2, round(float(b.conf[0]), 3)])
        frames.append(row)
        if (i + 1) % 100 == 0:
            print(f"  people: {i + 1} frames [{time.time() - t0:.0f}s]", flush=True)
    json.dump({"schema": "vv-players/1", "video": video, "model": model, "imgsz": imgsz, "frames": frames},
              open(paths.ensure(out), "w"), separators=(",", ":"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    stem = Path(a.video).stem
    out = Path(a.output) / stem / "players.json"
    if out.exists():
        print(f"[players] {stem}: cached  →  {out}")
        return
    detect(a.video, out, a.model, a.imgsz)
    print(f"[players] {stem}  →  {out}")


if __name__ == "__main__":
    main()
