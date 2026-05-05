"""
VolleyVision — Backfill clip.json files for already-rendered videos.

If you have `output/test27_tracked.mp4` from a previous run but no
`test27.clip.json`, this script re-runs YOLO + Kalman on the **raw** source
video (in `tests/`) and writes the JSON. It does NOT re-render the tracked
video.

Usage:

    # Single clip:
    python -m integration.scripts.backfill_json tests/test27.MP4

    # All raw videos in tests/:
    python -m integration.scripts.backfill_json --all
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from collections import deque

# Resolve sibling imports — this script is meant to live next to your existing
# pipeline/ and core/ packages, OR be invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.ball_tracker import BallTracker
from integration.pipeline.export import ClipExporter, update_manifest


def backfill(input_path: str, output_dir: str = "output",
             model_path: str = "models/best3.pt", conf: float = 0.40,
             net_width_px: int | None = None, net_height_px: int | None = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    cap.release()

    stem = Path(input_path).stem
    if ok:
        cv2.imwrite(f"{output_dir}/{stem}.poster.jpg", first, [cv2.IMWRITE_JPEG_QUALITY, 80])

    px_per_m = None
    if net_width_px and net_height_px:
        px_per_m = ((net_width_px / 9.0) + (net_height_px / 1.0)) / 2

    tracker = BallTracker(model_path=model_path, conf=conf)
    exporter = ClipExporter(
        input_path=input_path, output_dir=output_dir,
        fps=fps, width=width, height=height, total_frames=total,
        px_per_m=px_per_m, net_width_px=net_width_px, net_height_px=net_height_px,
        model_path=model_path, conf=conf,
    )

    prev_pos = None
    peak_speed = 0.0
    display_speed = 0.0
    speed_buf: deque[float] = deque(maxlen=5)
    EMA = 0.15

    for frame_idx, result in enumerate(
        tracker.model.predict(source=input_path, stream=True, conf=conf, verbose=False, imgsz=1280)
    ):
        raw_det = tracker._best_detection(result)
        pos = tracker.update_from_result(result)

        speed = 0.0
        if pos is not None and prev_pos is not None:
            pix = np.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) * fps
            raw_speed = (pix / px_per_m * 3.6) if px_per_m else pix
            speed_buf.append(raw_speed)
            speed = sum(speed_buf) / len(speed_buf)
            display_speed = EMA * speed + (1 - EMA) * display_speed
            if speed > peak_speed:
                peak_speed = speed

        exporter.record(
            frame_idx=frame_idx, detection=raw_det, kalman=pos,
            prev_pos=prev_pos, speed_kmh=display_speed,
            is_lost=(pos is not None and tracker._lost > 0),
        )
        prev_pos = pos

        if frame_idx % 60 == 0:
            print(f"  {frame_idx}/{total}")

    exporter.finalize(peak_speed_kmh=peak_speed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("--all", action="store_true", help="Process all videos in tests/")
    ap.add_argument("--tests-dir", default="tests")
    ap.add_argument("--output", default="output")
    ap.add_argument("--model", default="models/best3.pt")
    args = ap.parse_args()

    if args.all:
        from glob import glob
        files = sorted(glob(f"{args.tests_dir}/*.MP4") + glob(f"{args.tests_dir}/*.mov") +
                       glob(f"{args.tests_dir}/*.MOV") + glob(f"{args.tests_dir}/*.mp4"))
        for f in files:
            print(f"\n=== {f} ===")
            try:
                backfill(f, output_dir=args.output, model_path=args.model)
            except Exception as e:
                print(f"  failed: {e}")
        update_manifest(args.output)
    elif args.input:
        backfill(args.input, output_dir=args.output, model_path=args.model)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
