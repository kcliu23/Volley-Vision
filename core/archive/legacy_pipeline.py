"""ARCHIVED 2026-10-27: the original single-pass tracker (YOLO conf >= 0.40 + Kalman) and its
touch overlay, removed from core/video_pipeline.py (main.py --legacy). Needs
core/archive/ball_tracker.py and core/archive/touch_detector.py on the import path."""
import cv2
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from core.archive.ball_tracker import BallTracker
from core.archive.touch_detector import detect_touches, Touch


def run_legacy(
    input_path: str,
    output_dir: str = "output",
    model_path: str = "models/best4.pt",
    conf: float = 0.4,
    trail_len: int = 40,
    show_preview: bool = False,
    raw: bool = False,
    touches: bool = True,
) -> str:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"[VolleyVision] {Path(input_path).name}  {width}x{height}  {fps:.1f}fps  {total} frames")

    from core import paths
    stem     = Path(input_path).stem
    cdir     = paths.clip_dir(stem, output_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    out_path = str(cdir / "legacy_tracked.mp4")

    # Always render to a temp file first: touches are only known after the
    # whole clip is tracked, so they're overlaid in a second (cheap, no-YOLO)
    # pass, then the result is transcoded with ffmpeg if available.
    has_ffmpeg = shutil.which("ffmpeg") is not None
    tmp_path   = os.path.join(tempfile.gettempdir(), f"{stem}_noaudio.mp4")
    writer     = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = BallTracker(model_path=model_path, conf=conf, trail_len=trail_len)
    track_rows = []

    t0 = time.time()
    for frame_idx, result in enumerate(
        tracker.model.predict(source=input_path, stream=True, conf=conf, verbose=False, imgsz=1280)
    ):
        if raw:
            out = result.plot()
        else:
            pos = tracker.update_from_result(result)
            out = tracker.draw(result.orig_img)
            row = {"f": frame_idx}
            if tracker.last_det is not None:
                row["det"] = [round(tracker.last_det[0], 1), round(tracker.last_det[1], 1)]
                row["conf"] = round(tracker.last_conf, 3)
            if pos is not None:
                row["pos"] = [round(pos[0], 1), round(pos[1], 1)]
            track_rows.append(row)

        writer.write(out)

        if show_preview:
            cv2.imshow("VolleyVision", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 60 == 0:
            print(f"  {frame_idx / max(total, 1) * 100:5.1f}%  frame {frame_idx}/{total}  [{time.time()-t0:.1f}s]")

    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    render_path = tmp_path
    if not raw:
        track_path = str(cdir / "legacy_track.json")
        with open(track_path, "w") as fh:
            json.dump({"schema": "vv-track/1", "video": str(input_path), "fps": fps,
                       "width": width, "height": height, "frames": track_rows}, fh)
        print(f"[VolleyVision] Trajectory  →  {track_path}")

        if touches:
            pts = [(r["f"], *r["det"]) for r in track_rows if "det" in r]
            found = detect_touches(*map(np.array, zip(*pts)), fps, height) if pts else []
            touch_path = str(cdir / "legacy_touches.json")
            with open(touch_path, "w") as fh:
                json.dump({"schema": "vv-touches/1", "video": str(input_path), "fps": fps,
                           "touches": [t.to_dict() for t in found]}, fh, indent=1)
            print(f"[VolleyVision] {len(found)} touches  →  {touch_path}")
            for t in found:
                print(f"    {t.t:6.2f}s  frame {t.frame:5d}  turn {t.angle_deg:3.0f}°  "
                      f"{t.speed_in:5.0f} → {t.speed_out:5.0f} px/s{'  (during gap)' if t.in_gap else ''}")
            if found:
                annotated = os.path.join(tempfile.gettempdir(), f"{stem}_touches.mp4")
                overlay_touches(tmp_path, annotated, found, fps)
                os.remove(tmp_path)
                render_path = annotated

    if has_ffmpeg:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", render_path,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            out_path,
        ], check=True, capture_output=True)
        os.remove(render_path)
        print("[VolleyVision] Transcoded to H.264 + audio muxed")
    else:
        shutil.move(render_path, out_path)
        print("[VolleyVision] ffmpeg not found — audio not preserved")

    print(f"[VolleyVision] Done in {time.time()-t0:.1f}s  →  {out_path}")
    return out_path


def overlay_touches(src: str, dst: str, touches: list[Touch], fps: float, hold_s: float = 0.5):
    """Re-read a rendered video and mark each touch with an expanding, fading
    ring at the contact point, held for hold_s after the contact frame.
    Touch data (count, timing, velocities) lives in <stem>_touches.json only."""
    cap = cv2.VideoCapture(src)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    hold = max(1, int(round(hold_s * fps)))
    scale = h / 1080
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for t in touches:
            if not (t.frame <= f < t.frame + hold):
                continue
            fade = 1 - (f - t.frame) / hold
            c = (0, int(80 + 175 * fade), 255)
            cv2.circle(img, (int(t.x), int(t.y)), int((28 + 30 * (1 - fade)) * scale), c,
                       max(2, int(4 * scale)), cv2.LINE_AA)
        writer.write(img)
        f += 1
    cap.release()
    writer.release()


