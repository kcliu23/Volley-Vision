import cv2
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from core.ball_tracker import BallTracker


def run(
    input_path: str,
    output_dir: str = "output",
    model_path: str = "models/best4.pt",
    conf: float = 0.4,
    trail_len: int = 40,
    show_preview: bool = False,
    raw: bool = False,
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

    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_path).stem
    out_path = os.path.join(output_dir, f"{stem}_tracked.mp4")

    # Write to a temp file first when ffmpeg is available, then transcode to H.264 + audio.
    has_ffmpeg = shutil.which("ffmpeg") is not None
    tmp_path   = os.path.join(tempfile.gettempdir(), f"{stem}_noaudio.mp4")
    write_path = tmp_path if has_ffmpeg else out_path
    writer     = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = BallTracker(model_path=model_path, conf=conf, trail_len=trail_len)

    t0 = time.time()
    for frame_idx, result in enumerate(
        tracker.model.predict(source=input_path, stream=True, conf=conf, verbose=False, imgsz=1280)
    ):
        if raw:
            out = result.plot()
        else:
            tracker.update_from_result(result)
            out = tracker.draw(result.orig_img)

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

    if has_ffmpeg:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c:v", "libx264", "-crf", "23", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            out_path,
        ], check=True, capture_output=True)
        os.remove(tmp_path)
        print("[VolleyVision] Transcoded to H.264 + audio muxed")
    else:
        print("[VolleyVision] ffmpeg not found — audio not preserved")

    print(f"[VolleyVision] Done in {time.time()-t0:.1f}s  →  {out_path}")
    return out_path
