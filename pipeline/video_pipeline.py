import cv2
import time
import os
from pathlib import Path

from core.ball_tracker import BallTracker


def run(
    input_path: str,
    output_dir: str = "output",
    model_path: str = "models/YOLOv26n.pt",
    conf: float = 0.35,
    trail_len: int = 40,
    show_preview: bool = False,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VolleyVision] {Path(input_path).name}  {width}x{height}  {fps:.1f}fps  {total} frames")

    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_path).stem
    out_path = os.path.join(output_dir, f"{stem}_tracked.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = BallTracker(model_path=model_path, conf=conf, trail_len=trail_len)

    t0 = time.time()
    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        pos = tracker.update(frame)
        out = tracker.draw(frame)
        out = _hud(out, frame_idx, total, fps, pos)
        writer.write(out)

        if show_preview:
            cv2.imshow("VolleyVision", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 60 == 0:
            pct = frame_idx / total * 100
            print(f"  {pct:5.1f}%  frame {frame_idx}/{total}  [{time.time()-t0:.1f}s]")

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    print(f"[VolleyVision] Done in {time.time()-t0:.1f}s  →  {out_path}")
    return out_path


def _hud(frame, idx, total, fps, pos):
    h, w = frame.shape[:2]
    lines = [
        ("VolleyVision",                       (0, 220, 255)),
        (f"Frame  {idx}/{total}",              (200, 200, 200)),
        (f"Time   {idx/fps:.2f}s",             (200, 200, 200)),
        (f"Ball   {'TRACKED' if pos else 'LOST'}", (0,255,100) if pos else (60,60,255)),
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (240, 88), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (16, 28 + i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return frame
