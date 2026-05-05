"""
VolleyVision — Integration pipeline with JSON export.

Drop-in replacement for `pipeline/video_pipeline.py`. Same CLI behaviour,
but also writes `<stem>.clip.json` and updates `output/manifest.json` so
the frontend can load real data.
"""

import cv2
import time
import os
import shutil
import subprocess
import tempfile
import numpy as np
from collections import deque
from pathlib import Path

from core.ball_tracker import BallTracker
from integration.pipeline.export import ClipExporter


_NET_WIDTH_M  = 9.0
_NET_HEIGHT_M = 1.0


def run(
    input_path: str,
    output_dir: str = "output",
    model_path: str = "models/best4.pt",
    conf: float = 0.4,
    trail_len: int = 40,
    show_preview: bool = False,
    raw: bool = False,
    net_width_px: int | None = None,
    net_height_px: int | None = None,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first_frame = cap.read()
    cap.release()

    print(f"[VolleyVision] {Path(input_path).name}  {width}x{height}  {fps:.1f}fps  {total} frames")

    os.makedirs(output_dir, exist_ok=True)
    stem     = Path(input_path).stem
    out_path = os.path.join(output_dir, f"{stem}_tracked.mp4")

    # Save poster image from first frame
    if ok:
        poster_path = os.path.join(output_dir, f"{stem}.poster.jpg")
        cv2.imwrite(poster_path, first_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    has_ffmpeg = shutil.which("ffmpeg") is not None
    tmp_path   = os.path.join(tempfile.gettempdir(), f"{stem}_noaudio.mp4")
    write_path = tmp_path if has_ffmpeg else out_path
    writer     = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    prev_pos: tuple[float, float] | None = None
    speed: float        = 0.0
    peak_speed: float   = 0.0
    speed_buf: deque[float] = deque(maxlen=5)
    display_speed: float = 0.0
    _EMA_ALPHA = 0.15

    # Calibration
    px_per_m: float | None = None
    if net_width_px and net_height_px:
        px_per_m_x = net_width_px  / _NET_WIDTH_M
        px_per_m_y = net_height_px / _NET_HEIGHT_M
        px_per_m   = (px_per_m_x + px_per_m_y) / 2
        print(f"[VolleyVision] Calibration: {px_per_m:.1f} px/m  ({px_per_m_x:.1f} horiz, {px_per_m_y:.1f} vert)")
    elif ok:
        px_per_m = _interactive_calibrate(first_frame)

    # Optional ignore regions (e.g. wall painting) — only in interactive mode
    ignore_regions = []
    if ok and net_width_px is None:
        ignore_regions = _interactive_ignore_regions(first_frame)

    tracker = BallTracker(model_path=model_path, conf=conf, trail_len=trail_len,
                          ignore_regions=ignore_regions)

    # JSON exporter
    exporter = ClipExporter(
        input_path=input_path, output_dir=output_dir,
        fps=fps, width=width, height=height, total_frames=total,
        px_per_m=px_per_m, net_width_px=net_width_px, net_height_px=net_height_px,
        model_path=model_path, conf=conf,
    )

    t0 = time.time()
    for frame_idx, result in enumerate(
        tracker.model.predict(source=input_path, stream=True, conf=conf, verbose=False, imgsz=1280)
    ):
        frame = result.orig_img

        if raw:
            out = result.plot()
        else:
            raw_det = tracker._best_detection(result)
            pos     = tracker.update_from_result(result)

            if pos is not None and prev_pos is not None:
                pixel_speed  = np.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) * fps
                raw_speed    = (pixel_speed / px_per_m * 3.6) if px_per_m else pixel_speed
                speed_buf.append(raw_speed)
                speed         = sum(speed_buf) / len(speed_buf)
                display_speed = _EMA_ALPHA * speed + (1 - _EMA_ALPHA) * display_speed
                if speed > peak_speed:
                    peak_speed = speed
            elif pos is None:
                display_speed = _EMA_ALPHA * 0 + (1 - _EMA_ALPHA) * display_speed

            exporter.record(
                frame_idx=frame_idx,
                detection=raw_det,
                kalman=pos,
                prev_pos=prev_pos,
                speed_kmh=display_speed,
                is_lost=(pos is not None and tracker._lost > 0),
            )

            prev_pos = pos
            out = tracker.draw(frame)
            out = _hud(out, frame_idx, total, fps, pos, display_speed, peak_speed,
                       calibrated=px_per_m is not None)

        writer.write(out)

        if show_preview:
            cv2.imshow("VolleyVision", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % 60 == 0:
            pct = frame_idx / total * 100
            print(f"  {pct:5.1f}%  frame {frame_idx}/{total}  [{time.time()-t0:.1f}s]")

    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    # Write JSON export
    exporter.finalize(peak_speed_kmh=peak_speed)

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
        print(f"[VolleyVision] Transcoded to H.264 + audio muxed ✓")
    else:
        print(f"[VolleyVision] ffmpeg not found — audio not preserved")

    print(f"[VolleyVision] Done in {time.time()-t0:.1f}s  →  {out_path}")
    return out_path


def _interactive_calibrate(frame) -> float | None:
    """Show first frame and collect 4 clicks: 2 for net width, 2 for net height.
    Returns px/m or None if user skips."""
    clicks = []
    stage  = [0]
    labels = [
        "Click LEFT pole, then RIGHT pole  (net width = 9m)",
        "Click TOP of net, then BOTTOM     (net height = 1m)",
    ]

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicks.append((x, y))
        cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
        if len(clicks) % 2 == 0:
            p1, p2 = clicks[-2], clicks[-1]
            cv2.line(display, p1, p2, (0, 255, 0), 2)
            stage[0] += 1

    display = frame.copy()
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", on_click)
    print("[Calibration] Press S to skip, ENTER when done.")

    while True:
        hint = display.copy()
        if stage[0] < 2:
            cv2.putText(hint, labels[stage[0]], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(hint, "Press ENTER to confirm", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)
        cv2.imshow("Calibration", hint)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(clicks) >= 4:
            break
        if key == ord("s"):
            cv2.destroyWindow("Calibration")
            print("[Calibration] Skipped — speed shown in px/s")
            return None

    cv2.destroyWindow("Calibration")
    width_px   = np.hypot(clicks[1][0] - clicks[0][0], clicks[1][1] - clicks[0][1])
    height_px  = np.hypot(clicks[3][0] - clicks[2][0], clicks[3][1] - clicks[2][1])
    px_per_m_x = width_px  / _NET_WIDTH_M
    px_per_m_y = height_px / _NET_HEIGHT_M
    px_per_m   = (px_per_m_x + px_per_m_y) / 2
    print(f"[Calibration] {px_per_m:.1f} px/m  (horiz: {px_per_m_x:.1f}, vert: {px_per_m_y:.1f})")
    return px_per_m


def _interactive_ignore_regions(frame) -> list[tuple[int, int, int, int]]:
    """Let the user drag rectangles over false-positive regions (e.g. wall painting).
    Press S to skip, ENTER or C to confirm and continue."""
    regions = []
    drawing = [False]
    start   = [0, 0]
    current = [0, 0]
    display = frame.copy()
    base    = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            start[0], start[1] = x, y
            current[0], current[1] = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            current[0], current[1] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            x1, y1 = min(start[0], x), min(start[1], y)
            x2, y2 = max(start[0], x), max(start[1], y)
            if x2 - x1 > 5 and y2 - y1 > 5:
                regions.append((x1, y1, x2, y2))

    cv2.namedWindow("Ignore Regions", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Ignore Regions", on_mouse)
    print("[Ignore Regions] Drag boxes over false detections. Press S to skip, ENTER when done.")

    while True:
        display = base.copy()
        # Draw confirmed regions
        for (x1, y1, x2, y2) in regions:
            cv2.rectangle(display, (x1, y1), (x2, y2), (50, 50, 220), 2)
            cv2.putText(display, "ignored", (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 220), 1, cv2.LINE_AA)
        # Draw in-progress rectangle
        if drawing[0]:
            cv2.rectangle(display, (start[0], start[1]), (current[0], current[1]),
                          (50, 50, 220), 1)
        hint = f"Drag to mask | {len(regions)} region(s) | ENTER=confirm  S=skip  Z=undo"
        cv2.putText(display, hint, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow("Ignore Regions", display)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("c")):   # ENTER or C → confirm
            break
        if key == ord("s"):          # S → skip entirely
            regions.clear()
            break
        if key == ord("z") and regions:  # Z → undo last region
            regions.pop()

    cv2.destroyWindow("Ignore Regions")
    if regions:
        print(f"[Ignore Regions] {len(regions)} region(s) masked: {regions}")
    else:
        print("[Ignore Regions] Skipped — no regions masked")
    return regions


def _hud(frame, idx, total, fps, pos, speed: float = 0.0, peak_speed: float = 0.0,
         calibrated: bool = False):
    unit = "km/h" if calibrated else "px/s"
    fmt  = ".1f" if calibrated else ".0f"
    lines = [
        ("VolleyVision",                           (0, 220, 255)),
        (f"Frame  {idx}/{total}",                  (200, 200, 200)),
        (f"Time   {idx/fps:.2f}s",                 (200, 200, 200)),
        (f"Ball   {'TRACKED' if pos else 'LOST'}", (0, 255, 100) if pos else (60, 60, 255)),
        (f"Speed  {speed:{fmt}} {unit}",           (0, 220, 255) if speed > 0 else (200, 200, 200)),
        (f"Peak   {peak_speed:{fmt}} {unit}",      (0, 255, 150)),
    ]
    pad, line_h, font_scale, thickness = 18, 36, 1.0, 2
    box_w = 360
    box_h = pad * 2 + len(lines) * line_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (16, 8 + pad + line_h + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return frame
