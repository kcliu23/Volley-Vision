"""Drawing + encoding shared by the pipeline and the dev tools."""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2


def draw_trail(img, pts, smooth: int = 3):
    """Fading trail through recent positions (None breaks the line). Same look as BallTracker.draw."""
    total = len([p for p in pts if p is not None])
    drawn = 0

    def flush(seg):
        nonlocal drawn
        if len(seg) < 2:
            return
        h = smooth // 2
        sm = []
        for i in range(len(seg)):
            w = seg[max(0, i - h):i + h + 1]
            sm.append((sum(q[0] for q in w) / len(w), sum(q[1] for q in w) / len(w)))
        for i in range(1, len(sm)):
            drawn += 1
            a = drawn / max(total, 1)
            cv2.line(img, (int(sm[i - 1][0]), int(sm[i - 1][1])), (int(sm[i][0]), int(sm[i][1])),
                     (int(255 * a), int(200 * (1 - a)), 30), max(1, int(3 * a)), cv2.LINE_AA)

    seg = []
    for p in pts:
        if p is None:
            flush(seg)
            seg = []
        else:
            seg.append(p)
    flush(seg)


def draw_ball(img, p, ring: bool = False):
    cx, cy = int(p[0]), int(p[1])
    if ring:
        cv2.circle(img, (cx, cy), 22, (0, 255, 255), 3, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), 12, (0, 180, 180), 2)


def draw_touch_rings(img, f: int, touches, fps: float, hold_s: float = 0.5, color=None):
    """Expanding, fading ring at each contact for hold_s after the contact frame.
    Default colour (orange) for touches; pass color for other kinds (e.g. white = bounce)."""
    hold = max(1, int(round(hold_s * fps)))
    scale = img.shape[0] / 1080
    for t in touches:
        t0 = int(round(t.frame))
        if not (t0 <= f < t0 + hold):
            continue
        fade = 1 - (f - t0) / hold
        c = color if color is not None else (0, int(80 + 175 * fade), 255)
        cv2.circle(img, (int(t.x), int(t.y)), int((28 + 30 * (1 - fade)) * scale), c,
                   max(2, int(4 * scale)), cv2.LINE_AA)


def finish_video(tmp_path: str, source_video: str, out_path: str) -> bool:
    """H.264 + original audio via ffmpeg if available, else move the OpenCV file. Returns True if ffmpeg ran."""
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", tmp_path, "-i", source_video, "-map", "0:v:0", "-map", "1:a?",
                        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", out_path], check=True, capture_output=True)
        os.remove(tmp_path)
        return True
    shutil.move(tmp_path, out_path)
    return False
