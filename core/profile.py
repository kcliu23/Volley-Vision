"""Clip profile: everything about a video that the tracker must normalize for.

Computed once per clip and cached in the detections file (dets["profile"]):

    times       real timestamp of every decoded frame, seconds (handles uneven phone timing)
    fps         nominal frame rate (median of 1 / frame interval)
    dup         frame repeats the previous image (screen recordings, frame-rate conversion)
    content     picture area inside the frame (black/gray bars excluded)
    ball_px     typical ball size in pixels: the unit of distance ("ball-widths",
                1 ball-width ≈ 0.21 m at the ball's own depth)
    camera      per-frame global image shift (px) and whether the camera is static

Everything downstream works in seconds and ball-widths, so the same settings apply
at 30, 60 or 120 fps and at 720p, 1080p or 4K.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import cv2
import numpy as np

PROFILE_VERSION = 1


def frame_times(path: str, n_expected: int | None = None) -> list[float]:
    """Presentation timestamp of every frame, seconds from the first frame.
    ffprobe if available (exact), else OpenCV's position property."""
    times = []
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "frame=best_effort_timestamp_time", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=300).stdout
            times = [float(x.strip().rstrip(",")) for x in out.splitlines() if x.strip() and x.strip() != "N/A"]
        except Exception:
            times = []
    if not times or (n_expected and abs(len(times) - n_expected) > 2):
        cap = cv2.VideoCapture(path)
        times = []
        while True:
            ok = cap.grab()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        cap.release()
    if not times:
        return []
    t0 = times[0]
    times = [t - t0 for t in times]
    # enforce monotonic (some containers repeat a timestamp)
    for i in range(1, len(times)):
        if times[i] <= times[i - 1]:
            times[i] = times[i - 1] + 1e-4
    if n_expected:
        if len(times) > n_expected:
            times = times[:n_expected]
        while len(times) < n_expected:
            dt = (times[-1] - times[-2]) if len(times) > 1 else 1 / 30
            times.append(times[-1] + dt)
    return times


def camera_motion(path: str, step: int = 1, width: int = 480):
    """Per-frame global translation (dx, dy) in full-resolution px, by phase correlation
    on downscaled gray frames. Small values everywhere = static camera."""
    cap = cv2.VideoCapture(path)
    prev, shifts, scale = None, [], None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if scale is None:
            scale = img.shape[1] / width
        g = cv2.cvtColor(cv2.resize(img, (width, int(img.shape[0] / scale))), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is None:
            shifts.append((0.0, 0.0))
        else:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g)
            shifts.append((float(dx * scale), float(dy * scale)) if resp > 0.05 else (0.0, 0.0))
        prev = g
    cap.release()
    return shifts


def build(path: str, dets: dict, with_camera: bool = True) -> dict:
    """Profile for a clip; reuses dup flags / content box already in dets if present."""
    from core.refine import video_dup_flags, content_box
    n = len(dets["frames"])
    times = frame_times(path, n)
    dts = np.diff(times) if len(times) > 1 else np.array([1 / (dets.get("fps") or 30)])
    dts_real = dts[dts > 1e-3]
    fps = float(1.0 / np.median(dts_real)) if len(dts_real) else float(dets.get("fps") or 30)
    dup = dets.get("dup") if len(dets.get("dup") or []) == n else video_dup_flags(path)
    content = dets.get("content") or content_box(path)
    sizes = [max(c[2], c[3]) for f in dets["frames"] for c in f if c[4] >= 0.4]
    ball_px = float(np.median(sizes)) if sizes else 30.0
    prof = {"version": PROFILE_VERSION, "fps": round(fps, 3), "times": [round(t, 5) for t in times],
            "dup": dup, "content": content, "ball_px": round(ball_px, 2),
            "uneven_timing": bool(len(dts_real) and np.std(dts_real) / np.median(dts_real) > 0.15)}
    if with_camera:
        sh = camera_motion(path)
        mags = np.hypot(*np.array(sh).T) if sh else np.zeros(1)
        prof["camera"] = {"shift": [[round(a, 2), round(b, 2)] for a, b in sh],
                          "p95_shift_bw": round(float(np.percentile(mags, 95)) / ball_px, 3),
                          "static": bool(np.percentile(mags, 95) < 0.15 * ball_px)}
    dets["dup"], dets["content"] = dup, content
    return prof


def get(path: str, dets: dict) -> dict:
    """Cached profile (computed on first use)."""
    prof = dets.get("profile")
    if not prof or prof.get("version") != PROFILE_VERSION or len(prof.get("times", [])) != len(dets["frames"]):
        prof = build(path, dets)
        dets["profile"] = prof
    return prof


def tau(prof: dict) -> np.ndarray:
    """Real time expressed in nominal frames: tau[f] = times[f] * fps.
    Equals the frame index for perfectly even video; differs where timing is uneven."""
    return np.asarray(prof["times"], float) * prof["fps"]


def link_args(dets: dict) -> dict:
    """Keyword args for core.linker.link / cost_linker.link from a dets dict, using the
    cached profile when present (real timestamps, nominal fps), else the raw fields."""
    prof = dets.get("profile")
    if prof and len(prof.get("times", [])) == len(dets["frames"]):
        return {"fps": prof["fps"], "dup_flags": prof["dup"], "content": prof["content"], "times": prof["times"]}
    dup = dets.get("dup") if len(dets.get("dup") or []) == len(dets["frames"]) else None
    return {"fps": dets["fps"], "dup_flags": dup, "content": dets.get("content"), "times": None}
