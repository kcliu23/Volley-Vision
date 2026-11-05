"""Check the 3D ball by eye: the video with the ball's floor "shadow" drawn under it.

    python -m tools.ball3d_video videos/test74.mov        # needs output/<clip>/ball3d.json (tools.ball3d_run)

Per frame: the ball (yellow circle), a line straight down to the floor point under it (red
cross: where the 3D fit says the ball is over the court), and its height in meters. If the
depth is right, the red cross runs under the player at every touch and sits on the bounce spot
when the ball lands. Only free flights are drawn (held / rolling / badly fitted ones are left
out). A mini top view (court + ball + camera) sits in the corner.
Writes output/<clip>/ball3d.mp4.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402
from core.camera import Camera  # noqa: E402
from core.court import LENGTH, WIDTH, ATTACK  # noqa: E402


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    stem = Path(a.video).stem
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    B = json.load(open(Path(a.output) / stem / "ball3d.json"))
    frames = {int(k): v for k, v in B["frames"].items() if v.get("free")}
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or B["fps"]
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = Path(a.output) / stem / "ball3d_tmp.mp4"
    out = Path(a.output) / stem / "ball3d.mp4"
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    # mini top view
    S = 22                                   # px per meter
    mw, mh = int(28 * S), int(14 * S)
    def top(x, y):
        return int((x + 15) * S), int((7 - y) * S)
    base = np.full((mh, mw, 3), 40, np.uint8)
    hx, hy = LENGTH / 2, WIDTH / 2
    for p, q in (((-hx, -hy), (hx, -hy)), ((-hx, hy), (hx, hy)), ((-hx, -hy), (-hx, hy)), ((hx, -hy), (hx, hy)),
                 ((0, -hy), (0, hy)), ((-ATTACK, -hy), (-ATTACK, hy)), ((ATTACK, -hy), (ATTACK, hy))):
        cv2.line(base, top(*p), top(*q), (90, 200, 90), 1, cv2.LINE_AA)
    cv2.line(base, top(0, -hy - 0.5), top(0, hy + 0.5), (230, 230, 230), 2)
    c = cam.position
    cv2.circle(base, top(c[0], c[1]), 5, (60, 60, 230), -1)
    trail: list = []
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        mini = base.copy()
        fr = frames.get(f)
        if fr is not None:
            X = np.array(fr["xyz"])
            u = cam.project([X])[0]
            s = cam.project([[X[0], X[1], 0.0]])[0]
            cv2.line(img, (int(u[0]), int(u[1])), (int(s[0]), int(s[1])), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(img, (int(s[0]), int(s[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 4)
            cv2.circle(img, (int(u[0]), int(u[1])), 24, (0, 255, 255), 3, cv2.LINE_AA)
            sd = np.linalg.norm(fr.get("std") or [0, 0, 0])
            cv2.putText(img, f"{X[2]:.1f} m", (int(u[0]) + 30, int(u[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                        (0, 255, 255), 4, cv2.LINE_AA)
            trail.append((X[0], X[1]))
        else:
            trail.append(None)
        trail = trail[-45:]
        pts = [p for p in trail if p is not None]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(mini, top(*p), top(*q), (0, 200, 255), 2, cv2.LINE_AA)
        if fr is not None:
            cv2.circle(mini, top(X[0], X[1]), 6, (0, 255, 255), -1)
        img[H - mh - 10:H - 10, W - mw - 250:W - 250] = mini
        cv2.putText(img, f"frame {f}", (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        vw.write(img)
        f += 1
    vw.release()
    cap.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "26", "-preset", "fast", str(out)], check=False)
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(out)
    print(f"[ball3d] {stem}: check video  →  {out}")


if __name__ == "__main__":
    main()
