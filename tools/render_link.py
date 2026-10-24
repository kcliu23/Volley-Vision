"""Render a video from a link result (same trail style as BallTracker.draw).

    python -m tools.render_link output/dev/test42_link_rule.json tests/test42.MP4 out.mp4 [--side-by-side old_tracked.mp4]
Filled (interpolated) frames are drawn the same as detected ones; pass --mark-filled
to draw them as a hollow marker instead, for debugging.
"""
import argparse, json, os, shutil, subprocess, tempfile
import cv2
import numpy as np


def draw_trail(img, pts, trail_len=40, smooth=3):
    total = len([p for p in pts if p is not None])
    drawn = 0
    seg = []
    def flush(seg):
        nonlocal drawn
        if len(seg) < 2:
            return
        h = smooth // 2
        sm = [(sum(q[0] for q in seg[max(0, i - h):i + h + 1]) / len(seg[max(0, i - h):i + h + 1]),
               sum(q[1] for q in seg[max(0, i - h):i + h + 1]) / len(seg[max(0, i - h):i + h + 1]))
              for i in range(len(seg))]
        for i in range(1, len(sm)):
            drawn += 1
            a = drawn / max(total, 1)
            cv2.line(img, (int(sm[i-1][0]), int(sm[i-1][1])), (int(sm[i][0]), int(sm[i][1])),
                     (int(255 * a), int(200 * (1 - a)), 30), max(1, int(3 * a)), cv2.LINE_AA)
    for p in pts:
        if p is None:
            flush(seg); seg = []
        else:
            seg.append(p)
    flush(seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("link"); ap.add_argument("video"); ap.add_argument("out")
    ap.add_argument("--trail-len", type=int, default=40)
    ap.add_argument("--mark-filled", action="store_true")
    ap.add_argument("--side-by-side", help="another rendered video to put on the left (e.g. old tracker)")
    a = ap.parse_args()
    d = json.load(open(a.link)); R = d["frames"]
    cap = cv2.VideoCapture(a.video)
    other = cv2.VideoCapture(a.side_by_side) if a.side_by_side else None
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(3)), int(cap.get(4))
    outW = W * 2 if other else W
    scale = 0.5 if other else 1.0
    tmp = os.path.join(tempfile.gettempdir(), "render_link_tmp.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(outW * scale), int(H * scale)))
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        lo = max(0, f - a.trail_len + 1)
        pts = [tuple(R[i]["pos"]) if i < len(R) and "pos" in R[i] else None for i in range(lo, f + 1)]
        draw_trail(img, pts, a.trail_len)
        if f < len(R) and "pos" in R[f]:
            cx, cy = (int(v) for v in R[f]["pos"])
            if a.mark_filled and R[f]["state"] == "filled":
                cv2.circle(img, (cx, cy), 12, (255, 0, 255), 2)
            else:
                cv2.circle(img, (cx, cy), 10, (0, 255, 255), -1)
                cv2.circle(img, (cx, cy), 12, (0, 180, 180), 2)
        if other:
            ok2, left = other.read()
            if not ok2:
                left = np.zeros_like(img)
            left = cv2.resize(left, (W, H))
            cv2.putText(left, "current", (30, 70), 0, 2, (255, 255, 255), 4)
            cv2.putText(img, "linker", (30, 70), 0, 2, (255, 255, 255), 4)
            img = np.hstack([left, img])
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale)
        wr.write(img)
        f += 1
    wr.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                        "-pix_fmt", "yuv420p", a.out], check=True, capture_output=True)
        os.remove(tmp)
    else:
        shutil.move(tmp, a.out)
    print("→", a.out)


if __name__ == "__main__":
    main()
