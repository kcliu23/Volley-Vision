"""Try the two-pass tracker (detect -> link -> zoom search) on any video.

    python -m tools.dev.try_linker videos/test10.mov
    python -m tools.dev.try_linker videos/test2*.MP4 videos/test3*.mov
    python -m tools.dev.try_linker videos/test10.mov --ring          # hollow marker, ball stays visible
    python -m tools.dev.try_linker videos/test10.mov --linker cost   # cost-based linker instead of the rule-based one

For each video:
  1. YOLO detection pass (skipped if output/<clip>/cache/dets.json exists)
  2. linker + zoom search  ->  output/dev/<clip>_link_<linker>.json
  3. side-by-side video    ->  output/dev/<clip>_compare.mp4
     left: current tracker (same detections, conf >= 0.40)   right: linker
Prints old vs new coverage (share of frames with a ball position).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.linker import LinkParams  # noqa: E402
from core.refine import link_with_zoom  # noqa: E402
from tools.compare_tracking import run_old  # noqa: E402
from tools.dev.render_link import draw_trail  # noqa: E402


def marker(img, p, ring, color=(0, 255, 255)):
    cx, cy = int(p[0]), int(p[1])
    if ring:
        cv2.circle(img, (cx, cy), 22, color, 3, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), 10, (0, 255, 255), -1)
        cv2.circle(img, (cx, cy), 12, (0, 180, 180), 2)


def render(video, old_pos, new_pos, out, ring=False, trail_len=40, new_label="linker"):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    tmp = os.path.join(tempfile.gettempdir(), f"{Path(out).stem}_tmp.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H // 2))
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        panes = []
        for label, P in (("current", old_pos), (new_label, new_pos)):
            im = img.copy()
            lo = max(0, f - trail_len + 1)
            draw_trail(im, [P[i] if i < len(P) else None for i in range(lo, f + 1)], trail_len)
            if f < len(P) and P[f] is not None:
                marker(im, P[f], ring)
            cv2.putText(im, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 255, 255), 6, cv2.LINE_AA)
            panes.append(cv2.resize(im, (W // 2, H // 2)))
        wr.write(np.hstack(panes))
        f += 1
    wr.release()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                        "-pix_fmt", "yuv420p", out], check=True, capture_output=True)
        os.remove(tmp)
    else:
        shutil.move(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="models/best4.pt")
    ap.add_argument("--ring", action="store_true", help="hollow ring marker instead of a solid dot")
    ap.add_argument("--no-video", action="store_true", help="only compute + print coverage")
    ap.add_argument("--linker", choices=["rule", "cost"], default="rule",
                    help="rule: step-by-step rules (core/linker.py); cost: one global optimization (core/cost_linker.py)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    from core import paths
    rows = []
    for v in args.videos:
        stem = Path(v).stem
        dpath = paths.ensure(paths.dets(stem))
        if not dpath.exists():
            from tools.dump_detections import dump
            print(f"[{stem}] detection pass (first time only)...")
            dump(v, model, dpath, 0.05, 1280)
        dets = json.load(open(dpath))
        print(f"[{stem}] linking...")
        track = link_with_zoom(dets, v, model, LinkParams(), linker=args.linker)
        json.dump(dets, open(dpath, "w"), separators=(",", ":"))       # caches zoom + repeat-frame flags
        json.dump({"schema": "vv-link/1", "video": v, "fps": dets["fps"], "width": dets["width"],
                   "height": dets["height"], "frames": track.to_rows()}, open(paths.ensure(paths.dev() / f"{stem}_link_{args.linker}.json"), "w"))
        old_state, old_pos = run_old(dets["frames"], with_pos=True)
        o_cov = 1 - old_state.count("none") / len(old_state)
        s = track.summary()
        rows.append((stem, len(old_state), o_cov, s))
        if not args.no_video:
            out = f"output/dev/{stem}_compare_{args.linker}.mp4"
            render(v, old_pos, track.pos, out, ring=args.ring, new_label=f"{args.linker} linker")
            print(f"[{stem}] → {out}")

    print(f"\n{'clip':10s} {'frames':>6s}  {'current':>7s}  {'linker':>6s}   (detected / weak / filled)")
    for stem, n, oc, s in rows:
        print(f"{stem:10s} {n:6d}  {oc:7.1%}  {s['coverage']:6.1%}   ({s['detected']} / {s['weak']} / {s['filled']})")


if __name__ == "__main__":
    main()
