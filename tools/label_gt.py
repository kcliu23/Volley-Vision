"""Label ground-truth ball positions, every Nth frame, by clicking.

    python -m tools.label_gt videos/test34.mov
    python -m tools.label_gt videos/test34.mov --every 5
    python -m tools.label_gt --suggested            # the 10 clips picked for variety, one after another

Controls (window must be focused):
    left click      the ball's centre is here            -> next frame
    n               ball not visible (hidden/off-screen)  -> next frame
    s               skip: can't tell                      -> next frame
    b / backspace   back one frame (re-label it)
    q / esc         save and quit (resume later: labelled frames are skipped)

A magnifier (3x) follows the mouse in the top-left corner, so small or far balls
can be clicked precisely. The previous label is shown as a small grey dot.
Labels save after every action to data/gt/<clip>_gt.json (committed to git):
    {"video", "every", "width", "height", "labels": {"<frame>": [x, y] | "none" | "skip"}}
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SUGGESTED = ["test10.mov", "test12.mov", "test17.MP4", "test21.MP4", "test29.MP4",
             "test34.mov", "test36.mov", "test42.MP4", "test7.mov", "test30.MP4"]
WIN = "VolleyVision labels"
MAX_W, MAX_H = 1500, 900


def load_frames(video, every):
    """Decode sequentially (seeking is unreliable on phone video); keep every Nth frame as JPEG bytes."""
    cap = cv2.VideoCapture(video)
    out, f = {}, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f % every == 0:
            out[f] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1]
        f += 1
    cap.release()
    return out


def label_clip(video, every, out_dir):
    stem = Path(video).stem
    out_path = Path(out_dir) / f"{stem}_gt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.load(open(out_path)) if out_path.exists() else {"video": video, "every": every, "labels": {}}
    every = data.get("every", every)

    print(f"[{stem}] decoding every {every}th frame...")
    frames = load_frames(video, every)
    order = sorted(frames)
    H, W = cv2.imdecode(frames[order[0]], cv2.IMREAD_COLOR).shape[:2]
    data["width"], data["height"] = W, H
    scale = min(MAX_W / W, MAX_H / H, 1.0)
    labels = data["labels"]

    def save():
        json.dump(data, open(out_path, "w"), indent=0)

    mouse = {"x": -1, "y": -1, "click": None}

    def on_mouse(ev, x, y, flags, param):
        mouse["x"], mouse["y"] = x, y
        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x / scale, y / scale)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    i = next((k for k, f in enumerate(order) if str(f) not in labels), len(order))
    cache_i, full, disp = None, None, None
    while i < len(order):
        f = order[i]
        if cache_i != i:
            full = cv2.imdecode(frames[f], cv2.IMREAD_COLOR)
            disp = cv2.resize(full, None, fx=scale, fy=scale) if scale < 1 else full.copy()
            cache_i = i
        view = disp.copy()
        # previous label as a hint of where to look
        if i > 0:
            pl = labels.get(str(order[i - 1]))
            if isinstance(pl, list):
                cv2.circle(view, (int(pl[0] * scale), int(pl[1] * scale)), 4, (160, 160, 160), -1)
        cur = labels.get(str(f))
        if isinstance(cur, list):
            cv2.circle(view, (int(cur[0] * scale), int(cur[1] * scale)), 14, (0, 255, 0), 2)
        # magnifier
        mx, my = mouse["x"], mouse["y"]
        if mx >= 0:
            fx, fy = int(mx / scale), int(my / scale)
            r = 60
            x0, y0 = int(np.clip(fx - r, 0, W - 2 * r)), int(np.clip(fy - r, 0, H - 2 * r))
            crop = cv2.resize(full[y0:y0 + 2 * r, x0:x0 + 2 * r], (6 * r, 6 * r), interpolation=cv2.INTER_NEAREST)
            c = (3 * (fx - x0), 3 * (fy - y0))
            cv2.line(crop, (c[0] - 12, c[1]), (c[0] + 12, c[1]), (0, 0, 255), 1)
            cv2.line(crop, (c[0], c[1] - 12), (c[0], c[1] + 12), (0, 0, 255), 1)
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (255, 255, 255), 2)
            h, w = crop.shape[:2]
            if h < view.shape[0] and w < view.shape[1]:
                view[0:h, 0:w] = crop
        done = sum(1 for k in order if str(k) in labels)
        bar = f"{stem}  frame {f}  ({done}/{len(order)} labelled)   click=ball  n=not visible  s=skip  b=back  q=quit"
        cv2.rectangle(view, (0, view.shape[0] - 30), (view.shape[1], view.shape[0]), (0, 0, 0), -1)
        cv2.putText(view, bar, (8, view.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, view)

        key = cv2.waitKey(20) & 0xFF
        if mouse["click"] is not None:
            x, y = mouse["click"]
            mouse["click"] = None
            labels[str(f)] = [round(x, 1), round(y, 1)]
            save()
            i += 1
        elif key == ord("n"):
            labels[str(f)] = "none"
            save()
            i += 1
        elif key == ord("s"):
            labels[str(f)] = "skip"
            save()
            i += 1
        elif key in (ord("b"), 8, 127):
            i = max(0, i - 1)
        elif key in (ord("q"), 27):
            save()
            cv2.destroyWindow(WIN)
            return False
    save()
    cv2.destroyWindow(WIN)
    print(f"[{stem}] done: {sum(isinstance(v, list) for v in labels.values())} ball positions, "
          f"{sum(v == 'none' for v in labels.values())} not visible  ->  {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default="data/gt")
    ap.add_argument("--suggested", action="store_true", help="label the 10 suggested clips in turn")
    args = ap.parse_args()
    vids = args.videos or ([f"videos/{v}" for v in SUGGESTED] if args.suggested else [])
    if not vids:
        ap.error("give video paths, or --suggested")
    for v in vids:
        if not Path(v).exists():
            print(f"missing: {v}", file=sys.stderr)
            continue
        if not label_clip(v, args.every, args.out):
            break


if __name__ == "__main__":
    main()
