"""Look at raw YOLO candidates (output of tools.dump_detections) before any tracking.

    python -m tools.analyze_detections output/test42/cache/dets.json \
        --track output/test42/track.json --video tests/test42.MP4

Reports:
  - candidates per frame, confidence and box-size distributions
  - static clutter: spots that fire in many frames without moving (lights, logos)
  - for frames where the current tracker had no ball: what YOLO actually saw there
  - with --video: a strip of every blank-frame candidate (crop + conf) to eyeball
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(path):
    d = json.load(open(path))
    return d, [np.array(f, dtype=float).reshape(-1, 5) for f in d["frames"]]


def static_clusters(frames, cell=24, min_frac=0.15):
    """Grid cells hit in >= min_frac of frames: almost certainly not a moving ball."""
    hits = Counter()
    for f in frames:
        for cx, cy, *_ in f:
            hits[(int(cx // cell), int(cy // cell))] += 1
    n = len(frames)
    return {k: v for k, v in hits.items() if v >= min_frac * n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dets")
    ap.add_argument("--track", help="current tracker's <stem>_track.json, to find blank frames")
    ap.add_argument("--video")
    ap.add_argument("--out", default="output/dev")
    args = ap.parse_args()

    d, frames = load(args.dets)
    n = len(frames)
    allc = np.concatenate([f for f in frames if len(f)]) if any(len(f) for f in frames) else np.zeros((0, 5))
    print(f"{Path(args.dets).name}: {n} frames, {len(allc)} candidates, imgsz={d['imgsz']} conf>={d['conf']}")

    per = np.array([len(f) for f in frames])
    print(f"  candidates/frame: 0:{(per==0).sum()}  1:{(per==1).sum()}  2:{(per==2).sum()}  3+:{(per>=3).sum()}")
    conf, size = allc[:, 4], np.maximum(allc[:, 2], allc[:, 3])
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        print(f"  conf {lo:.1f}-{hi:.1f}: {m.sum():5d}   median size {np.median(size[m]) if m.any() else 0:5.1f}px")
    print(f"  box size px: p5={np.percentile(size,5):.0f} p50={np.percentile(size,50):.0f} p95={np.percentile(size,95):.0f}"
          f"   <13px: {(size<13).sum()}   >85px: {(size>85).sum()}")
    best = np.array([f[:, 4].max() if len(f) else 0 for f in frames])
    for th in (0.4, 0.25, 0.15, 0.05):
        print(f"  frames with any candidate >= {th}: {(best >= th).sum()}/{n}")

    sc = static_clusters(frames)
    print(f"  static clutter cells (24px, >=15% of frames): {len(sc)}  "
          + ", ".join(f"({x*24},{y*24})x{v}" for (x, y), v in sorted(sc.items(), key=lambda kv: -kv[1])[:8]))

    if not args.track:
        return
    tr = {r["f"]: r for r in json.load(open(args.track))["frames"]}
    blank = [i for i in range(n) if "det" not in tr.get(i, {})]
    print(f"\n  current tracker: no detection used in {len(blank)} frames")
    rows = []
    for i in blank:
        for cx, cy, w, h, c in frames[i]:
            rows.append((i, cx, cy, w, h, c))
    have = sorted({r[0] for r in rows})
    print(f"  ...of those, YOLO had >=1 candidate in {len(have)} frames ({len(rows)} candidates)")
    for i in blank:
        f = frames[i]
        if len(f):
            s = "  ".join(f"({cx:.0f},{cy:.0f}) {max(w,h):.0f}px c={c:.2f}" for cx, cy, w, h, c in f[np.argsort(-f[:, 4])][:4])
            print(f"    f{i:4d}: {s}")

    if args.video and rows:
        import cv2
        want = {}
        for r in rows:
            want.setdefault(r[0], []).append(r)
        cap = cv2.VideoCapture(args.video)
        tiles, i = [], 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            for (_, cx, cy, w, h, c) in want.get(i, []):
                H, W = img.shape[:2]
                s = 100
                x0, y0 = int(np.clip(cx - s, 0, W - 2 * s)), int(np.clip(cy - s, 0, H - 2 * s))
                t = img[y0:y0 + 2 * s, x0:x0 + 2 * s].copy()
                cv2.rectangle(t, (int(cx - w / 2 - x0), int(cy - h / 2 - y0)),
                              (int(cx + w / 2 - x0), int(cy + h / 2 - y0)), (0, 255, 0), 1)
                cv2.putText(t, f"f{i} {c:.2f}", (3, 16), 0, 0.5, (0, 255, 255), 1)
                tiles.append(t)
            i += 1
        cols = 10
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        Path(args.out).mkdir(parents=True, exist_ok=True)
        out = Path(args.out) / f"{Path(args.dets).name.split('_dets')[0]}_blank_candidates.jpg"
        cv2.imwrite(str(out), grid[:4000], [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  strip → {out}")


if __name__ == "__main__":
    main()
