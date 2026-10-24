"""Build a TrackNetV3 training set from your own clips + the YOLO dataset's labels.

    python -m tools.build_tracknet_data --labels output/dataset_labels.json --out output/tracknet_data

Labels come from training/datasets/volleyball_combined (Roboflow export, files named
testN_<frame>_jpg.rf.*): image N of a clip is decoded frame N of tests/testN.* and its
normalized YOLO box/polygon gives the ball center. Frames are taken from the original
video (no Roboflow resizing/augmentation), resized to TrackNet's 512x288 and saved as JPEG.

Layout (what TrackNetV3's dataset.py expects, IMG_FORMAT='jpg'):
    <out>/{train,val}/match<k>/median.npz              background of the whole clip (RGB)
    <out>/{train,val}/match<k>/frame/<rally>/<i>.jpg
    <out>/{train,val}/match<k>/csv/<rally>_ball.csv    Frame,Visibility,X,Y  (512x288 px)
A rally is a run of labeled frames with gaps of at most --max-gap; small gaps between two
visible frames are filled with interpolated positions. Clips with GT labels (data/gt) and
--holdout clips are never used.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 512, 288


def pick_box(boxes, ref):
    if not boxes:
        return None
    if len(boxes) == 1 or ref is None:
        return max(boxes, key=lambda b: b[2] * b[3])
    return min(boxes, key=lambda b: (b[0] - ref[0]) ** 2 + (b[1] - ref[1]) ** 2)


def rallies(lab, max_gap, min_len):
    """lab: {frame: [boxes]} -> list of [(frame, (cx, cy) normalized | None)] runs."""
    frames = sorted(lab)
    runs, cur, ref = [], [], None
    for f in frames:
        b = pick_box(lab[f], ref)
        c = None if b is None else (b[0], b[1])
        if cur and f - cur[-1][0] > 1:
            pf, pc = cur[-1]
            if f - pf <= max_gap and pc is not None and c is not None:
                for g in range(pf + 1, f):                  # interpolate a short gap
                    t = (g - pf) / (f - pf)
                    cur.append((g, (pc[0] + t * (c[0] - pc[0]), pc[1] + t * (c[1] - pc[1]))))
            else:
                runs.append(cur); cur = []
        cur.append((f, c))
        ref = c if c is not None else ref
    if cur:
        runs.append(cur)
    return [r for r in runs if len(r) >= min_len]


def build_clip(video, runs, out_match):
    need = {f for r in runs for f, _ in r}
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = set(np.linspace(0, max(n - 1, 0), min(n, 300)).astype(int).tolist())
    imgs, med_frames, f = {}, [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f in need or f in sample:
            small = np.asarray(Image.fromarray(img[:, :, ::-1]).resize((W, H)))   # RGB, bicubic
            if f in need:
                imgs[f] = small
            if f in sample:
                med_frames.append(small)
        f += 1
    cap.release()
    os.makedirs(out_match, exist_ok=True)
    np.savez(f"{out_match}/median.npz", median=np.median(np.stack(med_frames), 0))
    kept = 0
    for ri, run in enumerate(runs):
        rid = f"r{ri:03d}"
        rdir = f"{out_match}/frame/{rid}"
        os.makedirs(rdir, exist_ok=True)
        os.makedirs(f"{out_match}/csv", exist_ok=True)
        rows = ["Frame,Visibility,X,Y"]
        for i, (fr, c) in enumerate(run):
            if fr not in imgs:
                break
            Image.fromarray(imgs[fr]).save(f"{rdir}/{i}.jpg", quality=95)
            if c is None:
                rows.append(f"{i},0,0,0")
            else:
                rows.append(f"{i},1,{c[0] * W:.2f},{c[1] * H:.2f}")
            kept += 1
        open(f"{out_match}/csv/{rid}_ball.csv", "w").write("\n".join(rows) + "\n")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="output/dataset_labels.json")
    ap.add_argument("--video-dir", default="/mnt/user-data/uploads/Volley-Vision/tests")
    ap.add_argument("--out", default="output/tracknet_data")
    ap.add_argument("--val", nargs="*", default=["test9", "test19", "test26"])
    ap.add_argument("--holdout", nargs="*", default=["test7", "test12", "test17"])
    ap.add_argument("--max-gap", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=8)
    a = ap.parse_args()
    labels = json.load(open(a.labels))
    gt = {Path(g).name.replace("_gt.json", "") for g in glob.glob("data/gt/*_gt.json")}
    skip = gt | set(a.holdout)
    summary = []
    k = {"train": 0, "val": 0}
    for v in sorted(labels, key=lambda s: int(s[4:])):
        if v in skip:
            print(f"{v}: held out")
            continue
        vids = glob.glob(f"{a.video_dir}/{v}.*")
        if not vids:
            print(f"{v}: no video, skipped")
            continue
        lab = {int(f): b for f, b in labels[v].items()}
        runs = rallies(lab, a.max_gap, a.min_len)
        split = "val" if v in a.val else "train"
        k[split] += 1
        out_match = f"{a.out}/{split}/match{k[split]}"
        kept = build_clip(vids[0], runs, out_match)
        vis = sum(c is not None for r in runs for _, c in r)
        print(f"{v} -> {split}/match{k[split]}: {len(runs)} rallies, {kept} frames ({vis} with ball)", flush=True)
        summary.append({"clip": v, "split": split, "match": f"match{k[split]}", "rallies": len(runs), "frames": kept})
    json.dump({"clips": summary, "held_out": sorted(skip)}, open(f"{a.out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
