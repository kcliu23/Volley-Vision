"""Turn checked labels (data/verified, from tools.review) into training sets for YOLO and TrackNet.

    python -m tools.export_training --name v2
    python -m tools.export_training --name v2 --tracknet-base training/tracknet/datasets/tracknet_v1

Writes, in the same formats as the existing data:

  training/datasets/yolo_<name>/          same layout AND processing as volleyball_combined:
      train|valid|test/images/<clip>_<frame>_jpg.rf.vv<k>.jpg   resized to fit within 1280
      train|valid|test/labels/...txt          "0 cx cy w h" (normalized); empty file = no ball
      train: 3 augmented versions per frame (crop 0-15%, rotate +-6 deg, brightness +-25%),
      exactly what Roboflow did to the old data; valid and test: 1 plain copy
      every volleyball in the picture is labelled, like the old data: the game ball (your
      checked label) + the other balls you confirmed in the review tool (shift+click)
      data.yaml                               this set alone
      data_with_combined.yaml                 this set + volleyball_combined (old + new, for fine-tuning)
  training/tracknet/datasets/tracknet_<name>/   TrackNetV3 layout (train|val/match<k>/...), every frame,
      optionally starting from an existing set (--tracknet-base) so old + new train together
  manifest.json in each: which clips, how many frames, from which verified files

What goes where:
  - splits are by WHOLE CLIP, never frame by frame (neighbouring frames are near-duplicates,
    so a frame split leaks): train = clips in videos/; valid = --valid clips, else every 5th
    training clip; test = your holdout/ clips (checked, role "test"), never trained on
  - data/gt clips (old dev set) are left out of every split
  - YOLO: every --yolo-every'th visible frame, plus every frame you fixed by hand (the hard
    ones), plus every --neg-every'th "no ball" frame as an empty label; "hidden" and "skip"
    frames are left out (ambiguous for a single image). Frames already in
    volleyball_combined are skipped.
  - TrackNet: every checked frame, in runs of >= 8 consecutive frames; "none" and "hidden"
    are Visibility 0 (no ball visible), "skip" breaks a run
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def find_video(v: dict, stem: str, video_dirs: list[str]) -> str | None:
    if v.get("video") and Path(v["video"]).exists():
        return v["video"]
    for d in video_dirs:
        for p in glob.glob(str(Path(d).expanduser() / f"{stem}.*")):
            if p.lower().endswith(VIDEO_EXT):
                return p
    return None


def existing_yolo_frames(dataset: Path) -> set[tuple[str, int]]:
    """(clip, frame) pairs already in a Roboflow-style dataset (names <clip>_<frame>_jpg.rf.*.jpg or <clip>_<frame>.jpg)."""
    have = set()
    for p in dataset.glob("*/images/*"):
        m = re.match(r"(.+?)_(\d+)(?:_jpg\.rf\..*|\.jpg|\.png)$", p.name)
        if m:
            have.add((m.group(1), int(m.group(2))))
    return have


def write_yaml(path: Path, train, val):
    fmt = lambda x: x if isinstance(x, str) else "[" + ", ".join(x) + "]"
    path.write_text(f"train: {fmt(train)}\nval: {fmt(val)}\n\nnc: 1\nnames: ['volleyball']\n")


# ── same processing as the Roboflow versions behind volleyball_combined (README.roboflow.txt) ──
#   pre-processing: resize to fit within 1280x1280
#   augmentation (train only): 3 versions of each image, each with a random crop of 0-15%,
#   rotation of -6..+6 degrees and brightness -25..+25%; valid / test: 1 plain copy
FIT = 1280
AUG_VERSIONS = 3
AUG_CROP = 0.15
AUG_ROT_DEG = 6.0
AUG_BRIGHT = 0.25


def fit_within(img, size: int = FIT):
    h, w = img.shape[:2]
    s = min(size / w, size / h, 1.0)
    # rounded down, like Roboflow (2622x1206 -> 1280x588, 2532x1170 -> 1280x591)
    return img if s == 1.0 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def augment(img, balls, rng):
    """One Roboflow-style version: crop 0-15% (then back to the same size), rotate +-6 deg, brightness +-25%.
    balls: [(cx, cy, w, h), ...] in pixels. Returns (image, the balls still (mostly) in the picture).
    The box keeps its size (a ball is round; rotating it doesn't make it bigger), scaled by the crop."""
    import numpy as np
    h, w = img.shape[:2]
    p = rng.uniform(0, AUG_CROP)
    cw, ch = w * (1 - p), h * (1 - p)
    x0, y0 = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
    sx, sy = w / cw, h / ch
    ang = rng.uniform(-AUG_ROT_DEG, AUG_ROT_DEG)
    # crop+resize, then rotate about the centre: one affine warp
    A = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]])
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), [0, 0, 1]])
    M = (R @ A)[:2]
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    b = 1 + rng.uniform(-AUG_BRIGHT, AUG_BRIGHT)
    out = np.clip(out.astype(np.float32) * b, 0, 255).astype(np.uint8)
    kept = []
    for cx, cy, bw, bh in balls:
        nx, ny = M @ [cx, cy, 1]
        nw, nh = bw * sx, bh * sy
        # clip the box to the picture; keep it while at least half the ball is still visible
        x1, y1, x2, y2 = max(nx - nw / 2, 0), max(ny - nh / 2, 0), min(nx + nw / 2, w), min(ny + nh / 2, h)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 0.5 * nw * nh:
            continue                             # (mostly) cropped or rotated out of the picture
        kept.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out, kept


def write_label(path: Path, balls, w: int, h: int):
    with open(path, "w") as fh:                  # empty file = no ball (same as Roboflow)
        for cx, cy, bw, bh in balls:
            fh.write(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")


def export_yolo(clips, out: Path, every: int, neg_every: int, skip_existing: set, log=print,
                versions: int = AUG_VERSIONS, seed: int = 0) -> dict:
    """Images named like Roboflow's (<clip>_<frame>_jpg.rf.<tag>.jpg) so every tool that reads
    volleyball_combined reads these the same way."""
    import random
    from core.other_balls import auto as auto_others
    stats = {"images": 0, "frames": 0, "ball": 0, "no_ball": 0, "hand_fixed": 0, "skipped_existing": 0,
             "ball_left_picture": 0, "other_balls": 0, "other_balls_checked": 0}
    for c in clips:
        V, video, split = c["V"], c["video"], c["split"]
        W, H = V["width"], V["height"]
        labels = {int(k): x for k, x in V["labels"].items()}
        boxes = {int(k): x for k, x in V.get("boxes", {}).items()}
        src = {int(k): x for k, x in V.get("src", {}).items()}
        others_edit = {int(k): x for k, x in V.get("others", {}).items()}
        dp = paths.dets(c["stem"])
        det_frames = json.load(open(dp))["frames"] if dp.exists() else None
        if det_frames is None:
            log(f"  yolo   {c['stem']:10s}: no YOLO cache ({dp}), can't count unconfirmed suggestions")
        sizes = sorted(max(b) for b in boxes.values()) or [V.get("ball_px", 30)]
        ball_px = sizes[len(sizes) // 2]

        def others(f):
            # only other balls you confirmed in the review tool: YOLO's own suggestions are often
            # wall clocks or lights, and labelling those as balls would teach YOLO the wrong thing
            return others_edit.get(f, [])

        def unconfirmed(f):
            if det_frames is None or f >= len(det_frames):
                return 0
            lab = labels.get(f)
            sug = auto_others(det_frames[f], lab if isinstance(lab, list) else None, ball_px)
            return sum(1 for o in sug if not any(abs(o[0] - m[0]) + abs(o[1] - m[1]) <= ball_px for m in others(f)))
        pick = {}
        vis = [f for f in sorted(labels) if isinstance(labels[f], list)]
        none = [f for f in sorted(labels) if labels[f] == "none"]
        for i, f in enumerate(vis):
            if i % every == 0 or src.get(f) in ("fixed", "interp"):
                pick[f] = "ball"
        for i, f in enumerate(none):
            if i % neg_every == 0:
                pick[f] = "none"
        pick = {f: k for f, k in pick.items() if (c["stem"], f) not in skip_existing}
        stats["skipped_existing"] += sum(1 for f in vis + none if (c["stem"], f) in skip_existing)
        if not pick:
            continue
        idir, ldir = out / split / "images", out / split / "labels"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video)
        f, last = 0, max(pick)
        while f <= last:
            ok, img = cap.read()
            if not ok:
                break
            if f in pick:
                small = fit_within(img)
                s = small.shape[1] / img.shape[1]
                h, w = small.shape[:2]
                balls = []
                if pick[f] == "ball":
                    x, y = labels[f]
                    bw, bh = boxes.get(f) or [ball_px] * 2
                    balls.append((x * s, y * s, bw * s, bh * s))
                    stats["ball"] += 1
                    stats["hand_fixed"] += src.get(f) in ("fixed", "interp")
                else:
                    stats["no_ball"] += 1
                oth = others(f)                  # every other volleyball too, like the old labels
                balls += [(o[0] * s, o[1] * s, o[2] * s, o[3] * s) for o in oth]
                stats["other_balls"] += len(oth)
                stats["other_balls_checked"] += unconfirmed(f)
                stats["frames"] += 1
                base = f"{c['stem']}_{f}_jpg.rf."
                if split == "train" and versions > 0:
                    for k in range(versions):
                        rng = random.Random(f"{seed}:{c['stem']}:{f}:{k}")
                        im2, b2 = augment(small, balls, rng)
                        stats["ball_left_picture"] += len(balls) - len(b2)
                        cv2.imwrite(str(idir / f"{base}vv{k}.jpg"), im2, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        write_label(ldir / f"{base}vv{k}.txt", b2, w, h)
                        stats["images"] += 1
                else:
                    cv2.imwrite(str(idir / f"{base}vv.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    write_label(ldir / f"{base}vv.txt", balls, w, h)
                    stats["images"] += 1
            f += 1
        cap.release()
        n_img = len(pick) * (versions if split == "train" and versions > 0 else 1)
        log(f"  yolo   {c['stem']:10s} -> {split}: {len(pick)} frames, {n_img} images")
    return stats


def export_tracknet(clips, out: Path, base: Path | None, log=print, size=(512, 288)) -> dict:
    from tools.build_tracknet_data import build_clip
    stats = {"frames": 0, "rallies": 0, "base_matches": 0}
    count = {"train": 0, "val": 0}
    if base:
        for split in ("train", "val"):
            for m in sorted((base / split).glob("match*"), key=lambda p: int(p.name[5:])):
                count[split] += 1
                shutil.copytree(m, out / split / f"match{count[split]}")
                stats["base_matches"] += 1
        log(f"  tracknet: copied {stats['base_matches']} clips from {base}")
    for c in clips:
        V = c["V"]
        W, H = V["width"], V["height"]
        labels = {int(k): x for k, x in V["labels"].items()}
        runs, cur = [], []
        for f in range(V["n"]):
            lab = labels.get(f)
            if lab is None or lab == "skip":
                if cur:
                    runs.append(cur); cur = []
                continue
            cur.append((f, (lab[0] / W, lab[1] / H) if isinstance(lab, list) else None))
        if cur:
            runs.append(cur)
        runs = [r for r in runs if len(r) >= 8]
        if not runs:
            continue
        split = "val" if c["split"] == "valid" else "train"
        count[split] += 1
        kept = build_clip(c["video"], runs, str(out / split / f"match{count[split]}"), size)
        stats["frames"] += kept
        stats["rallies"] += len(runs)
        log(f"  tracknet {c['stem']:10s} -> {split}/match{count[split]}: {len(runs)} runs, {kept} frames")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="dataset version, e.g. v2")
    ap.add_argument("--verified", default=str(paths.VERIFIED))
    ap.add_argument("--video-dir", action="append", default=["videos"], help="where to find videos (repeatable)")
    ap.add_argument("--valid", nargs="*", default=None, help="clips for the validation split (default: every 5th)")
    ap.add_argument("--yolo-every", type=int, default=2,
                    help="every Nth visible frame (default 2: about the density of the old Roboflow labels)")
    ap.add_argument("--versions", type=int, default=3, help="augmented versions per training frame (Roboflow used 3)")
    ap.add_argument("--neg-every", type=int, default=10, help="every Nth no-ball frame, as an empty label")
    ap.add_argument("--combined", default=str(paths.YOLO_DATASETS / "volleyball_combined"),
                    help="existing YOLO set: its frames are skipped, and data_with_combined.yaml includes it")
    ap.add_argument("--tracknet-base", default=None, help="existing TrackNet set to start from (old + new)")
    ap.add_argument("--only", choices=["yolo", "tracknet"], default=None)
    ap.add_argument("--archive", action="store_true",
                    help="also write yolo_<name>.zip and tracknet_<name>.tar next to the sets, ready to upload to Drive")
    ap.add_argument("--tracknet-size", default="512x288",
                    help="stored frame size WxH for the TrackNet set (e.g. 640x288 = your clips' shape, less squeezed); "
                         "frames from --tracknet-base keep their size and are resized when loaded")
    a = ap.parse_args()

    gt = set(paths.gt_stems())
    hold = set(paths.holdout_stems())
    clips, tests, skipped = [], [], []
    for vp in sorted(Path(a.verified).glob("*.json")):
        V = json.load(open(vp))
        stem = vp.stem
        if stem in gt:
            skipped.append(f"{stem} (old dev clip, data/gt)")
            continue
        is_test = V.get("role") == "test" or stem in hold
        if not V.get("labels"):
            skipped.append(f"{stem} (nothing checked yet)")
            continue
        video = find_video(V, stem, a.video_dir + [str(paths.HOLDOUT)])
        if not video:
            skipped.append(f"{stem} (video not found: --video-dir)")
            continue
        (tests if is_test else clips).append({"stem": stem, "V": V, "video": video, "file": str(vp), "split": "test"})
    if not clips:
        sys.exit("no checked training clips in " + a.verified + (f"  (skipped: {', '.join(skipped)})" if skipped else ""))
    for i, c in enumerate(clips):
        c["split"] = "valid" if (c["stem"] in a.valid if a.valid is not None else i % 5 == 4) else "train"
    print(f"{len(clips)} training clips ({sum(c['split'] == 'valid' for c in clips)} for validation), "
          f"{len(tests)} holdout clips for test"
          + (f"; skipped: {', '.join(skipped)}" if skipped else ""))
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M"), "name": a.name,
                "clips": [{"clip": c["stem"], "split": c["split"], "verified": c["file"],
                           "checked_frames": len(c["V"]["labels"])} for c in clips + tests]}

    if a.only in (None, "yolo"):
        out = paths.YOLO_DATASETS / f"yolo_{a.name}"
        if out.exists():
            sys.exit(f"{out} exists: pick a new --name (datasets are versioned, never overwritten)")
        comb = Path(a.combined)
        have = existing_yolo_frames(comb) if comb.is_dir() else set()
        st = export_yolo(clips + tests, out, a.yolo_every, a.neg_every, have, versions=a.versions)
        write_yaml(out / "data.yaml", "../train/images", "../valid/images")
        if tests:
            with open(out / "data.yaml", "a") as fh:
                fh.write("test: ../test/images\n")
        if comb.is_dir():
            # list entries are resolved relative to this yaml's folder (no Roboflow "../" quirk)
            rel = lambda p: str(Path("..") / p.relative_to(paths.YOLO_DATASETS))
            write_yaml(out / "data_with_combined.yaml",
                       ["train/images", rel(comb / "train" / "images")],
                       ["valid/images", rel(comb / "valid" / "images")])
        json.dump({**manifest, "stats": st}, open(out / "manifest.json", "w"), indent=1)
        print(f"YOLO     -> {out}: {st['frames']} frames -> {st['images']} images ({st['ball']} ball, "
              f"{st['no_ball']} no-ball, {st['hand_fixed']} hand-fixed; {st['skipped_existing']} already in "
              f"{comb.name}; ball cropped out in {st['ball_left_picture']} augmented images)\n"
              f"         + {st['other_balls']} other balls you confirmed; {st['other_balls_checked']} YOLO suggestions "
              f"left unlabelled (not confirmed in the review tool)")
        if a.archive:
            z = shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
            print(f"         -> {z}")

    if a.only in (None, "tracknet"):
        out = paths.TRACKNET_DATASETS / f"tracknet_{a.name}"
        if out.exists():
            sys.exit(f"{out} exists: pick a new --name (datasets are versioned, never overwritten)")
        tw, th = (int(v) for v in a.tracknet_size.lower().split("x"))
        st = export_tracknet(clips, out, Path(a.tracknet_base) if a.tracknet_base else None, size=(tw, th))
        json.dump({**manifest, "base": a.tracknet_base, "stats": st}, open(out / "manifest.json", "w"), indent=1)
        print(f"TrackNet -> {out}: {st['frames']} new frames in {st['rallies']} runs"
              + (f", plus {st['base_matches']} clips from {a.tracknet_base}" if a.tracknet_base else ""))
        if a.archive:
            t = shutil.make_archive(str(out), "tar", root_dir=out.parent, base_dir=out.name)
            print(f"         -> {t}")


if __name__ == "__main__":
    main()
