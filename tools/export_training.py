"""Turn checked labels (data/verified, from tools.review) into training sets for YOLO and TrackNet.

    python -m tools.export_training --name v2
    python -m tools.export_training --name v2 --tracknet-base training/tracknet/datasets/tracknet_v1

Writes, in the same formats as the existing data:

  training/datasets/yolo_<name>/          Roboflow/YOLO layout, like volleyball_combined
      train|valid/images/<clip>_<frame>.jpg   full-resolution frames
      train|valid/labels/<clip>_<frame>.txt   "0 cx cy w h" (normalized); empty file = no ball
      data.yaml                               this set alone
      data_with_combined.yaml                 this set + volleyball_combined (old + new, for fine-tuning)
  training/tracknet/datasets/tracknet_<name>/   TrackNetV3 layout (train|val/match<k>/...), every frame,
      optionally starting from an existing set (--tracknet-base) so old + new train together
  manifest.json in each: which clips, how many frames, from which verified files

What goes where:
  - only clips with role "train" (review --test / "t" marks a clip as a test clip) and no
    data/gt labels; test clips never enter training
  - YOLO: every --yolo-every'th visible frame, plus every frame you fixed by hand (the hard
    ones), plus every --neg-every'th "no ball" frame as an empty label; "hidden" and "skip"
    frames are left out (ambiguous for a single image). Frames already in
    volleyball_combined are skipped.
  - TrackNet: every checked frame, in runs of >= 8 consecutive frames; "none" and "hidden"
    are Visibility 0 (no ball visible), "skip" breaks a run
  - validation split by clip: --valid clips, else every 5th clip
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


def export_yolo(clips, out: Path, every: int, neg_every: int, skip_existing: set, log=print) -> dict:
    stats = {"images": 0, "ball": 0, "no_ball": 0, "hand_fixed": 0, "skipped_existing": 0}
    for c in clips:
        V, video, split = c["V"], c["video"], c["split"]
        W, H = V["width"], V["height"]
        labels = {int(k): x for k, x in V["labels"].items()}
        boxes = {int(k): x for k, x in V.get("boxes", {}).items()}
        src = {int(k): x for k, x in V.get("src", {}).items()}
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
                name = f"{c['stem']}_{f}"
                cv2.imwrite(str(idir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                with open(ldir / f"{name}.txt", "w") as fh:
                    if pick[f] == "ball":
                        x, y = labels[f]
                        w, h = boxes.get(f) or [V.get("ball_px", 30)] * 2
                        fh.write(f"0 {x / W:.6f} {y / H:.6f} {w / W:.6f} {h / H:.6f}\n")
                        stats["ball"] += 1
                        stats["hand_fixed"] += src.get(f) in ("fixed", "interp")
                    else:
                        stats["no_ball"] += 1
                stats["images"] += 1
            f += 1
        cap.release()
        log(f"  yolo   {c['stem']:10s} -> {split}: {len(pick)} images")
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
    ap.add_argument("--video-dir", action="append", default=["tests"], help="where to find videos (repeatable)")
    ap.add_argument("--valid", nargs="*", default=None, help="clips for the validation split (default: every 5th)")
    ap.add_argument("--yolo-every", type=int, default=4, help="every Nth visible frame (consecutive frames are near-duplicates)")
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
    clips, skipped = [], []
    for vp in sorted(Path(a.verified).glob("*.json")):
        V = json.load(open(vp))
        stem = vp.stem
        if V.get("role") == "test" or stem in gt:
            skipped.append(f"{stem} (test clip)")
            continue
        if not V.get("labels"):
            skipped.append(f"{stem} (nothing checked yet)")
            continue
        video = find_video(V, stem, a.video_dir)
        if not video:
            skipped.append(f"{stem} (video not found: --video-dir)")
            continue
        clips.append({"stem": stem, "V": V, "video": video, "file": str(vp)})
    if not clips:
        sys.exit("no checked training clips in " + a.verified + (f"  (skipped: {', '.join(skipped)})" if skipped else ""))
    for i, c in enumerate(clips):
        c["split"] = "valid" if (c["stem"] in a.valid if a.valid is not None else i % 5 == 4) else "train"
    print(f"{len(clips)} clips ({sum(c['split'] == 'valid' for c in clips)} for validation)"
          + (f"; skipped: {', '.join(skipped)}" if skipped else ""))
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M"), "name": a.name,
                "clips": [{"clip": c["stem"], "split": c["split"], "verified": c["file"],
                           "checked_frames": len(c["V"]["labels"])} for c in clips]}

    if a.only in (None, "yolo"):
        out = paths.YOLO_DATASETS / f"yolo_{a.name}"
        if out.exists():
            sys.exit(f"{out} exists: pick a new --name (datasets are versioned, never overwritten)")
        comb = Path(a.combined)
        have = existing_yolo_frames(comb) if comb.is_dir() else set()
        st = export_yolo(clips, out, a.yolo_every, a.neg_every, have)
        write_yaml(out / "data.yaml", "../train/images", "../valid/images")
        if comb.is_dir():
            # list entries are resolved relative to this yaml's folder (no Roboflow "../" quirk)
            rel = lambda p: str(Path("..") / p.relative_to(paths.YOLO_DATASETS))
            write_yaml(out / "data_with_combined.yaml",
                       ["train/images", rel(comb / "train" / "images")],
                       ["valid/images", rel(comb / "valid" / "images")])
        json.dump({**manifest, "stats": st}, open(out / "manifest.json", "w"), indent=1)
        print(f"YOLO     -> {out}: {st['images']} images ({st['ball']} ball, {st['no_ball']} no-ball, "
              f"{st['hand_fixed']} hand-fixed; {st['skipped_existing']} already in {comb.name})")
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
