"""Propose ball labels for every frame of new videos, for a human to check in tools.review.

    python -m tools.autolabel videos/test40.mov videos/test41.mov
    python -m tools.autolabel --dir ~/Videos/games            # every video in a folder
    python -m tools.autolabel videos/test40.mov --classic      # the old pipeline (+ --tracknet)

Runs the finder-first pipeline with the loose crop rule (fewest missed balls: a miss costs
more clicks in review than a false ball), plus full-frame YOLO on every frame (cached), which
the review tool's other-ball suggestions and the export's hard negatives need. No render.
Writes data/proposals/<clip>.json:

  frames[f]   {"state": detected|weak|tracknet|filled|none, "xy": [x, y] | null,
               "box": [w, h], "doubt": [reasons]}
  segments    the clip cut into "rally" stretches (ball tracked, with 0.5 s margins) and
              "idle" stretches (no ball for > 1 s), so review can go rally by rally

Doubt reasons flag frames worth a closer look (shown in red on the review timeline):
  filled      position is a physics guess (no detection) — the ball may be hidden
  tracknet    position from the ball finder only (WASB, or TrackNet with --classic)
  resting     a ball not in play (held, resting, caught in the net): the tracker leaves it out,
              but it's a visible ball, so it's proposed for the labels
  low_conf    weak YOLO detection (conf < 0.15)
  edge        ball at the picture edge
  jump        position off the line through its neighbours (> 1.5 ball-widths)
  disagree    the finder strongly sees the ball somewhere else (> 3 ball-widths away)
  missed      no position, but the finder strongly sees a ball
  contact     odd contact: direction barely changes (likely a split arc, e.g. a long lob)
Clips that already have ground truth in data/gt are skipped (they are the test set).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def _box_sizes(track_rows, dets, ball_px):
    """Box (w, h) per frame: the YOLO candidate at the track position if there is one,
    else the median observed size within ±1 s, else the clip's ball size."""
    n = len(track_rows)
    fps = dets.get("fps") or 60
    frames = dets["frames"]
    zoom = dets.get("zoom") or {}
    obs = {}
    for r in track_rows:
        f, p = r["f"], r.get("pos")
        if p is None or r["state"] not in ("detected", "weak"):
            continue
        cands = list(frames[f]) + list((zoom.get(str(f)) or {}).get("cands", []))
        best = min(cands, key=lambda c: math.hypot(c[0] - p[0], c[1] - p[1]), default=None)
        if best is not None and math.hypot(best[0] - p[0], best[1] - p[1]) <= 0.5 * ball_px:
            obs[f] = (float(best[2]), float(best[3]))
    keys = np.array(sorted(obs))
    out = []
    R = max(1, int(round(fps)))
    for f in range(n):
        if f in obs:
            out.append(obs[f])
            continue
        if len(keys):
            near = keys[np.abs(keys - f) <= R]
            if len(near) >= 3:
                out.append((float(np.median([obs[k][0] for k in near])), float(np.median([obs[k][1] for k in near]))))
                continue
        out.append((ball_px, ball_px))
    return out


def propose(video: str, out_dir: str = "data/proposals", tracknet: str | None = None,
            output_dir: str = "output", model_path: str = "models/best4.pt", classic: bool = False,
            finder: str = "models/wasb/wasb_ft2.pth.tar") -> Path:
    from core.video_pipeline import run
    from core import paths, finders
    stem = Path(video).stem
    if classic:
        run(input_path=video, output_dir=output_dir, model_path=model_path, tracknet=tracknet, render=False,
            pipeline="classic")
        dets = json.load(open(paths.dets(stem, output_dir)))
        prof = dets.get("profile") or {}
        tpath = paths.tracknet(stem, output_dir)
        from core.tracknet import correct
        tn = correct(json.load(open(tpath))["frames"], float(prof.get("ball_px") or 30.0)) \
            if (tracknet and tpath.exists()) else None
    else:
        dp = paths.dets(stem, output_dir)
        if not dp.exists():           # full-frame YOLO: other-ball suggestions + hard negatives need it
            from ultralytics import YOLO
            from tools.dump_detections import dump
            print("[autolabel] full-frame YOLO pass (for other balls / hard negatives; cached)")
            dump(video, YOLO(model_path), paths.ensure(dp), 0.05, 1280)
        run(input_path=video, output_dir=output_dir, model_path=model_path, render=False,
            pipeline="finder-first", finder=finder, crop_p=None)
        dets = json.load(open(dp))
        sample = json.load(open(paths.cache_dir(stem, output_dir) / "sample_dets.json"))
        prof = sample.get("profile") or {}
        ftag = finders.tag(finder) or "tracknet"
        crops = json.load(open(paths.cache_dir(stem, output_dir) / f"crops_{ftag}.json"))
        dets["zoom"] = crops["found"]           # box sizes at crop-confirmed positions
        tn = finders.correct(finders.load_cached(stem, finder, output_dir), float(prof.get("ball_px") or 30.0), finder)
        tracknet = finder
    track = json.load(open(paths.track(stem, output_dir)))
    cpath = paths.contacts(stem, output_dir)
    contacts = json.load(open(cpath))["contacts"] if cpath.exists() else []
    fps = prof.get("fps") or track.get("fps") or 60.0
    ball = float(prof.get("ball_px") or 30.0)
    W, H = track["width"], track["height"]
    x0, y0, x1, y1 = prof.get("content") or dets.get("content") or (0, 0, W, H)
    rows = track["frames"]
    n = len(rows)
    boxes = _box_sizes(rows, dets, ball)

    frames = []
    for r in rows:
        f = r["f"]
        p = r.get("pos")
        doubt = []
        st = r["state"]
        if r.get("resting") and st in ("none", "filled"):   # a real detection beats a guess
            # a visible ball the tracker set aside as not in play (held, resting, caught in
            # the net): the training labels mark every visible ball, so propose it
            p, st = r["resting"][:2], "resting"
            doubt.append("resting")
        if st == "filled":
            doubt.append("filled")
        elif st == "tracknet":
            doubt.append("tracknet")
        elif st == "weak" and (r.get("conf") or 0) < 0.15:
            doubt.append("low_conf")
        if p is not None:
            m = 0.6 * ball
            if p[0] <= x0 + m or p[1] <= y0 + m or p[0] >= x1 - m or p[1] >= y1 - m:
                doubt.append("edge")
        if tn is not None and f < len(tn):
            strong = [c for c in tn[f] if c[4] >= 0.6]
            if p is not None and strong and min(math.hypot(c[0] - p[0], c[1] - p[1]) for c in strong) > 3 * ball:
                doubt.append("disagree")
            if p is None and strong:
                doubt.append("missed")
        frames.append({"state": st, "xy": [round(p[0], 1), round(p[1], 1)] if p is not None else None,
                       "box": [round(boxes[f][0], 1), round(boxes[f][1], 1)], "doubt": doubt})

    # jumps: off the line through the neighbours on both sides
    for f in range(1, n - 1):
        a, b, c = frames[f - 1]["xy"], frames[f]["xy"], frames[f + 1]["xy"]
        if a and b and c:
            mx, my = (a[0] + c[0]) / 2, (a[1] + c[1]) / 2
            if math.hypot(b[0] - mx, b[1] - my) > 1.5 * ball:
                frames[f]["doubt"].append("jump")
    # contacts where the direction barely changes are usually a split arc, not a real hit
    for c in contacts:
        if c.get("angle_deg", 180) < 15:
            for f in range(max(0, int(c["frame"]) - 3), min(n, int(c["frame"]) + 4)):
                if "contact" not in frames[f]["doubt"]:
                    frames[f]["doubt"].append("contact")

    # rally / idle segments
    has = [fr["xy"] is not None for fr in frames]
    gap_join, margin = int(round(1.0 * fps)), int(round(0.5 * fps))
    runs, f = [], 0
    while f < n:
        if has[f]:
            g = f
            while g + 1 < n and (has[g + 1] or any(has[g + 1:min(n, g + 1 + gap_join)])):
                g += 1
            runs.append([max(0, f - margin), min(n - 1, g + margin)])
            f = g + 1
        else:
            f += 1
    merged = []
    for r in runs:
        if merged and r[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    segments, cur = [], 0
    for a, b in merged:
        if a > cur:
            segments.append({"f0": cur, "f1": a - 1, "kind": "idle"})
        segments.append({"f0": a, "f1": b, "kind": "rally"})
        cur = b + 1
    if cur < n:
        segments.append({"f0": cur, "f1": n - 1, "kind": "idle"})
    for s in segments:
        s["doubt"] = sum(1 for g in range(s["f0"], s["f1"] + 1) if frames[g]["doubt"])

    out = Path(out_dir) / f"{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"schema": "vv-proposals/1", "video": str(video), "fps": fps, "width": W, "height": H,
               "n": n, "ball_px": ball, "content": [x0, y0, x1, y1],
               "pipeline": {"model": model_path, "tracknet": tracknet if tn is not None else None,
                            "linker": track.get("linker")},
               "segments": segments, "frames": frames}, open(out, "w"), separators=(",", ":"))
    nd = sum(1 for fr in frames if fr["doubt"])
    ral = sum(s["f1"] - s["f0"] + 1 for s in segments if s["kind"] == "rally")
    print(f"[autolabel] {stem}: {n} frames, {sum(s['kind'] == 'rally' for s in segments)} rallies "
          f"({ral / fps:.0f}s), {nd} frames flagged ({nd / max(n, 1):.1%})  →  {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--dir", help="propose labels for every video in this folder")
    ap.add_argument("--out", default="data/proposals")
    ap.add_argument("--model", default="models/best4.pt")
    # TrackNet is off by default for labeling: its points run ~0.35 ball-widths off (right
    # direction, wrong spot), which costs more fixing than the gaps it fills. Turn it on with
    # --tracknet (bare flag = the current checkpoint) once a retrained model is more precise.
    ap.add_argument("--tracknet", nargs="?", const="models/tracknet/FT_TrackNet_best.pt", default=None,
                    help="(--classic) also use TrackNet")
    ap.add_argument("--classic", action="store_true", help="the old pipeline (YOLO every frame + rule linker + zoom)")
    ap.add_argument("--finder", default="models/wasb/wasb_ft2.pth.tar", help="ball finder (finder-first)")
    ap.add_argument("--no-tracknet", action="store_true", help="(default; kept for old commands)")
    ap.add_argument("--include-gt", action="store_true", help="also process clips that have ground truth (test set)")
    ap.add_argument("--redo", action="store_true", help="re-propose clips that already have a proposals file")
    a = ap.parse_args()
    vids = list(a.videos)
    if a.dir:
        vids += sorted(p for p in glob.glob(str(Path(a.dir).expanduser() / "*")) if p.lower().endswith(VIDEO_EXT))
    if not vids:
        ap.error("give video files or --dir")
    gt = {Path(g).name.replace("_gt.json", "") for g in glob.glob("data/gt/*_gt.json")}
    for v in vids:
        stem = Path(v).stem
        if stem in gt and not a.include_gt:
            print(f"[autolabel] {stem}: has ground truth (test set), skipped")
            continue
        if (Path(a.out) / f"{stem}.json").exists() and not a.redo:
            print(f"[autolabel] {stem}: already proposed, skipped (--redo to redo)")
            continue
        propose(v, a.out, None if a.no_tracknet else a.tracknet, model_path=a.model, classic=a.classic,
                finder=a.finder)


if __name__ == "__main__":
    main()
