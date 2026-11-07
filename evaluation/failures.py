"""Where does ball tracking fail, and why? A breakdown of every wrong frame on the holdout clips.

    python -m evaluation.failures                                  # current models
    python -m evaluation.failures --finder models/wasb/wasb_ft3.pth.tar   # a retrained WASB
    python -m evaluation.failures --clips holdout3 holdout8

Runs the pipeline (as evaluation.scoreboard does, reusing caches) on the holdout clips, then compares
every frame with your checked labels, for two things:
  WASB alone   its best guess per frame, and whether the ball was among its top 3 guesses
  pipeline     the final ball position (WASB + YOLO crops + linker + physics search)

Each failure frame is tagged with what was going on (a frame can have several tags):
  fast        the ball moves > 35 px per frame (spikes, serves): blurred, stretched
  small       the ball is < 18 px across (far court)
  edge        the ball is within 40 px of the picture's edge
  static      the wrong guess sits where wrong guesses keep landing in this clip (a light, a fan,
              a logo, a ball on the floor): the same spot, 25+ frames
  in-top-3    WASB had the right ball among its guesses, the linker picked another: a linker
              problem, not a WASB one
  hidden      no ball in your labels (hidden / out of the picture) but the pipeline shows one

Prints rates per clip and per tag, the longest failure stretches (frame ranges to look at), and
saves a contact sheet of examples per tag to output/dev/failures/<tag>.jpg.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from core.common import paths
from evaluation.scoreboard import CUR_FINDER, CUR_YOLO, find_video, run_one

FAST_PX, SMALL_PX, EDGE_PX, STATIC_N, STATIC_R = 35.0, 18.0, 40.0, 25, 12.0


def wasb_guesses(stem, finder, root):
    from core.ball import finders
    p = finders.cache_path(stem, finder, root)
    if not p.exists():
        p = finders.cache_path(stem, finder)
    if not p.exists():
        return None
    return json.load(open(p))["frames"]


def analyse(stem, track, labels, boxes, guesses, W, H, ball):
    rows = []
    gt_pts = {int(k): v for k, v in labels.items() if isinstance(v, list)}
    wrong_spots = []                                     # where the pipeline's wrong guesses land
    for f in range(len(track.pos)):
        v = labels.get(str(f))
        if v is None or v == "skip":
            continue
        p, s = track.pos[f], track.state[f]
        r = {"clip": stem, "f": f, "tags": set()}
        if isinstance(v, list):
            gx, gy = v[0], v[1]
            prev = gt_pts.get(f - 1)
            speed = math.hypot(gx - prev[0], gy - prev[1]) if prev else 0.0
            size = float(np.mean(boxes.get(str(f), [ball, ball])))
            if speed > FAST_PX:
                r["tags"].add("fast")
            if size < SMALL_PX:
                r["tags"].add("small")
            if min(gx, gy, W - gx, H - gy) < EDGE_PX:
                r["tags"].add("edge")
            tol = max(ball, 0.75 * size)
            if p is None:
                r["kind"] = "miss"
            elif math.hypot(p[0] - gx, p[1] - gy) <= tol:
                r["kind"] = "hit"
            else:
                r["kind"] = "wrong"
                wrong_spots.append((f, p))
            # WASB alone
            g = guesses[f] if guesses and f < len(guesses) else []
            if g:
                top = g[0]
                r["wasb"] = "hit" if math.hypot(top[0] - gx, top[1] - gy) <= tol else "wrong"
                r["in3"] = any(math.hypot(c[0] - gx, c[1] - gy) <= tol for c in g[:3])
            else:
                r["wasb"], r["in3"] = "miss", False
            if r["kind"] != "hit" and r["in3"]:
                r["tags"].add("in-top-3")
            r["gt"], r["pred"] = (gx, gy), p
        else:
            r["kind"] = "false" if (p is not None and s in ("detected", "weak", "tracknet")) else "ok"
            if r["kind"] == "false":
                r["tags"].add("hidden")
                wrong_spots.append((f, p))
            r["gt"], r["pred"] = None, p
        rows.append(r)
    # static spots: wrong guesses that keep landing at the same place
    if wrong_spots:
        P = np.array([p for _, p in wrong_spots], float)
        for (f, p) in wrong_spots:
            n = int(np.sum(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]) < STATIC_R))
            if n >= STATIC_N:
                for r in rows:
                    if r["f"] == f:
                        r["tags"].add("static")
                        break
    return rows


def stretches(rows):
    """Consecutive failure frames (wrong / miss / false) as (clip, f0, f1, kind, tags)."""
    out, cur = [], None
    for r in rows:
        bad = r["kind"] in ("wrong", "miss", "false")
        if bad and cur and cur[0] == r["clip"] and r["f"] == cur[2] + 1:
            cur[2] = r["f"]; cur[3][r["kind"]] += 1; cur[4].update(r["tags"])
        else:
            if cur:
                out.append(cur)
            cur = [r["clip"], r["f"], r["f"], Counter({r["kind"]: 1}), Counter(r["tags"])] if bad else None
    if cur:
        out.append(cur)
    return out


def contact_sheet(rows, tag, videos, out, n=12):
    import cv2
    pick = [r for r in rows if tag in r["tags"] and r["kind"] != "hit"]
    if not pick:
        return
    step = max(1, len(pick) // n)
    pick = pick[::step][:n]
    tiles = []
    for r in pick:
        cap = cv2.VideoCapture(videos[r["clip"]])
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["f"])
        ok, img = cap.read()
        if not ok:
            continue
        c = r["gt"] or r["pred"]
        x, y = int(c[0]), int(c[1])
        if r["gt"]:
            cv2.circle(img, (int(r["gt"][0]), int(r["gt"][1])), 22, (0, 255, 0), 3)        # green: the ball
        if r["pred"]:
            cv2.drawMarker(img, (int(r["pred"][0]), int(r["pred"][1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 4)
        h, w = img.shape[:2]
        x0, y0 = min(max(0, x - 300), w - 600), min(max(0, y - 220), h - 440)
        t = img[y0:y0 + 440, x0:x0 + 600].copy()
        cv2.putText(t, f"{r['clip']} {r['f']} {r['kind']}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out), cv2.resize(grid, None, fx=0.6, fy=0.6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--yolo", default=CUR_YOLO)
    ap.add_argument("--finder", default=CUR_FINDER)
    ap.add_argument("--video-dir", action="append", default=["holdout", "videos"])
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args()
    from tools.calibrate import ball_px
    tag = "_".join(Path(x).name.split(".")[0] for x in (a.yolo, a.finder))
    root = paths.dev() / "failures" / tag
    hold = [s for s in paths.holdout_stems() if paths.verified(s).exists()]
    clips = [c for c in hold if not a.clips or c in a.clips]
    rows, videos = [], {}
    for stem in clips:
        video = find_video(stem, a.video_dir)
        if not video:
            print(f"  {stem}: video not found, skipped")
            continue
        videos[stem] = video
        print(f"== {stem}", flush=True)
        track = run_one(stem, video, root, a.yolo, a.finder, reuse_yolo=(a.yolo == CUR_YOLO),
                        reuse_finder=(a.finder == CUR_FINDER), pipeline="finder-first")
        V = json.load(open(paths.verified(stem)))
        ball = ball_px(json.load(open(paths.cache_dir(stem, root) / "sample_dets.json")))
        rows += analyse(stem, track, V["labels"], V.get("boxes", {}), wasb_guesses(stem, a.finder, root),
                        V["width"], V["height"], ball)

    vis = [r for r in rows if r["gt"] is not None]
    hid = [r for r in rows if r["gt"] is None]
    k = Counter(r["kind"] for r in vis)
    print(f"\nmodels: YOLO {a.yolo}   WASB {a.finder}   ({len(clips)} holdout clips, {len(vis)} frames with the ball)")
    print(f"pipeline   on-ball {k['hit'] / len(vis):6.1%}   wrong {k['wrong'] / len(vis):5.1%}   missed {k['miss'] / len(vis):5.1%}"
          f"   false-when-hidden {sum(r['kind'] == 'false' for r in hid) / max(len(hid), 1):5.1%}")
    kw = Counter(r["wasb"] for r in vis)
    print(f"WASB alone on-ball {kw['hit'] / len(vis):6.1%}   ball in its top 3: {sum(r['in3'] for r in vis) / len(vis):5.1%}")

    print("\nper clip (pipeline on-ball / WASB alone on-ball)")
    for stem in clips:
        v = [r for r in vis if r["clip"] == stem]
        if v:
            print(f"  {stem:10s} {sum(r['kind'] == 'hit' for r in v) / len(v):6.1%} / {sum(r['wasb'] == 'hit' for r in v) / len(v):6.1%}")

    bad = [r for r in rows if r["kind"] in ("wrong", "miss", "false")]
    print(f"\nwhat the {len(bad)} failure frames have in common (a frame can have several tags)")
    allv = Counter(t for r in vis for t in r["tags"] if t in ("fast", "small", "edge"))
    tags = Counter(t for r in bad for t in r["tags"])
    for t, n in tags.most_common():
        base = f"   ({allv[t] / len(vis):.0%} of all ball frames)" if t in allv else ""
        print(f"  {t:9s} {n:5d} frames  {n / len(bad):5.1%} of failures{base}")
    print(f"  {'(none)':9s} {sum(not r['tags'] for r in bad):5d} frames  {sum(not r['tags'] for r in bad) / max(len(bad), 1):5.1%}")

    print("\nlongest failure stretches (look at these)")
    for clip, f0, f1, kinds, tg in sorted(stretches(rows), key=lambda s: s[1] - s[2])[:15]:
        print(f"  {clip:10s} {f0:5d}-{f1:<5d} ({f1 - f0 + 1:3d} frames)  {dict(kinds)}  {', '.join(t for t, _ in tg.most_common(3))}")

    out = paths.dev() / "failures"
    json.dump([{**r, "tags": sorted(r["tags"])} for r in rows], open(out / f"{tag}.json", "w"))
    if not a.no_sheets:
        for t in ("fast", "small", "edge", "static", "in-top-3", "hidden"):
            contact_sheet(rows, t, videos, out / f"{t}.jpg")
        print(f"\nexamples per tag (green = the ball, red x = the guess): {out}/<tag>.jpg")


if __name__ == "__main__":
    main()
