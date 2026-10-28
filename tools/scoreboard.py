"""Should a new checkpoint replace the current one? Score both through the whole pipeline.

    python -m tools.scoreboard --yolo models/best5.pt                      # finder-first (default)
    python -m tools.scoreboard --finder models/wasb/wasb_ft4.pth.tar
    python -m tools.scoreboard --pipeline classic --tracknet models/tracknet/FT2_TrackNet_best.pt

Runs the full pipeline (main.py without the video render) with the current models and with
the candidate, each in its own folder under output/dev/scoreboard/ (your real outputs are
untouched), and compares:
  - holdout clips (holdout/, checked every frame): the real test; decide on these
  - dev clips (data/gt, every 5th frame; test7/12/17 were in best4's training)
  - your checked clips (data/verified, every frame; marked train, so a model trained on them
    has seen them: read these as "did it learn them", the test clips as "does it generalize")
  - the 11 regression cases
Caches are copied from output/<clip>/cache when a model is unchanged, so only what changed is
recomputed (finder-first: a new YOLO redoes the sparse pass and the crops, a new finder its
pass and the crops; classic: a new YOLO a full detection pass, a new TrackNet a TrackNet pass).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import paths  # noqa: E402

CUR_YOLO = "models/best4.pt"
CUR_TN = "models/tracknet/FT_TrackNet_best.pt"
CUR_FINDER = "models/wasb/wasb_ft2.pth.tar"
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")


def find_video(stem: str, dirs: list[str]) -> str | None:
    for d in dirs:
        for p in glob.glob(str(Path(d).expanduser() / f"{stem}.*")):
            if p.lower().endswith(VIDEO_EXT):
                return p
    return None


def run_one(stem, video, root, yolo, finder, reuse_yolo, reuse_finder, pipeline):
    from core.video_pipeline import run
    from core import finders

    def reuse(src, dst):
        if src.exists() and not dst.exists():
            shutil.copy(src, paths.ensure(dst))
    if pipeline == "classic":
        if reuse_yolo:
            reuse(paths.dets(stem), paths.dets(stem, root))
        if reuse_finder:
            reuse(paths.tracknet(stem), paths.tracknet(stem, root))
        run(input_path=video, output_dir=str(root), model_path=yolo, tracknet=finder, render=False,
            pipeline="classic")
    else:
        cache, dst = paths.cache_dir(stem), paths.cache_dir(stem, root)
        if reuse_yolo:
            reuse(cache / "sample_dets.json", dst / "sample_dets.json")
        if reuse_finder:
            reuse(finders.cache_path(stem, finder), finders.cache_path(stem, finder, root))
        if reuse_yolo and reuse_finder:              # crops depend on both models
            for c in cache.glob(f"crops_{finders.tag(finder)}*.json"):
                reuse(c, dst / c.name)
        run(input_path=video, output_dir=str(root), model_path=yolo, finder=finder, render=False)
    rows = json.load(open(paths.track(stem, root)))["frames"]
    return SimpleNamespace(pos=[tuple(r["pos"]) if "pos" in r else None for r in rows],
                           state=[r["state"] for r in rows])


def score(track, labels: dict, ball: float) -> dict:
    r = {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0}
    for k, v in labels.items():
        f = int(k)
        if v == "skip" or f >= len(track.pos):
            continue
        p, s = track.pos[f], track.state[f]
        if isinstance(v, list):
            if p is None:
                r["miss"] += 1
            elif math.hypot(p[0] - v[0], p[1] - v[1]) <= ball:
                r["hit"] += 1
            else:
                r["wrong"] += 1
        else:
            r["false" if (p is not None and s in ("detected", "weak", "tracknet")) else "ok"] += 1
    return r


def fmt(t: dict) -> str:
    vis, hid = t["hit"] + t["wrong"] + t["miss"], t["ok"] + t["false"]
    return (f"on-ball {t['hit'] / max(vis, 1):6.1%}  wrong {t['wrong'] / max(vis, 1):5.1%}  "
            f"missed {t['miss'] / max(vis, 1):5.1%}  false-when-hidden {t['false'] / max(hid, 1):5.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--yolo", default=None, help="candidate YOLO weights (default: keep the current)")
    ap.add_argument("--finder", default=None, help="candidate ball finder, WASB (finder-first)")
    ap.add_argument("--tracknet", default=None, help="candidate TrackNet checkpoint (classic)")
    ap.add_argument("--pipeline", choices=["finder-first", "classic"], default="finder-first")
    ap.add_argument("--video-dir", action="append", default=["videos", "holdout"])
    ap.add_argument("--clips", nargs="*", help="only these clips")
    a = ap.parse_args()
    classic = a.pipeline == "classic"
    if not a.yolo and not (a.tracknet if classic else a.finder):
        ap.error("give --yolo and/or " + ("--tracknet" if classic else "--finder"))
    from tools.calibrate import ball_px
    from tools.regression import _cases
    cur_f = CUR_TN if classic else CUR_FINDER
    cand_yolo, cand_f = a.yolo or CUR_YOLO, (a.tracknet if classic else a.finder) or cur_f
    tag = a.pipeline + "_" + "_".join(Path(x).name.split(".")[0] for x in (cand_yolo, cand_f))
    roots = {"current": paths.dev() / "scoreboard" / f"{a.pipeline}_current",
             "candidate": paths.dev() / "scoreboard" / tag}
    models = {"current": (CUR_YOLO, cur_f), "candidate": (cand_yolo, cand_f)}

    gt = paths.gt_stems()
    hold = [s for s in paths.holdout_stems() if paths.verified(s).exists() and s not in gt]
    checked = [p.stem for p in sorted(paths.VERIFIED.glob("*.json")) if p.stem not in gt and p.stem not in hold]
    reg = sorted({c[0] for c in _cases()})
    clips = [c for c in dict.fromkeys(hold + gt + checked + reg) if not a.clips or c in a.clips]
    tracks = {"current": {}, "candidate": {}}
    for stem in clips:
        video = find_video(stem, a.video_dir)
        if not video:
            print(f"  {stem}: video not found (--video-dir), skipped")
            continue
        for name in ("current", "candidate"):
            y, t = models[name]
            print(f"== {stem} [{name}]", flush=True)
            tracks[name][stem] = run_one(stem, video, roots[name], y, t, reuse_yolo=(y == CUR_YOLO),
                                         reuse_finder=(t == cur_f), pipeline=a.pipeline)

    print(f"\n{a.pipeline} pipeline; candidate: YOLO {cand_yolo}   {'TrackNet' if classic else 'finder'} {cand_f}\n")
    for title, group, src in (("HOLDOUT clips (real test: never trained on, every frame)", hold, "verified"),
                              ("dev clips (data/gt; test7/12/17 were in best4's training)", gt, "gt"),
                              ("your checked training clips", checked, "verified")):
        tot = {n: {"hit": 0, "wrong": 0, "miss": 0, "ok": 0, "false": 0} for n in tracks}
        per = []
        for stem in group:
            if stem not in tracks["current"]:
                continue
            labels = json.load(open(paths.gt(stem) if src == "gt" else paths.verified(stem)))["labels"]
            dp = paths.dets(stem, roots["current"]) if classic else \
                paths.cache_dir(stem, roots["current"]) / "sample_dets.json"
            ball = ball_px(json.load(open(dp)))
            row = {}
            for n in tracks:
                r = score(tracks[n][stem], labels, ball)
                row[n] = r
                for k in r:
                    tot[n][k] += r[k]
            per.append((stem, row))
        if not per:
            continue
        print(title)
        for n in ("current", "candidate"):
            print(f"  {n:10s} {fmt(tot[n])}")
        for stem, row in per:
            h = {n: row[n]["hit"] / max(1, row[n]["hit"] + row[n]["wrong"] + row[n]["miss"]) for n in row}
            print(f"     {stem:8s} current {h['current']:6.1%}   candidate {h['candidate']:6.1%}"
                  + ("   <" if h["candidate"] < h["current"] - 0.005 else ""))
    print("regression cases")
    for n in ("current", "candidate"):
        res = [(stem, desc, *check(tracks[n][stem])) for stem, desc, check in _cases() if stem in tracks[n]]
        print(f"  {n:10s} {sum(ok for _, _, ok, _ in res)}/{len(res)} passed"
              + "".join(f"\n     FAIL {s} {d}: {det}" for s, d, ok, det in res if not ok))


if __name__ == "__main__":
    main()
