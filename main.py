"""VolleyVision: one clip, the whole pipeline.

    python -m main videos/match.mov                  # 2D ball -> people (pose) -> 3D ball -> viewer
    python -m main videos/match.mov --video          # + both videos (tracked.mp4, ball3d.mp4)
    python -m main videos/match.mov --redo           # redo the poses too (otherwise kept from last time)
    python -m main holdout/holdout1.mov --labels     # 3D from your checked labels, not the tracker's
    python -m main videos/match.mov --2d-only        # just the 2D ball track

Needs the court calibration first (python -m tools.calibrate_court <video>); without it, only the
2D track is made. Each step prints its time; results land in output/<clip>/. Options this command
doesn't know (e.g. --ring, --trail-len) go to the 2D step (python -m tools.track2d -h).
The tools/ commands are helpers: labeling, single steps, viewers.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TIMES: list[tuple[str, float]] = []


def step(*args):
    print(f"\n[volleyvision] python -m {' '.join(args)}", flush=True)
    t = time.time()
    if subprocess.run([sys.executable, "-m", *args]).returncode:
        sys.exit(f"\n[volleyvision] stopped: python -m {args[0]} failed (its error is above)")
    TIMES.append((args[0], time.time() - t))


def report(stem: str, t0: float):
    tot = max(time.time() - t0, 1e-9)
    print(f"\n[volleyvision] {stem}: time per step ({tot:.0f} s in all)")
    for name, dt in TIMES:
        print(f"[volleyvision]   {dt:7.1f} s  {100 * dt / tot:4.0f}%  {name}")


def main():
    ap = argparse.ArgumentParser(prog="volleyvision", description=__doc__.split("\n")[0],
                                 epilog="Other options go to the 2D step: python -m tools.track2d -h")
    ap.add_argument("video")
    ap.add_argument("--output", default="output")
    ap.add_argument("--video", dest="render_video", action="store_true",
                    help="also render the videos: tracked.mp4 (2D) and ball3d.mp4 (3D)")
    ap.add_argument("--redo", action="store_true", help="run pose again even if output/<clip>/pose.json exists")
    ap.add_argument("--labels", action="store_true", help="3D from your checked labels instead of the tracker's output")
    ap.add_argument("--no-players", action="store_true", help="skip pose (faster; no player check on touches)")
    ap.add_argument("--2d-only", dest="only_2d", action="store_true", help="only the 2D ball track")
    a, rest = ap.parse_known_args()
    if not Path(a.video).exists():
        sys.exit(f"Error: video file not found at {a.video}")
    from core.common import paths
    stem = Path(a.video).stem
    t0 = time.time()

    step("tools.track2d", a.video, "--output", a.output, *([] if a.render_video else ["--no-render"]), *rest)
    if a.only_2d:
        report(stem, t0)
        return
    if not (Path("data/cameras") / f"{stem}.json").exists():
        report(stem, t0)
        sys.exit(f"\n[volleyvision] {stem}: 2D track done; no court calibration yet, so no 3D. Run:\n"
                 f"  python -m tools.calibrate_court {a.video}\n  python -m main {a.video}")

    from core.common.frametime import frame_times, summary
    print(f"[volleyvision] {stem}: frame timing: {summary(frame_times(a.video, paths.clip_dir(stem, a.output)))}")
    if not a.no_players and (a.redo or not (paths.clip_dir(stem, a.output) / "pose.json").exists()):
        step("tools.pose", a.video, "--output", a.output, *(["--redo"] if a.redo else []))
    step("tools.ball3d_run", a.video, "--output", a.output, *(["--labels"] if a.labels else []))
    step("tools.ball3d_viewer", a.video, "--output", a.output)
    if a.render_video:
        step("tools.ball3d_video", a.video, "--output", a.output)
    print(f"\n[volleyvision] {stem}: done → {paths.clip_dir(stem, a.output)}/ball3d_view.html")
    report(stem, t0)


if __name__ == "__main__":
    main()
