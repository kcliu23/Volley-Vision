"""The 3D scene for one clip, one command: frame timing -> players -> ball in 3D -> viewer.

    python -m tools.scene3d videos/test75.mov --labels           # from your checked labels
    python -m tools.scene3d videos/test75.mov                    # from the tracker's output
    python -m tools.scene3d videos/test75.mov --labels --video   # + the overlay video
    python -m tools.scene3d videos/test75.mov --no-players       # skip person detection (slow)

Needs the court calibration (tools.calibrate_court -> data/cameras/<clip>.json) and the 2D track
(main.py or autolabel -> output/<clip>/{track,contacts}.json). Each step is cached: frame times
and players are computed once per clip; delete output/<clip>/{frame_times,players}.json to redo.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from core.common import paths


def step(*args):
    print(f"\n[scene3d] python -m {' '.join(args)}")
    subprocess.run([sys.executable, "-m", *args], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--labels", action="store_true", help="use your checked labels instead of the tracker's output")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--no-players", action="store_true", help="skip person detection (no player check on touches)")
    ap.add_argument("--video", dest="render_video", action="store_true", help="also render the overlay video (ball3d.mp4)")
    a = ap.parse_args()
    stem = Path(a.video).stem
    if not (Path("data/cameras") / f"{stem}.json").exists():
        sys.exit(f"[scene3d] {stem}: no court calibration yet. Run first:\n"
                 f"  python -m tools.calibrate_court {a.video}")
    if not paths.track(stem, a.output).exists():
        sys.exit(f"[scene3d] {stem}: no 2D track yet. Run first:\n  python main.py {a.video}")

    from core.common.frametime import frame_times, summary
    print(f"[scene3d] {stem}: frame timing: {summary(frame_times(a.video, paths.clip_dir(stem, a.output)))}")
    if not a.no_players and not (paths.clip_dir(stem, a.output) / "players.json").exists():
        step("tools.players", a.video, "--output", a.output)
    step("tools.ball3d_run", a.video, "--output", a.output, *(["--labels"] if a.labels else []))
    step("tools.ball3d_viewer", a.video, "--output", a.output)
    if a.render_video:
        step("tools.ball3d_video", a.video, "--output", a.output)
    print(f"\n[scene3d] {stem}: done → {paths.clip_dir(stem, a.output)}/ball3d_view.html")


if __name__ == "__main__":
    main()
