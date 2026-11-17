"""The 3D from your labels, to check them (dev): ball, touch marks, player numbers, who touched it.

    python -m tools.labeled videos/test.MP4            # 3D + viewer + re-check list
    python -m tools.labeled videos/test.MP4 --video    # + the 3D video, rendered exactly like main's
    python -m tools.labeled videos/test.MP4 --video --players all   # ... with every player tagged (a check)

main.py is the public pipeline and uses none of your labels; this is the same 3D built from them,
written next to it as output/<clip>/ball3d_labels.* (ball3d_labels.mp4, ball3d_labels_view.html,
touches_labels.json), so the two can be compared. Run by tools.annotate (step 5).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from core.common import paths


def _run(args: list[str]):
    print(f"\n[labeled] python -m {' '.join(args)}", flush=True)
    if subprocess.run([sys.executable, "-m", *args]).returncode:
        sys.exit(f"[labeled] stopped: python -m {args[0]} failed (its error is above)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--video", dest="render_video", action="store_true",
                    help="also ball3d_labels.mp4, rendered like main's video")
    ap.add_argument("--players", choices=["none", "touch", "all"], default="none",
                    help="player tags in the video: none (default, as in main), touch, or all (to check numbers)")
    a = ap.parse_args()
    stem = Path(a.video).stem
    if not paths.track(stem, a.output).exists():
        sys.exit(f"[labeled] {stem}: run the pipeline first: python -m main {a.video}")
    _run(["tools.ball3d_run", a.video, "--output", a.output, "--use-labels"])
    _run(["tools.ball3d_viewer", a.video, "--output", a.output, "--labels"])
    if a.render_video:
        _run(["tools.ball3d_video", a.video, "--output", a.output, "--labels", "--players", a.players])
    _run(["tools.check_labels", stem, "--output", a.output])
    d = paths.clip_dir(stem, a.output)
    print(f"\n[labeled] {stem}: look at {d / 'ball3d_labels_view.html'}"
          + (f" and {d / 'ball3d_labels.mp4'}" if a.render_video else ""))


if __name__ == "__main__":
    main()
