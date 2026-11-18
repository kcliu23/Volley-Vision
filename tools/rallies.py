"""Rallies and the score again, from the saved 3D (no 3D rerun): output/<clip>/rallies.json.

    python -m tools.rallies videos/match.mov            # + --labels for the 3D from your labels
"""
from __future__ import annotations

import argparse
from pathlib import Path

from core.common import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--labels", action="store_true")
    a = ap.parse_args()
    from tools.ball3d_run import write_rallies
    stem = Path(a.video).stem
    write_rallies(paths.clip_dir(stem, a.output), "_labels" if a.labels else "", stem)


if __name__ == "__main__":
    main()
