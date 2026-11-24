"""2D ball tracking for one clip: the ball in every frame, its flights and the touches.

    python -m tools.track2d videos/match.mov               # + tracked.mp4 (the 2D overlay)
    python -m tools.track2d videos/match.mov --no-render   # track only

The first step of main.py (the full pipeline); run it alone when you only want the 2D track.
"""
import argparse
import sys
from pathlib import Path

from core.ball.finders import FINDER_DEFAULT


def main():
    parser = argparse.ArgumentParser(prog="tools.track2d", description=__doc__.split("\n")[0])
    parser.add_argument("input",        help="Path to the input video file (mp4, mov, etc.)")
    parser.add_argument("--output",     default="output",          help="Output directory")
    parser.add_argument("--model",      default="models/best4.pt", help="Path to YOLO weights")
    parser.add_argument("--finder",     default=FINDER_DEFAULT,    help="ball finder (core.ball.finders)")
    parser.add_argument("--no-render",  action="store_true",       help="Track only: no tracked.mp4 (saves ~45 s)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: video file not found at {args.input}")
        sys.exit(1)

    from core.pipeline import run
    run(input_path=args.input, output_dir=args.output, model_path=args.model, finder=args.finder,
        render=not args.no_render)


if __name__ == "__main__":
    main()
