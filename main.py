import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="volleyvision",
        description="VolleyVision — volleyball ball tracking",
    )
    parser.add_argument("input",        help="Path to the input video file (mp4, mov, etc.)")
    parser.add_argument("--output",     default="output",          help="Directory to save the tracked video")
    parser.add_argument("--model",      default="models/best4.pt", help="Path to YOLO weights")
    parser.add_argument("--conf",       type=float, default=0.40,  help="Confidence threshold for ball detection")
    parser.add_argument("--trail-len",  type=int,   default=40,    help="Length of the visual ball trail (frames)")
    parser.add_argument("--no-preview", action="store_true",       help="Disable the real-time preview window")
    parser.add_argument("--raw",        action="store_true",       help="Show raw YOLO detections without Kalman/trail")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: video file not found at {args.input}")
        sys.exit(1)

    from core.video_pipeline import run

    run(
        input_path=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf=args.conf,
        trail_len=args.trail_len,
        show_preview=not args.no_preview,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()
