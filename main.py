import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="volleyvision",
        description="VolleyVision — Volleyball Analytics & Ball Tracking Platform",
    )
    sub = parser.add_subparsers(dest="command", required=True, help="Task to perform")

    # ── video sub-command ──────────────────────────────────────────────────
    vid = sub.add_parser("video", help="Analyze a pre-recorded volleyball match")
    vid.add_argument("input",            help="Path to the input video file (mp4, avi, etc.)")
    vid.add_argument("--output",         default="output", help="Directory to save the tracked video")
    vid.add_argument("--model",          default="models/best.pt", help="Path to your custom YOLO weights")
    vid.add_argument("--conf",           type=float, default=0.30, help="Confidence threshold for ball detection")
    vid.add_argument("--trail-len",      type=int,   default=40,   help="Length of the visual ball trajectory")
    vid.add_argument("--no-preview",     action="store_true",      help="Disable the real-time preview window")
    vid.add_argument("--raw",            action="store_true",      help="Show raw YOLO detections without Kalman/trail")
    vid.add_argument("--net-width-px",  type=int, default=None,   help="Net width in pixels (pole to pole = 9m) for speed calibration")
    vid.add_argument("--net-height-px", type=int, default=None,   help="Net height in pixels (top to bottom = 1m) for speed calibration")

    # ── live sub-command ───────────────────────────────────────────────────
    # Placeholder for future live implementation (e.g., using a USB/CSI camera)
    # lv = sub.add_parser("live", help="Run ball tracking on a live camera feed")
    # lv.add_argument("--camera",         default=0, type=int, help="Camera index (usually 0 for built-in)")
    # lv.add_argument("--model",          default="models/YOLOv26n.pt")
    # lv.add_argument("--conf",           type=float, default=0.35)

    args = parser.parse_args()

    # Route to the correct pipeline
    if args.command == "video":
        from pipeline.video_pipeline import run as run_video_pipeline

        input_file = Path(args.input)
        if not input_file.exists():
            print(f"Error: Video file not found at {args.input}")
            sys.exit(1)

        print("=" * 40)
        print("VolleyVision — Configuration")
        print("=" * 40)
        print(f"  Input:       {args.input}")
        print(f"  Output dir:  {args.output}")
        print(f"  Model:       {args.model}")
        print(f"  Confidence:  {args.conf}")
        print(f"  Trail len:   {args.trail_len}")
        print(f"  Preview:     {not args.no_preview}")
        print(f"  Raw mode:    {args.raw}")
        print(f"  Calibration: {f'{args.net_width_px}x{args.net_height_px} px' if args.net_width_px and args.net_height_px else 'interactive'}")
        print("=" * 40)

        run_video_pipeline(
            input_path=args.input,
            output_dir=args.output,
            model_path=args.model,
            conf=args.conf,
            trail_len=args.trail_len,
            show_preview=not args.no_preview,
            raw=args.raw,
            net_width_px=args.net_width_px,
            net_height_px=args.net_height_px,
        )

    elif args.command == "live":
        print("Live pipeline is not yet implemented. Use 'video' for now.")
        # When ready, you'll call run_live_pipeline here.

if __name__ == "__main__":
    main()