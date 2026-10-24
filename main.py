import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="volleyvision",
        description="VolleyVision — volleyball ball tracking (two-pass: detect once, then link the whole clip)",
    )
    parser.add_argument("input",        help="Path to the input video file (mp4, mov, etc.)")
    parser.add_argument("--output",     default="output",          help="Output directory")
    parser.add_argument("--model",      default="models/best4.pt", help="Path to YOLO weights")
    parser.add_argument("--linker",     choices=["rule", "cost"], default="rule",
                        help="rule (default, best measured) or cost (global optimization)")
    parser.add_argument("--no-zoom",    action="store_true",       help="Skip zoom search (faster, fewer recovered frames)")
    parser.add_argument("--redetect",   action="store_true",       help="Re-run the YOLO pass even if cached")
    parser.add_argument("--trail-len",  type=int,   default=40,    help="Length of the visual ball trail (frames)")
    parser.add_argument("--ring",       action="store_true",       help="Hollow ring marker (ball stays visible)")
    parser.add_argument("--preview",    action="store_true",       help="Show the render while it is written")
    parser.add_argument("--no-touches", action="store_true",       help="Skip touch detection / overlay")
    parser.add_argument("--tracknet",   default="models/tracknet/FT_TrackNet_best.pt",
                        help="TrackNet checkpoint: fills gaps / track ends YOLO missed (skipped if the file is missing)")
    parser.add_argument("--no-tracknet", action="store_true",      help="Skip the TrackNet stage")
    parser.add_argument("--legacy",     action="store_true",       help="Old single-pass tracker (YOLO conf>=0.40 + Kalman)")
    parser.add_argument("--conf",       type=float, default=0.40,  help="(legacy only) detection confidence threshold")
    parser.add_argument("--raw",        action="store_true",       help="(legacy only) show raw YOLO boxes")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: video file not found at {args.input}")
        sys.exit(1)

    if args.legacy:
        from core.video_pipeline import run_legacy
        run_legacy(input_path=args.input, output_dir=args.output, model_path=args.model, conf=args.conf,
                   trail_len=args.trail_len, show_preview=args.preview, raw=args.raw,
                   touches=not args.no_touches)
        return

    from core.video_pipeline import run
    run(input_path=args.input, output_dir=args.output, model_path=args.model, linker=args.linker,
        zoom=not args.no_zoom, redetect=args.redetect, trail_len=args.trail_len,
        show_preview=args.preview, touches=not args.no_touches, ring=args.ring,
        tracknet=None if args.no_tracknet else args.tracknet)


if __name__ == "__main__":
    main()
