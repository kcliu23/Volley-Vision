import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="volleyvision",
        description="VolleyVision — volleyball ball tracking. Default: finder-first (WASB on every frame, "
                    "YOLO on crops around its candidates, joint linker, physics). --pipeline classic: "
                    "YOLO on every frame + rule linker + zoom search + TrackNet.",
    )
    parser.add_argument("input",        help="Path to the input video file (mp4, mov, etc.)")
    parser.add_argument("--output",     default="output",          help="Output directory")
    parser.add_argument("--model",      default="models/best4.pt", help="Path to YOLO weights")
    parser.add_argument("--pipeline",   choices=["finder-first", "classic"], default="finder-first",
                        help="finder-first (default, best on the holdout clips) or classic (the old pipeline)")
    parser.add_argument("--finder",     default="models/wasb/wasb_ft2.pth.tar",
                        help="(finder-first) ball finder checkpoint, WASB")
    parser.add_argument("--crop-top",   type=int, default=1,
                        help="(finder-first) YOLO-check up to this many finder candidates per frame")
    parser.add_argument("--crop-p",     type=float, default=0.6,
                        help="(finder-first) ... those with calibrated P(ball) >= this")
    parser.add_argument("--loose",      action="store_true",
                        help="(finder-first) the top 2 candidates in every frame, no threshold (~2.5x the YOLO work)")
    parser.add_argument("--linker",     choices=["rule", "cost"], default="rule",
                        help="(classic) rule or cost (global optimization)")
    parser.add_argument("--no-zoom",    action="store_true",       help="(classic) Skip zoom search")
    parser.add_argument("--redetect",   action="store_true",       help="(classic) Re-run the YOLO pass even if cached")
    parser.add_argument("--trail-len",  type=int,   default=40,    help="Length of the visual ball trail (frames)")
    parser.add_argument("--ring",       action="store_true",       help="Hollow ring marker (ball stays visible)")
    parser.add_argument("--preview",    action="store_true",       help="Show the render while it is written")
    parser.add_argument("--no-touches", action="store_true",       help="Skip touch detection / overlay")
    parser.add_argument("--tracknet",   default="models/tracknet/FT_TrackNet_best.pt",
                        help="(classic) gap-filling finder: TrackNet (.pt) or WASB (.pth.tar)")
    parser.add_argument("--no-tracknet", action="store_true",      help="(classic) Skip the TrackNet stage")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: video file not found at {args.input}")
        sys.exit(1)

    from core.video_pipeline import run
    run(input_path=args.input, output_dir=args.output, model_path=args.model, linker=args.linker,
        zoom=not args.no_zoom, redetect=args.redetect, trail_len=args.trail_len,
        show_preview=args.preview, touches=not args.no_touches, ring=args.ring,
        tracknet=None if args.no_tracknet else args.tracknet, pipeline=args.pipeline, finder=args.finder,
        crop_top=2 if args.loose else args.crop_top, crop_p=None if args.loose else args.crop_p)


if __name__ == "__main__":
    main()
