"""Run touch detection on a saved trajectory, without re-running YOLO.

    python -m tools.dev.detect_touches output/test10/track.json
    python -m tools.dev.detect_touches output/archive/*.clip.json  # old May exports work too
    python -m tools.dev.detect_touches output/test10/track.json --frames videos/test10.mov

--frames writes a contact sheet per touch (frames -6..+6 cropped around the
ball) to output/dev/touch_sheets/ so you can eyeball whether each one is real.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.touch_detector import detect_touches, TouchParams  # noqa: E402


def load_track(path: str):
    """Returns (frames, xs, ys, fps, w, h) of raw detections in pixels."""
    d = json.load(open(path))
    fps, w, h = d["fps"], d["width"], d["height"]
    if d.get("schema") == "vv-track/2":      # two-pass pipeline: real positions only (filled = guesses)
        pts = [(r["f"], r["pos"][0], r["pos"][1]) for r in d["frames"]
               if r.get("pos") and r.get("state") in ("detected", "weak")]
    elif d.get("schema") == "vv-track/1":
        pts = [(r["f"], r["det"][0], r["det"][1]) for r in d["frames"] if r.get("det")]
    else:  # vv-clip/1 (May dashboard export): normalised coords
        pts = [(r["f"], r["x"] * w, r["y"] * h) for r in d["detections"] if r.get("x") is not None]
    pts.sort()
    f, x, y = (np.array(a, dtype=float) for a in zip(*pts)) if pts else ([], [], [])
    return f, x, y, fps, w, h, d


def contact_sheet(video: str, touches, out_dir: Path, stem: str, offsets=range(-6, 7, 2), crop=360):
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    for k, t in enumerate(touches):
        tiles = []
        for o in offsets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t.frame + o)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((crop, crop, 3), np.uint8)
            else:
                H, W = img.shape[:2]
                cx = int(np.clip(t.x, crop // 2, W - crop // 2))
                cy = int(np.clip(t.y, crop // 2, H - crop // 2))
                img = img[cy - crop // 2: cy + crop // 2, cx - crop // 2: cx + crop // 2].copy()
            cv2.putText(img, f"{o:+d}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            tiles.append(img)
        sheet = np.hstack(tiles)
        label = f"{stem} touch {k} f={t.frame} t={t.t:.2f}s ang={t.angle_deg:.0f} imp={t.impulse:.2f}"
        cv2.putText(sheet, label, (8, crop - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"{stem}_t{k:02d}_f{t.frame}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--frames", help="source video, to write contact sheets (single track only)")
    ap.add_argument("--json", action="store_true", help="also write output/dev/<clip>_touches.json")
    args = ap.parse_args()

    for p in args.tracks:
        f, x, y, fps, w, h, d = load_track(p)
        stem = Path(p).parent.name if Path(p).name == "track.json" else Path(p).name.split(".")[0].replace("_track", "")
        touches = detect_touches(f, x, y, fps, h, TouchParams())
        print(f"\n{stem}: {len(f)} detections, {len(touches)} touches")
        print("   frame      t   ang  imp   rel  gain  v_in→v_out (px/s)   gap")
        for t in touches:
            print(f"  {t.frame:5d} {t.t:6.2f} {t.angle_deg:5.0f} {t.impulse:4.2f} {t.rel_impulse:5.2f} {t.gain:5.0f}"
                  f"  {t.speed_in:6.0f}→{t.speed_out:<6.0f}   {'gap' if t.in_gap else ''}")
        if args.json:
            out = Path("output/dev") / f"{stem}_touches.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump({"schema": "vv-touches/1", "fps": fps, "touches": [t.to_dict() for t in touches]},
                      open(out, "w"), indent=1)
        if args.frames:
            contact_sheet(args.frames, touches, Path("output/dev/touch_sheets"), stem)


if __name__ == "__main__":
    main()
