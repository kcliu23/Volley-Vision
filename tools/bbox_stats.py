"""
Run from the Volley-Vision folder:
    python bbox_stats.py tests/test.MP4
    python bbox_stats.py tests/test2.MP4   # run on a few for better coverage

Prints bbox size distribution across all detections so you can pick
good _MIN_BALL_PX / _MAX_BALL_PX values for the size filter.
"""
import sys
import numpy as np
from ultralytics import YOLO

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "tests/test.MP4"
MODEL = "models/best3.pt"
CONF  = 0.30   # low threshold — captures borderline detections too

model = YOLO(MODEL)
rows  = []   # (conf, w, h)

for result in model.predict(source=VIDEO, stream=True, conf=CONF, imgsz=1280, verbose=False):
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        w = float(x2 - x1)
        h = float(y2 - y1)
        c = float(box.conf[0])
        rows.append((c, w, h))

if not rows:
    print("No detections found.")
    sys.exit()

confs   = np.array([r[0] for r in rows])
widths  = np.array([r[1] for r in rows])
heights = np.array([r[2] for r in rows])
sizes   = np.maximum(widths, heights)

print(f"\n{'='*55}")
print(f"Video            : {VIDEO}")
print(f"Total detections : {len(rows)}")
print(f"\n  Width   min={widths.min():.1f}  max={widths.max():.1f}  "
      f"mean={widths.mean():.1f}  median={np.median(widths):.1f}")
print(f"  Height  min={heights.min():.1f}  max={heights.max():.1f}  "
      f"mean={heights.mean():.1f}  median={np.median(heights):.1f}")
print(f"  Conf    min={confs.min():.2f}  max={confs.max():.2f}  "
      f"mean={confs.mean():.2f}  median={np.median(confs):.2f}")

print(f"\n  Max-side (px) distribution:")
buckets = [(0,15),(15,30),(30,50),(50,75),(75,100),(100,150),(150,9999)]
for lo, hi in buckets:
    mask  = (sizes >= lo) & (sizes < hi)
    count = mask.sum()
    avg_c = confs[mask].mean() if count else 0
    bar   = '█' * int(count / max(len(sizes), 1) * 50)
    label = f"{hi}px" if hi < 9999 else "∞"
    print(f"    {lo:4d}–{label:>6} : {count:5d} det  avg_conf={avg_c:.2f}  {bar}")

print(f"\n  Largest 10 detections (likely false positives):")
top_idx = np.argsort(sizes)[-10:][::-1]
for i in top_idx:
    print(f"    {sizes[i]:.0f}px  (w={widths[i]:.0f} h={heights[i]:.0f}  conf={confs[i]:.2f})")

print(f"{'='*55}\n")
