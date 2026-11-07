"""Interactive 3D view of a clip's ball (one self-contained HTML page).

    python -m tools.ball3d_viewer videos/test74.mov      # needs output/<clip>/ball3d.json + data/cameras/<clip>.json

Writes output/<clip>/ball3d_view.html: court, net, the phone's position, the ball flying (free
flights only), its floor shadow and height; play / scrub / speed; views: orbit, from the phone
(same angle as the video), side, top, behind the far end; a list of flights (apex, speed, net
clearance), click one to jump to it. Frame numbers match the video and tools.ball3d_video.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from core.common import paths
from core.geometry.camera import Camera

TEMPLATE = Path(__file__).with_name("ball3d_viewer.html")
UNSURE_M, UNSURE_N = 1.0, 8        # flights with more uncertainty / fewer 2D points than this are drawn as unsure


def build(stem: str, output: str = str(paths.OUTPUT)) -> Path:
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    cj = json.load(open(Path("data/cameras") / f"{stem}.json"))
    B = json.load(open(Path(output) / stem / "ball3d.json"))
    fl = [f for f in B["flights"]]
    keep = {i for i, f in enumerate(fl) if f["free_flight"]}
    frames = [[int(k)] + [round(v, 3) for v in fr["xyz"]] + [fr["flight"]]
              for k, fr in sorted(B["frames"].items(), key=lambda kv: int(kv[0])) if fr["flight"] in keep]
    R = cam.R
    data = {
        "clip": stem, "fps": B["fps"], "source": B.get("source", "track"),
        "n": max(int(k) for k in B["frames"]) + 1 if B["frames"] else 0,
        "net": cj.get("net_height", 2.43),
        "camera": {"pos": [round(float(v), 3) for v in cam.position],
                   "fwd": [round(float(v), 5) for v in R[2]], "down": [round(float(v), 5) for v in R[1]],
                   "vfov": round(math.degrees(2 * math.atan(cam.height / 2 / cam.f)), 3)},
        "flights": [{"id": i, "f0": f["f0"], "f1": f["f1"], "apex": f["apex_m"], "speed": f["speed_start_mps"],
                     "net": f["net_crossing"]["clears_net_by_m"] if f["net_crossing"] else None,
                     "unc": f["depth_std_m"],
                     "unsure": bool(f["depth_std_m"] > UNSURE_M or f.get("n_points", 99) < UNSURE_N)}
                    for i, f in enumerate(fl) if i in keep],
        "frames": frames,
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    html = html.replace("__CLIP__", stem)
    out = Path(output) / stem / "ball3d_view.html"
    head = ('<!doctype html><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    out.write_text(head + html)                    # a standalone page to open in a browser
    out.with_name("ball3d_view.body.html").write_text(html)   # the same page without the skeleton (to publish)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    a = ap.parse_args()
    out = build(Path(a.video).stem, a.output)
    print(f"[ball3d] viewer  →  {out}  (open it in a browser)")


if __name__ == "__main__":
    main()
