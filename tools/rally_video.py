"""The clip with the dead time cut out: every rally, back to back, with the score.

    python -m tools.rally_video videos/match.mov            # from output/<clip>/rallies.json
    python -m tools.rally_video videos/match.mov --labels   # from rallies_labels.json (tools.labeled)

Each rally runs from PRE_S before the serve to POST_S after the ball lands. The picture is the 3D
video (ball3d.mp4) when it exists, else the original. The score sits top right and changes when a
rally ends. Writes output/<clip>/rallies.mp4 (no sound: the cuts would break it).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from core.common import paths
from core.common.labels import Labels

PRE_S = 2.5    # s before the rally start: the server bouncing + the toss (your call, Nov 21: 2-3 s)
POST_S = 1.5


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--labels", action="store_true", help="rallies_labels.json / ball3d_labels.mp4")
    a = ap.parse_args()
    stem = Path(a.video).stem
    d = paths.clip_dir(stem, a.output)
    sfx = "_labels" if a.labels else ""
    R = json.load(open(d / f"rallies{sfx}.json"))["rallies"]
    src = d / f"ball3d{sfx}.mp4"
    src = str(src) if src.exists() else a.video
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    L = Labels(H)
    keep = []                                            # (first frame, last frame, rally index)
    for i, r in enumerate(R):
        a0, b0 = max(0, r["f0"] - int(PRE_S * fps)), min(n - 1, r["f1"] + int(POST_S * fps))
        if keep and a0 <= keep[-1][1]:
            a0 = keep[-1][1] + 1
        if a0 <= b0:
            keep.append((a0, b0, i))
    tmp = os.path.join(tempfile.gettempdir(), f"{stem}_rallies.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for a0, b0, i in keep:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a0)
        r = R[i]
        before = (R[i - 1]["near"], R[i - 1]["far"]) if i else (0, 0)
        for f in range(a0, b0 + 1):
            ok, img = cap.read()
            if not ok:
                break
            nr, fr_ = (r["near"], r["far"]) if f > r["f1"] else before
            x = img.shape[1] - int(24 * H / 1080)
            y = int(24 * H / 1080)
            plates = [(f"FAR  {fr_}", Labels.FAR), (f"NEAR  {nr}", Labels.NEAR)]
            for text, col in plates:                     # right-aligned, stacked: far on top (as on court)
                spr_w = L._plate(text, col, 26).shape[1]
                L.tag(img, text, (x - spr_w, y + int(18 * H / 1080)), col, 26)
                y += int(44 * H / 1080)
            vw.write(img)
    vw.release()
    out = d / f"rallies{sfx}.mp4"
    os.replace(tmp, out)
    total = sum(b - a + 1 for a, b, _ in keep)
    print(f"[rallies] {stem}: {len(keep)} rallies, {total / fps:.1f} s of {n / fps:.1f} s kept  →  {out}")


if __name__ == "__main__":
    main()
