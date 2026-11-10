"""The 3D ball as a video, drawn the way the tracker's video looks, plus player numbers and ball height.

    python -m tools.ball3d_video videos/test74.mov            # needs output/<clip>/ball3d.json (tools.ball3d_run)
    python -m tools.ball3d_video videos/test74.mov --debug    # + floor shadow, mini top view, frame number

Per frame:
  ball      the 3D ball projected into the picture: the tracker's fading trail and yellow dot,
            with its height in meters beside it. Frames the 3D left out (held / rolling / bad
            fits) show the 2D track's ball with no height, so the ball never just vanishes.
  players   your number (#4) above each player's head, "?" for a player nobody named (only on
            or next to the court, so the bench and crowd stay quiet). While a player is in the
            air the tag turns orange and shows their feet height.
Plays in real time on phone clips with uneven frame timing, keeps the clip's sound.
Writes output/<clip>/ball3d.mp4.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from core.common import paths
from core.common.render import draw_ball, draw_trail, finish_video
from core.geometry.camera import Camera
from core.geometry.court import ATTACK, LENGTH, WIDTH

TRAIL = 15                                  # frames of trail, as in the tracker's video
NEAR_COURT = (LENGTH / 2 + 1.5, WIDTH / 2 + 1.5)   # where unnamed players still get a "?"
YELLOW, ORANGE, GREY = (0, 255, 255), (0, 140, 255), (200, 200, 200)


def put_tag(cv2, img, text, center, color, scale):
    """Text centred on a point, on a dark box so it reads over any background."""
    font, th = cv2.FONT_HERSHEY_SIMPLEX, max(2, int(round(2 * scale)))
    (w, h), base = cv2.getTextSize(text, font, scale, th)
    x, y = int(center[0] - w / 2), int(center[1] + h / 2)
    pad = int(6 * scale)
    sub = img[max(0, y - h - pad):y + base + pad, max(0, x - pad):x + w + pad]
    sub[:] = (sub * 0.45).astype(np.uint8)
    cv2.putText(img, text, (x, y), font, scale, color, th, cv2.LINE_AA)


def main():
    import cv2

    from core.people.players import UNNAMED, load_people
    from core.pipeline import render_timing
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--debug", action="store_true", help="also the floor shadow, a mini top view and frame numbers")
    a = ap.parse_args()
    stem = Path(a.video).stem
    clip = Path(a.output) / stem
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    B = json.load(open(clip / "ball3d.json"))
    fps = B["fps"]
    ball3d = {int(k): np.array(v["xyz"]) for k, v in B["frames"].items() if v.get("free")}
    track2d = {}
    if paths.track(stem, a.output).exists():
        for row in json.load(open(paths.track(stem, a.output)))["frames"]:
            if row.get("pos") is not None:
                track2d[int(row["f"])] = tuple(row["pos"])

    people: dict[int, list] = {}
    for P in load_people(stem, a.output, cam, fps, log=lambda m: print(f"[ball3d_video] {stem}: " + m)) or []:
        for i, f in enumerate(P.frames):
            g = P.ground[i]
            if not np.isfinite(g[0]):
                continue
            if P.id >= UNNAMED and not (abs(g[0]) < NEAR_COURT[0] and abs(g[1]) < NEAR_COURT[1]):
                continue
            people.setdefault(int(f), []).append((P.id, g, bool(P.airborne[i]), float(P.feet_z[i]), P.height_m))

    times = None
    if (clip / "frame_times.json").exists():
        from core.common.frametime import frame_times
        times = list(frame_times(a.video, clip) / fps)
    out_fps, repeats = render_timing(times, fps)

    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = H / 1080
    tmp = os.path.join(tempfile.gettempdir(), f"{stem}_ball3d.mp4")
    out = clip / "ball3d.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    mini = MiniTop(cv2, cam) if a.debug else None

    uv: list = []                            # the ball in the picture, per frame (None = no ball)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        X = ball3d.get(f)
        p = cam.project([X])[0] if X is not None else track2d.get(f)
        uv.append(p)

        # players first, so the ball draws over their tags
        for pid, g, air, fz, hm in people.get(f, []):
            head = cam.project([[g[0], g[1], hm + fz + 0.25]])[0]
            name = "?" if pid >= UNNAMED else f"#{pid}"
            if air:
                put_tag(cv2, img, f"{name}  {fz:.2f} m up", head, ORANGE, 1.0 * scale)
            else:
                put_tag(cv2, img, name, head, GREY if pid >= UNNAMED else (255, 255, 255), 0.9 * scale)

        if a.debug and X is not None:
            s = cam.project([[X[0], X[1], 0.0]])[0]
            cv2.line(img, (int(p[0]), int(p[1])), (int(s[0]), int(s[1])), YELLOW, 2, cv2.LINE_AA)
            cv2.drawMarker(img, (int(s[0]), int(s[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 4)

        draw_trail(img, uv[max(0, f - TRAIL + 1):])
        if p is not None:
            draw_ball(img, p)
            if X is not None:
                put_tag(cv2, img, f"{X[2]:.1f} m", (p[0] + 70 * scale, p[1] - 30 * scale), YELLOW, 1.0 * scale)

        if mini is not None:
            mini.draw(img, f, X, people.get(f, []), W, H)
            cv2.putText(img, f"frame {f}", (int(40 * scale), int(60 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * scale, (255, 255, 255), 3)
        for _ in range(repeats[f] if repeats is not None and f < len(repeats) else 1):
            vw.write(img)
        f += 1
    vw.release()
    cap.release()
    finish_video(tmp, a.video, str(out))
    print(f"[ball3d] {stem}: video  →  {out}")


class MiniTop:
    """--debug: a small top view of the court with the ball's recent path and the players."""
    S = 22                                   # px per meter

    def __init__(self, cv2, cam):
        self.cv2 = cv2
        self.base = np.full((int(14 * self.S), int(28 * self.S), 3), 40, np.uint8)
        hx, hy = LENGTH / 2, WIDTH / 2
        for p, q in (((-hx, -hy), (hx, -hy)), ((-hx, hy), (hx, hy)), ((-hx, -hy), (-hx, hy)), ((hx, -hy), (hx, hy)),
                     ((0, -hy), (0, hy)), ((-ATTACK, -hy), (-ATTACK, hy)), ((ATTACK, -hy), (ATTACK, hy))):
            cv2.line(self.base, self.top(*p), self.top(*q), (90, 200, 90), 1, cv2.LINE_AA)
        cv2.line(self.base, self.top(0, -hy - 0.5), self.top(0, hy + 0.5), (230, 230, 230), 2)
        c = cam.position
        cv2.circle(self.base, self.top(c[0], c[1]), 5, (60, 60, 230), -1)
        self.trail: list = []

    def top(self, x, y):
        return int((x + 15) * self.S), int((7 - y) * self.S)

    def draw(self, img, f, X, people, W, H):
        cv2, m = self.cv2, self.base.copy()
        self.trail = (self.trail + [None if X is None else (X[0], X[1])])[-45:]
        pts = [p for p in self.trail if p is not None]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(m, self.top(*p), self.top(*q), (0, 200, 255), 2, cv2.LINE_AA)
        for _pid, g, air, _fz, _hm in people:
            if -15 < g[0] < 13 and -7 < g[1] < 7:
                cv2.circle(m, self.top(g[0], g[1]), 4, ORANGE if air else (255, 220, 0), -1)
        if X is not None:
            cv2.circle(m, self.top(X[0], X[1]), 6, YELLOW, -1)
        mh, mw = m.shape[:2]
        img[H - mh - 10:H - 10, W - mw - 250:W - 250] = m


if __name__ == "__main__":
    main()
