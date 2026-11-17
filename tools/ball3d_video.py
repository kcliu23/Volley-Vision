"""The 3D ball as a video, drawn the way the tracker's video looks, plus player numbers and ball height.

    python -m tools.ball3d_video videos/test74.mov            # needs output/<clip>/ball3d.json (tools.ball3d_run)
    python -m tools.ball3d_video videos/test74.mov --debug    # + floor shadow, mini top view, frame number

Per frame:
  ball      the 3D ball projected into the picture: the tracker's fading trail and yellow dot,
            with its height in meters beside it. Frames the 3D left out (held / rolling / bad
            fits) show the 2D track's ball with no height, so the ball never just vanishes.
  players   your number on a small plate just above each player's head (from the pose's face points),
            cyan on the camera's side of the net, fuchsia on the far side (legend bottom left);
            "?" in grey for a player nobody named (only on or
            next to the court, so the bench and crowd stay quiet). In the air: "#4 ↑ 0.28 m",
            the feet height.
Labels: core.common.labels (one look for all overlay text).
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
from core.common.labels import Labels
from core.common.render import draw_ball, draw_trail, finish_video
from core.geometry.camera import Camera
from core.geometry.court import ATTACK, LENGTH, WIDTH

TRAIL = 15                                  # frames of trail, as in the tracker's video
NEAR_COURT = (LENGTH / 2 + 1.5, WIDTH / 2 + 1.5)   # where unnamed players still get a "?"
YELLOW, ORANGE = (0, 255, 255), (0, 140, 255)      # BGR, debug drawings only


def player_near(P, cam) -> bool:
    """Is this player on the camera's side of the net? Players don't change sides during a rally,
    so it's one answer per player, not per frame (feet in the air or hidden give nonsense floor
    spots: the side flickered). A numbered player: from the number (1-6 near, 7-12 far); an
    unnumbered one: the side their feet are on most, counting only frames with the feet on the
    floor, on the court area, clear of the net."""
    from core.people.players import UNNAMED
    if P.id < UNNAMED:
        return P.id <= 6
    g = P.ground
    ok = ~P.airborne & np.isfinite(g[:, 0]) & (np.abs(g[:, 0]) > 1.0) & (np.abs(g[:, 0]) < 12.0) & (np.abs(g[:, 1]) < 8.0)
    if not ok.any():
        return True                                       # feet never seen on the floor: likely near (cut off)
    return bool(np.mean(g[ok, 0] * cam.position[0] > 0) >= 0.5)


def main():
    import cv2

    from core.people.players import UNNAMED, load_people
    from core.pipeline import render_timing
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--output", default=str(paths.OUTPUT))
    ap.add_argument("--debug", action="store_true", help="also the floor shadow, a mini top view and frame numbers")
    ap.add_argument("--players", choices=["touch", "all", "none"], default="touch",
                    help="player tags: touch (default) = only whoever touched the ball, fading in and out around "
                         "the touch; all = every player (checking your numbers); none")
    ap.add_argument("--labels", action="store_true", help="the 3D built from your labels (ball3d_run --use-labels): "
                    "ball3d_labels.json -> ball3d_labels.mp4, with your player numbers")
    a = ap.parse_args()
    sfx = "_labels" if a.labels else ""
    stem = Path(a.video).stem
    clip = Path(a.output) / stem
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    B = json.load(open(clip / f"ball3d{sfx}.json"))
    fps = B["fps"]
    ball3d = {int(k): np.array(v["xyz"]) for k, v in B["frames"].items() if v.get("free")}
    track2d = {}                             # where the 3D has no ball: the 2D ball (yours with --labels)
    if a.labels and paths.verified(stem).exists():
        for k, v in (json.load(open(paths.verified(stem))).get("labels") or {}).items():
            if isinstance(v, list):                      # "none" / "hidden" / "skip": no ball drawn
                track2d[int(k)] = tuple(v[:2])
    elif paths.track(stem, a.output).exists():
        for row in json.load(open(paths.track(stem, a.output)))["frames"]:
            if row.get("pos") is not None:
                track2d[int(row["f"])] = tuple(row["pos"])

    heads = head_points(clip / "pose.json")
    people: dict[int, list] = {}
    for P in load_people(stem, a.output, cam, fps, log=lambda m: print(f"[ball3d_video] {stem}: " + m),
                         use_labels=a.labels) or []:
        near = player_near(P, cam)
        for i, f in enumerate(P.frames):
            g = P.ground[i]
            on_floor = bool(np.isfinite(g[0]))
            if P.id >= UNNAMED and not (on_floor and abs(g[0]) < NEAR_COURT[0] and abs(g[1]) < NEAR_COURT[1]):
                continue
            b = P.box[i]
            head = heads.get((int(f), _key(b)), ((b[0] + b[2]) / 2, b[1]))
            people.setdefault(int(f), []).append((P.id, head, near, bool(P.airborne[i]), float(P.feet_z[i]), g))

    # touch mode: the toucher's tag, from just before each touch to a little after, fading
    show: dict = {}                          # frame -> {player id: opacity}
    tp = clip / f"touches{sfx}.json"
    if a.players == "touch" and tp.exists():
        pre, hold, fade = int(0.15 * fps), int(0.5 * fps), int(0.3 * fps)
        for t in json.load(open(tp))["touches"]:
            w = t.get("toucher")
            if not w:
                continue
            pid = w["number"] if isinstance(w.get("number"), int) else UNNAMED + int(w["track"])
            f0 = t["frame"]
            for g in range(f0 - pre, f0 + hold + fade + 1):
                o = (g - (f0 - pre)) / max(pre, 1) if g < f0 else (1.0 if g <= f0 + hold else 1 - (g - f0 - hold) / max(fade, 1))
                show.setdefault(g, {})[pid] = max(show.get(g, {}).get(pid, 0.0), min(1.0, max(0.0, o)))

    times = None
    if (clip / "frame_times.json").exists():
        from core.common.frametime import frame_times
        times = list(frame_times(a.video, clip) / fps)
    out_fps, repeats = render_timing(times, fps)

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)            # phone videos stored sideways: upright, as everywhere else
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = H / 1080
    L = Labels(H)
    tmp = os.path.join(tempfile.gettempdir(), f"{stem}_ball3d{sfx}.mp4")
    out = clip / f"ball3d{sfx}.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))
    mini = MiniTop(cv2, cam) if a.debug else None

    eased: dict = {}                         # player id -> (frame, chip point)
    uv: list = []                            # the ball in the picture, per frame (None = no ball)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        X = ball3d.get(f)
        p = cam.project([X])[0] if X is not None else track2d.get(f)
        uv.append(p)

        # players first, so the ball draws over their chips; each chip sits just above the head,
        # eased over a few frames so it doesn't shake with the pose points
        chips = []
        for pid, hp, near, air, fz, _g in people.get(f, []):
            op = 1.0
            if a.players == "none":
                continue
            if a.players == "touch":
                op = show.get(f, {}).get(pid, 0.0)
                if op <= 0.02:
                    continue
            last = eased.get(pid)
            hp = hp if last is None or last[0] < f - 3 else (0.4 * hp[0] + 0.6 * last[1][0], 0.4 * hp[1] + 0.6 * last[1][1])
            eased[pid] = (f, hp)
            head = (hp[0], hp[1] - 6 * scale)
            if pid >= UNNAMED:
                chips.append(("?", head, Labels.UNNAMED, 24, op))
            else:
                text = f"#{pid}  \u2191 {fz:.2f} m" if air else f"#{pid}"
                chips.append((text, head, Labels.NEAR if near else Labels.FAR, 28, op))
        L.chips(img, chips)
        if any(c[2] is not Labels.UNNAMED for c in chips):
            L.legend(img)

        if a.debug and X is not None:
            s = cam.project([[X[0], X[1], 0.0]])[0]
            cv2.line(img, (int(p[0]), int(p[1])), (int(s[0]), int(s[1])), YELLOW, 2, cv2.LINE_AA)
            cv2.drawMarker(img, (int(s[0]), int(s[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 4)

        draw_trail(img, uv[max(0, f - TRAIL + 1):])
        if p is not None:
            draw_ball(img, p)
            if X is not None:
                L.tag(img, f"{X[2]:.2f} m", (p[0] + 26 * scale, p[1] - 26 * scale))

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


def _key(box):
    return tuple(int(round(v)) for v in box[:4])


def head_points(pose_path: Path) -> dict:
    """(frame, box key) -> the top of the head in the picture, from the pose points: the highest
    confident face point (nose / eyes / ears), raised by half its distance to the shoulders."""
    if not pose_path.exists():
        return {}
    out = {}
    for f, people in json.load(open(pose_path))["frames"].items():
        for p in people:
            kp = np.asarray(p.get("kp") or [], float)
            if len(kp) < 7:
                continue
            face = kp[:5][kp[:5, 2] > 0.3]
            sh = kp[5:7][kp[5:7, 2] > 0.3]
            if not len(face):
                continue
            top = face[:, 1].min()
            neck = sh[:, 1].mean() if len(sh) else top + 0.3 * (p["box"][3] - p["box"][1])
            out[(int(f), _key(p["box"]))] = (float(face[:, 0].mean()), float(top - 0.5 * max(neck - top, 0)))
    return out


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
        for _pid, _hp, _near, air, _fz, g in people:
            if np.isfinite(g[0]) and -15 < g[0] < 13 and -7 < g[1] < 7:
                cv2.circle(m, self.top(g[0], g[1]), 4, ORANGE if air else (255, 220, 0), -1)
        if X is not None:
            cv2.circle(m, self.top(X[0], X[1]), 6, YELLOW, -1)
        mh, mw = m.shape[:2]
        img[H - mh - 10:H - 10, W - mw - 250:W - 250] = m


if __name__ == "__main__":
    main()
