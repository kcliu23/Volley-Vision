"""One id per player for a whole clip: join the tracker's broken tracks (tracklets) into players.

The pose tracker (tools.pose) loses a player whenever they cross someone or get hidden, and starts
a new id: a 12-player clip comes out with 30-50 ids. Three things tie the pieces back together:
  side       a player stays on their side of the net for the whole rally: the median of where
             their feet are on the court (calibrated camera) says which side (sign of x)
  place      a new piece must start near where an old piece ended, within running distance for
             the time between them (on the court, in meters, not in pixels)
  look       the torso's colours (jersey, skin): a colour histogram over several frames
Pieces are joined greedily, best match first, never two pieces that are on screen at the same
time. Off-court people (feet outside the court + 1.5 m: the bench, referees, spectators) are kept
apart.

    ids = build_ids(pose_frames, video, cam, fps)
    ids["tracklet id"] -> {"player": n, "side": -1 near / +1 far / 0, "on_court": bool}
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from core.scene3d.params3d import ANKLE_H

RUN_MPS = 6.0          # fastest a player covers ground between two pieces
SLACK_M = 1.5          # plus this (feet jitter, a missed step)
MAX_GAP_S = 4.0        # pieces further apart in time aren't joined
LOOK_MAX = 0.45        # colour distance (Bhattacharyya) above this: not the same person
COURT = (9.0 + 1.5, 4.5 + 1.5)


def _feet(kp):
    if min(kp[15][2], kp[16][2]) > 0.5:
        return ((kp[15][0] + kp[16][0]) / 2, (kp[15][1] + kp[16][1]) / 2)
    return None


def _torso_box(kp, box):
    pts = [kp[i] for i in (5, 6, 11, 12) if kp[i][2] > 0.4]
    if len(pts) >= 3:
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    x1, y1, x2, y2 = box
    h = y2 - y1
    return int(x1), int(y1 + 0.2 * h), int(x2), int(y1 + 0.55 * h)


def _tracklets(pose_frames, cam):
    T = defaultdict(lambda: {"frames": [], "spots": [], "boxes": []})
    for f, people in pose_frames.items():
        f = int(f)
        for p in people:
            if p["id"] < 0:
                continue
            t = T[p["id"]]
            t["frames"].append(f)
            t["boxes"].append((f, _torso_box(p["kp"], p["box"])))
            a = _feet(p["kp"])
            if a is not None:
                g = cam.floor_point([a], ANKLE_H)[0][:2]
                t["spots"].append((f, float(g[0]), float(g[1])))
    for t in T.values():
        t["f0"], t["f1"] = min(t["frames"]), max(t["frames"])
        s = np.array([q[1:] for q in t["spots"]]) if t["spots"] else None
        t["median"] = np.median(s, axis=0) if s is not None else None
        t["on_court"] = s is not None and abs(t["median"][0]) <= COURT[0] and abs(t["median"][1]) <= COURT[1]
        t["side"] = int(np.sign(t["median"][0])) if s is not None and abs(t["median"][0]) > 0.5 else 0
    return dict(T)


def _looks(T, video, per_track: int = 10):
    """Colour histogram (HSV, hue x saturation) of each tracklet's torso, over up to per_track frames."""
    import cv2
    want = defaultdict(list)                               # frame -> [(tid, box)]
    for tid, t in T.items():
        bs = t["boxes"]
        for f, b in bs[::max(1, len(bs) // per_track)][:per_track]:
            want[f].append((tid, b))
    hists = defaultdict(list)
    cap = cv2.VideoCapture(str(video))
    f = 0
    last = max(want) if want else -1
    while f <= last:
        ok, img = cap.read()
        if not ok:
            break
        for tid, (x1, y1, x2, y2) in want.get(f, []):
            crop = img[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
            if crop.size < 30:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
            hists[tid].append(cv2.normalize(h, h).flatten())
        f += 1
    return {tid: np.mean(h, axis=0).astype(np.float32) for tid, h in hists.items() if h}


def build_ids(pose_frames: dict, video: str, cam, fps: float):
    import cv2
    T = _tracklets(pose_frames, cam)
    look = _looks(T, video)
    # players: lists of tracklet ids, joined greedily by the best (lowest cost) valid pair
    groups = {tid: [tid] for tid in T}

    def span(g):
        return min(T[t]["f0"] for t in g), max(T[t]["f1"] for t in g)

    def overlaps(g1, g2):
        s1 = set(f for t in g1 for f in T[t]["frames"])
        return any(f in s1 for t in g2 for f in T[t]["frames"])

    def end_spot(g, at_end):
        pts = sorted((q for t in g for q in T[t]["spots"]), key=lambda q: q[0])
        if not pts:
            return None
        q = pts[-1] if at_end else pts[0]
        return q

    def side(g):
        s = [T[t]["side"] * len(T[t]["frames"]) for t in g]
        return int(np.sign(sum(s))) if abs(sum(s)) > 0 else 0

    def cost(g1, g2):
        a0, a1 = span(g1)
        b0, b1 = span(g2)
        if a1 < b0:
            first, second, gap = g1, g2, b0 - a1
        elif b1 < a0:
            first, second, gap = g2, g1, a0 - b1
        else:
            return None                                    # on screen together
        if gap / fps > MAX_GAP_S:
            return None
        on1 = any(T[t]["on_court"] for t in g1)
        on2 = any(T[t]["on_court"] for t in g2)
        if on1 != on2:
            return None
        s1, s2 = side(g1), side(g2)
        if s1 and s2 and s1 != s2:
            return None                                    # players don't change sides mid-clip
        e, s = end_spot(first, True), end_spot(second, False)
        dist_cost = 0.5
        if e is not None and s is not None:
            d = float(np.hypot(e[1] - s[1], e[2] - s[2]))
            reach = RUN_MPS * (s[0] - e[0]) / fps + SLACK_M
            if d > reach:
                return None
            dist_cost = d / reach
        la = [look[t] for t in g1 if t in look]
        lb = [look[t] for t in g2 if t in look]
        look_cost = 0.3
        if la and lb:
            look_cost = float(cv2.compareHist(np.mean(la, axis=0), np.mean(lb, axis=0), cv2.HISTCMP_BHATTACHARYYA))
            if look_cost > LOOK_MAX:
                return None
        return look_cost + 0.5 * dist_cost + 0.2 * gap / fps

    while True:
        keys = list(groups)
        best = None
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                c = cost(groups[a], groups[b])
                if c is not None and (best is None or c < best[0]):
                    best = (c, a, b)
        if best is None:
            break
        _, a, b = best
        groups[a] = groups[a] + groups.pop(b)
    out = {}
    for n, (_gid, g) in enumerate(sorted(groups.items(), key=lambda kv: -sum(len(T[t]["frames"]) for t in kv[1]))):
        for t in g:
            out[t] = {"player": n, "side": side(g), "on_court": any(T[t2]["on_court"] for t2 in g)}
    return out
