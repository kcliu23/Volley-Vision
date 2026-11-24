"""One id per player for a whole clip: join the tracker's broken tracks (tracklets) into players.

The pose tracker (tools.pose) loses a player whenever they cross someone or get hidden, and starts
a new id: a 12-player clip comes out with 30-50 ids. Three things tie the pieces back together:
  side       a player stays on their side of the net for the whole rally: the median of where
             their feet are on the court (calibrated camera) says which side (sign of x)
  place      a new piece must start near where an old piece ended, within running distance for
             the time between them (on the court, in meters, not in pixels)
  look       the torso's colours (jersey, skin): a colour histogram over several frames, averaged
             with a person re-identification model's look vector when it is installed
             (core.people.reid_model; on holdout1-8: 269 -> 235 key presses to number every
             player, wrong joins 0-13% -> 0-8%)
Pieces are joined greedily, best match first, never two pieces that are on screen at the same
time. Off-court people (feet outside the court + 1.5 m: the bench, referees, spectators) are kept
apart.

    ids = build_ids(pose_frames, video, cam, fps)
    ids = build_ids_cached(pose_frames, pose_key, video, cam_path, fps, cache)   # same, saved for reuse
    ids["tracklet id"] -> {"player": n, "side": -1 near / +1 far / 0, "on_court": bool}
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict

import numpy as np

from core.scene3d.params3d import ANKLE_H

RUN_MPS = 6.0          # fastest a player covers ground between two pieces
SLACK_M = 1.5          # plus this (feet jitter, a missed step)
MAX_GAP_S = 4.0        # pieces further apart in time aren't joined
LOOK_MAX = 0.45        # look distance above this: not the same person (colour Bhattacharyya, or
                       # its average with the re-ID model's cosine distance when the model is there)
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
    embs = defaultdict(list)
    from core.people.reid_model import load_embedder
    embed = load_embedder()                                # None: not installed, colours only
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)              # phone clips: stored sideways + a rotation flag
    f = 0
    last = max(want) if want else -1
    while f <= last:
        ok, img = cap.read()
        if not ok:
            break
        H_, W_ = img.shape[:2]                             # boxes clipped to the picture; empty ones
        W = [(tid, (x1, y1, x2, y2)) for tid, (x1, y1, x2, y2) in    # (a far-zoom box at the edge) skipped:
             ((tid, (max(0, b[0]), max(0, b[1]), min(W_, b[2]), min(H_, b[3]))) for tid, b in want.get(f, []))
             if x2 - x1 >= 2 and y2 - y1 >= 2]                         # the re-ID model rejects the batch
        if W and embed is not None:
            for (tid, _), e in zip(W, embed(img, np.array([b for _, b in W], float))):
                embs[tid].append(e)
        for tid, (x1, y1, x2, y2) in W:
            crop = img[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
            if crop.size < 30:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
            hists[tid].append(cv2.normalize(h, h).flatten())
        f += 1
    looks = {tid: np.mean(h, axis=0).astype(np.float32) for tid, h in hists.items() if h}
    emb = {tid: (lambda m: m / np.linalg.norm(m))(np.mean(e, axis=0)) for tid, e in embs.items() if e}
    return looks, emb


def build_ids(pose_frames: dict, video: str, cam, fps: float):
    import cv2
    T = _tracklets(pose_frames, cam)
    look, emb = _looks(T, video)
    # players: lists of tracklet ids, joined greedily by the best (lowest cost) valid pair
    groups = {tid: [tid] for tid in T}
    pos = {tid: i for i, tid in enumerate(T)}             # ties go to the earlier pair, as in a full rescan

    def summary(g):
        """What cost() needs from a group, worked out once per merge instead of once per pair."""
        pts = sorted((q for t in g for q in T[t]["spots"]), key=lambda q: q[0])
        s = sum(T[t]["side"] * len(T[t]["frames"]) for t in g)
        la = [look[t] for t in g if t in look]
        ea = [emb[t] for t in g if t in emb]
        return {"f0": min(T[t]["f0"] for t in g), "f1": max(T[t]["f1"] for t in g),
                "on": any(T[t]["on_court"] for t in g), "side": int(np.sign(s)) if s else 0,
                "first": pts[0] if pts else None, "last": pts[-1] if pts else None,
                "look": np.mean(la, axis=0) if la else None, "emb": np.mean(ea, axis=0) if ea else None}

    def cost(A, B):
        if A["f1"] < B["f0"]:
            first, second, gap = A, B, B["f0"] - A["f1"]
        elif B["f1"] < A["f0"]:
            first, second, gap = B, A, A["f0"] - B["f1"]
        else:
            return None                                    # on screen together
        if gap / fps > MAX_GAP_S:
            return None
        if A["on"] != B["on"]:
            return None
        if A["side"] and B["side"] and A["side"] != B["side"]:
            return None                                    # players don't change sides mid-clip
        e, s = first["last"], second["first"]
        dist_cost = 0.5
        if e is not None and s is not None:
            d = float(np.hypot(e[1] - s[1], e[2] - s[2]))
            reach = RUN_MPS * (s[0] - e[0]) / fps + SLACK_M
            if d > reach:
                return None
            dist_cost = d / reach
        look_cost = 0.3
        if A["look"] is not None and B["look"] is not None:
            look_cost = float(cv2.compareHist(A["look"], B["look"], cv2.HISTCMP_BHATTACHARYYA))
            if A["emb"] is not None and B["emb"] is not None:
                ma, mb = A["emb"], B["emb"]
                look_cost = 0.5 * look_cost + 0.5 * (1.0 - float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb))))
            if not look_cost <= LOOK_MAX:                   # NaN (an empty histogram) is no match either
                return None
        return look_cost + 0.5 * dist_cost + 0.2 * gap / fps

    # greedy, best pair first: a heap of pair costs, and after a merge only the merged group's pairs
    # are rescored (a full rescan per merge took hours on a 2,900-piece game)
    S = {t: summary(g) for t, g in groups.items()}
    ver = dict.fromkeys(groups, 0)
    heap = []

    def push(a, b):
        if pos[a] > pos[b]:
            a, b = b, a
        c = cost(S[a], S[b])
        if c is not None:
            heapq.heappush(heap, (c, pos[a], pos[b], a, b, ver[a], ver[b]))

    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            push(a, b)
    while heap:
        _, _, _, a, b, va, vb = heapq.heappop(heap)
        if b not in groups or a not in groups or ver[a] != va or ver[b] != vb:
            continue                                       # stale: a side was merged since
        groups[a] = groups[a] + groups.pop(b)
        ver[a] += 1
        S[a] = summary(groups[a])
        for k in groups:
            if k != a:
                push(a, k)
    out = {}
    for n, (gid, g) in enumerate(sorted(groups.items(), key=lambda kv: -sum(len(T[t]["frames"]) for t in kv[1]))):
        for t in g:
            out[t] = {"player": n, "side": S[gid]["side"], "on_court": any(T[t2]["on_court"] for t2 in g)}
    return out


def build_ids_cached(pose_frames: dict, pose_key: str, video: str, cam_path, fps: float, cache):
    """build_ids, saved in `cache` and reused while the pose and camera files stay the same.
    On a full game the looks alone (every frame read) take many minutes."""
    from core.common.safe_io import file_key, save_json
    from core.geometry.camera import Camera
    key = f"v1:{pose_key}:{file_key(cam_path)}"           # bump v1 when build_ids changes its answer
    try:
        old = json.load(open(cache))
    except (OSError, ValueError):                         # none yet, or damaged: redo it
        old = {}
    if old.get("key") == key:
        return {int(t): v for t, v in old["ids"].items()}
    print("[reid] re-ID guesses (first time only; slow on a long clip) ...", flush=True)
    ids = build_ids(pose_frames, video, Camera.load(cam_path), fps)
    save_json(cache, {"key": key, "ids": ids})
    return ids
