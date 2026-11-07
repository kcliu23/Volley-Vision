"""Players on the court in meters, from person boxes + the calibrated camera, jumps included.

Standing, a player's feet are on the floor: the bottom-middle of their box hits the floor at their
court position (core.camera.Camera.floor_point). Jumping, the feet leave the floor and that
point lands too far away, so jumps are found and handled first:
  jump       candidates: the feet (box bottom) rise in the picture well above where they stood just
             before and after, for 0.12-1.0 s. Only the feet count: raised arms (setters, blockers)
             can't fake a jump. Then physics decides: a jump is a free flight of the body, so the
             feet height over time must be a gravity parabola (fixed g; fitted peak height and
             time), 0.1-1.0 m high. Walking toward the camera, running strides and occlusions
             (a sudden step when feet go behind someone) fail that test.
  in the air the spot under the player = the straight line from the takeoff spot to the landing
             spot (a jump drifts at most ~1 m); feet height = where the feet's line of sight
             crosses the vertical through that spot
  height     each player's standing height from grounded frames (box top at their spot): a
             ruler for later (size -> depth) and for reach (touch height)
Track pieces the tracker split (id changes) are stitched first. Box jitter is smoothed (median
over 5 frames). Boxes cut by the picture's bottom edge have no
feet: no position there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.params3d import (JUMP_FIT_M, JUMP_H, JUMP_MAX_S, JUMP_MIN_S, LIFT_FRAC, REACH_HEIGHT_X,  # noqa: F401
                           REACH_RADIUS_M)
G = 9.81


@dataclass
class Player:
    id: int
    frames: np.ndarray            # frame numbers
    box: np.ndarray               # (n, 4) x1, y1, x2, y2
    ground: np.ndarray            # (n, 2) court position under the player (m); NaN if unknown
    airborne: np.ndarray          # (n,) bool
    feet_z: np.ndarray            # (n,) feet height above the floor (m): 0 standing
    height_m: float               # standing height (box top at their spot; includes hair, posture)
    jumps: list = field(default_factory=list)   # [(takeoff frame, landing frame, peak feet height m)]

    def at(self, f: int):
        i = np.searchsorted(self.frames, f)
        if i < len(self.frames) and self.frames[i] == f:
            return i
        return None


def _medfilt(x: np.ndarray, k: int = 5) -> np.ndarray:
    out = x.copy()
    h = k // 2
    for i in range(len(x)):
        w = x[max(0, i - h):i + h + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def _height_at(cam, spot_xy, v_top: float) -> float:
    """Height (m) of the point above a floor spot that projects to picture row v_top."""
    lo, hi = 0.0, 3.5
    for _ in range(40):
        z = (lo + hi) / 2
        v = cam.project([[spot_xy[0], spot_xy[1], z]])[0][1]
        if v > v_top:
            lo = z
        else:
            hi = z
    return (lo + hi) / 2


def _feet_height(cam, spot_xy, feet_uv) -> float:
    """Height where the feet's line of sight passes over the floor spot (closest approach to the
    vertical line through it)."""
    c, d = cam.rays([feet_uv])
    d = d[0]
    # vertical line L(z) = (sx, sy, z); ray R(s) = c + s d; closest points: solve in the xy plane
    dxy = d[:2]
    s = float(np.dot(np.asarray(spot_xy) - c[:2], dxy) / max(np.dot(dxy, dxy), 1e-9))
    return float(max(0.0, c[2] + s * d[2]))


def _is_jump(t, z):
    """Peak feet height if (t, z) is a jump (a gravity parabola of plausible height), else None.
    (Box size is no test: players tuck their knees in the air, so the box shrinks in real jumps;
    occlusions are told apart by their sudden step, which no parabola fits.)"""
    ok = np.isfinite(z)
    if ok.sum() < 4:
        return None
    t, z = t[ok], z[ok]
    best = None
    for tp in np.linspace(t[0] - 0.1, t[-1] + 0.1, 41):  # peak time; peak height by least squares
        drop = 0.5 * G * (t - tp) ** 2
        h = float(np.mean(z + drop))
        r = float(np.sqrt(np.mean((z - (h - drop)) ** 2)))
        if best is None or r < best[0]:
            best = (r, h, tp)
    r, h, tp = best
    if r > JUMP_FIT_M or not (JUMP_H[0] <= h <= JUMP_H[1]) or not (t[0] - 0.05 <= tp <= t[-1] + 0.05):
        return None
    return h


def stitch(tracks: dict, fps: float, max_gap_s: float = 0.5) -> dict:
    """Join track pieces the tracker split (an id change mid-jump, after an occlusion): a track
    that starts within max_gap_s after another ended, with its first box near that one's last
    box (centres within one box width, similar size), is the same player. Greedy, closest first."""
    gap = max(1, int(round(max_gap_s * fps)))
    merged = {k: list(v) for k, v in tracks.items()}
    while True:
        best = None
        ends = {k: merged[k][-1] for k in merged}
        starts = {k: merged[k][0] for k in merged}
        for a in merged:
            fa, ax1, ay1, ax2, ay2 = ends[a]
            wa, ha = ax2 - ax1, ay2 - ay1
            for b in merged:
                if b == a:
                    continue
                fb, bx1, by1, bx2, by2 = starts[b]
                if not (0 < fb - fa <= gap):
                    continue
                d = np.hypot((ax1 + ax2 - bx1 - bx2) / 2, (ay1 + ay2 - by1 - by2) / 2)
                hb = by2 - by1
                if d <= max(wa, bx2 - bx1) and 0.6 < hb / max(ha, 1) < 1.6:
                    score = d / max(wa, 1) + (fb - fa) / gap
                    if best is None or score < best[0]:
                        best = (score, a, b)
        if best is None:
            return merged
        _, a, b = best
        merged[a] = merged[a] + merged.pop(b)


def build(players_json: dict, cam, fps: float, min_frames: int = 15) -> list[Player]:
    H = cam.height
    tracks: dict = {}
    for f, row in enumerate(players_json["frames"]):
        for tid, x1, y1, x2, y2, conf in row:
            if tid < 0:
                continue
            tracks.setdefault(tid, []).append((f, x1, y1, x2, y2))
    tracks = stitch(tracks, fps)
    out = []
    for tid, rows in tracks.items():
        if len(rows) < min_frames:
            continue
        a = np.array(rows, float)
        fr = a[:, 0].astype(int)
        box = a[:, 1:5]
        bh = box[:, 3] - box[:, 1]
        cut = box[:, 3] >= H - 4                         # feet out of the picture
        feet_v = np.where(cut, np.nan, box[:, 3])
        feet_u = (box[:, 0] + box[:, 2]) / 2
        feet_v_s = _medfilt(feet_v)
        # jump candidates, one per moment the feet are highest in the picture: takeoff = feet
        # lowest (largest v) in the 0.6 s before, landing = lowest in the 0.6 s after; lift is
        # measured against the straight line between them (a player who keeps moving toward the
        # camera after landing doesn't stretch the jump)
        tall = np.nanmedian(bh)
        airborne = np.zeros(len(fr), bool)
        win = max(2, int(round(0.6 * fps)))
        cands = []
        v = feet_v_s
        for m in range(1, len(fr) - 1):
            if not np.isfinite(v[m]) or not (v[m] <= np.nanmin(v[max(0, m - 3):m + 4])):
                continue                                   # not a local top
            lo, hi = max(0, m - win), min(len(fr), m + win + 1)
            pre, post = v[lo:m], v[m + 1:hi]
            has_pre, has_post = np.isfinite(pre).any(), np.isfinite(post).any()
            if not (has_pre or has_post):
                continue
            # the track may start or end mid-jump (the tracker changed the player's id): then
            # the one known end is the baseline
            a_ = lo + int(np.nanargmax(pre)) if has_pre else None
            b_ = m + 1 + int(np.nanargmax(post)) if has_post else None
            if a_ is not None and b_ is not None:
                base = v[a_] + (v[b_] - v[a_]) * (m - a_) / max(b_ - a_, 1)
            else:
                base = v[a_] if a_ is not None else v[b_]
                a_ = a_ if a_ is not None else max(0, m - 1)
                b_ = b_ if b_ is not None else min(len(fr) - 1, m + 1)
                if (fr[b_] - fr[a_]) / fps < JUMP_MIN_S:        # one side only: judge by the lift
                    a_, b_ = (lo if not has_pre else a_), (hi - 1 if not has_post else b_)
            dur = (fr[b_] - fr[a_]) / fps
            if base - v[m] > LIFT_FRAC * tall and JUMP_MIN_S <= dur <= JUMP_MAX_S:
                cands.append((base - v[m], a_, b_))
        for _, a_, b_ in sorted(cands, reverse=True):      # biggest lifts first, no overlaps
            if not airborne[a_:b_ + 1].any():
                airborne[a_ + 1:b_] = True
        # court position from the feet where grounded
        ground = np.full((len(fr), 2), np.nan)
        ok = ~airborne & np.isfinite(feet_v_s)          # (candidate jump frames are filled below)
        if ok.any():
            g = cam.floor_point(np.c_[feet_u[ok], feet_v_s[ok]])
            ground[ok] = g[:, :2]
        ground[:, 0] = _medfilt(ground[:, 0]); ground[:, 1] = _medfilt(ground[:, 1])
        # in the air: the line from takeoff to landing
        feet_z = np.zeros(len(fr))
        jumps = []
        i = 0
        while i < len(fr):
            if not airborne[i]:
                i += 1; continue
            j = i
            while j + 1 < len(fr) and airborne[j + 1]:
                j += 1
            pre = i - 1 if i > 0 and np.isfinite(ground[i - 1, 0]) else None
            post = j + 1 if j + 1 < len(fr) and np.isfinite(ground[j + 1, 0]) else None
            run = np.arange(i, j + 1)
            if pre is None and post is None:
                airborne[run] = False
                i = j + 1; continue
            # takeoff and landing spots; with only one (the tracker changed the player's id
            # mid-jump) the jump is taken as straight up from it
            A = ground[pre] if pre is not None else ground[post]
            B = ground[post] if post is not None else ground[pre]
            gl = np.array([A + (k - i + 1) / (j - i + 2) * (B - A) for k in run])
            z = np.array([_feet_height(cam, gl[n], (feet_u[k], feet_v[k])) if np.isfinite(feet_v[k]) else np.nan
                          for n, k in enumerate(run)])
            ok_jump = _is_jump(fr[run] / fps, z)
            if ok_jump is None:
                airborne[run] = False                      # not a jump: back to the feet
                i = j + 1; continue
            ground[run] = gl
            feet_z[run] = np.nan_to_num(z)
            jumps.append((int(fr[i]), int(fr[j]), round(ok_jump, 2)))
            i = j + 1
        # frames that turned out not to be jumps: court position from the feet again
        redo = ~airborne & ~np.isfinite(ground[:, 0]) & np.isfinite(feet_v_s)
        if redo.any():
            ground[redo] = cam.floor_point(np.c_[feet_u[redo], feet_v_s[redo]])[:, :2]
        # standing height: grounded frames with the whole body in the picture
        hs = []
        for k in np.where(ok & (box[:, 1] > 4))[0][::3]:
            if np.isfinite(ground[k, 0]):
                hs.append(_height_at(cam, ground[k], box[k, 1]))
        height = float(np.median(hs)) if hs else float("nan")
        out.append(Player(int(tid), fr, box, ground, airborne, feet_z, height, jumps))
    return out


def reach(players: list[Player], fps: float):
    """near(t, xyz): metres between the ball and the closest player's reach (a vertical cylinder
    over their court spot, radius REACH_RADIUS_M, from the floor to feet height + REACH_HEIGHT_X x
    standing height: standing reach plus any jump), or None where no player is known then."""
    by_f = {}
    for p in players:
        h = p.height_m if np.isfinite(p.height_m) else 1.85
        for k, f in enumerate(p.frames):
            if np.isfinite(p.ground[k, 0]):
                by_f.setdefault(int(f), []).append((p.ground[k, 0], p.ground[k, 1], p.feet_z[k] + REACH_HEIGHT_X * h))
    by_f = {f: np.array(v) for f, v in by_f.items()}

    def near(t, x):
        f = int(round(t * fps))
        P = next((by_f[g] for g in (f, f - 1, f + 1, f - 2, f + 2, f - 3, f + 3) if g in by_f), None)
        if P is None:
            return None
        dxy = np.maximum(0.0, np.hypot(P[:, 0] - x[0], P[:, 1] - x[1]) - REACH_RADIUS_M)
        dz = np.maximum(0.0, x[2] - P[:, 2])
        return float(np.min(np.hypot(dxy, dz)))
    return near


# --- who touched the ball (from poses: tools.pose -> output/<clip>/pose.json) --------------------
# COCO body points: 0 nose, 1-2 eyes, 5-6 shoulders (left, right), 9-10 wrists, 15-16 ankles

def _facing(k):
    """+1 facing the camera, -1 facing away, 0 unsure. Two cues: the face is visible (nose and
    eyes found), and the left shoulder is on the picture's right (mirror image when facing us)."""
    face = (k[0][2] + 0.5 * (k[1][2] + k[2][2])) / 2 - 0.45          # > 0: face visible
    sw = abs(k[5][0] - k[6][0]) + 1e-6
    sh = (k[5][0] - k[6][0]) / max(sw, 15.0) if min(k[5][2], k[6][2]) > 0.3 else 0.0
    s = 1.6 * face + 0.6 * float(np.clip(sh, -1, 1))
    return 1 if s > 0.35 else -1 if s < -0.35 else 0


def touch_evidence(pose_frames: dict, f: int, ball_uv, cam, ball_px: float, near_px: float = 2.5,
                   before_s: float = 0.6, fps: float = 60.0):
    """People whose wrist is at the ball around frame f (within near_px ball widths, +-2 frames).
    For each: side of the net (-1 near the camera, +1 far, 0 unsure) from which way they face and,
    when the feet are clearly seen, from where they stood before the touch (the lowest ankles in
    the before_s seconds before: the takeoff spot, so a jump doesn't move them); and that spot
    on the floor (None when the ankles aren't trustworthy). Returns [{id, side, spot}]."""
    out = {}
    u, v = float(ball_uv[0]), float(ball_uv[1])
    for g in range(f - 2, f + 3):
        for p in pose_frames.get(str(g), pose_frames.get(g, [])):
            k = p["kp"]
            d = min(np.hypot(k[w][0] - u, k[w][1] - v) for w in (9, 10) if k[w][2] > 0.3) \
                if any(k[w][2] > 0.3 for w in (9, 10)) else np.inf
            if d <= near_px * ball_px and (p["id"] not in out or d < out[p["id"]]["d"]):
                out[p["id"]] = {"id": p["id"], "d": float(d), "kp": k}
    res = []
    win = int(round(before_s * fps))
    for pid, c in out.items():
        sides = []
        for g in range(f - win, f + 1, 3):                  # facing over the approach, not one frame
            for p in pose_frames.get(str(g), pose_frames.get(g, [])):
                if p["id"] == pid:
                    sides.append(_facing(p["kp"]))
        face_side = int(np.sign(sum(sides))) if sides and abs(sum(sides)) >= 0.5 * len(sides) else 0
        # takeoff spot: the lowest ankles (largest v) before the touch, when both are clearly seen
        best = None
        for g in range(f - win, f + 1):
            for p in pose_frames.get(str(g), pose_frames.get(g, [])):
                if p["id"] == pid:
                    k = p["kp"]
                    if min(k[15][2], k[16][2]) > 0.6:
                        a = ((k[15][0] + k[16][0]) / 2, (k[15][1] + k[16][1]) / 2)
                        if best is None or a[1] > best[1]:
                            best = a
        spot = None
        side = face_side                                    # facing the camera = far side (+x)
        if best is not None:
            g_ = cam.floor_point([best])[0][:2]
            feet_side = int(np.sign(g_[0])) if abs(g_[0]) > 0.4 else 0
            if side == 0 or feet_side == 0 or feet_side == side:   # feet agree with the facing
                spot = (float(g_[0]), float(g_[1]))
                side = side or feet_side
        res.append({"id": pid, "side": side, "spot": spot, "wrist_px": round(c["d"], 1)})
    return res
