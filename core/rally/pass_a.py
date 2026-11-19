"""Rally pass A: windows of a full game where play may be, from the 2D ball alone (cheap, every
frame). Pose and the 3D then run only inside them (notes/ANALYSIS_DESIGN.md section 2).

The one clue: in play the ball goes above the net tape (in the picture) every few seconds; between
rallies it is held, bounced, rolled or thrown low. On full_rally_test (6 marked rallies, Nov 19)
in-play seconds had the ball above the tape a median 83% of the time; 90% of dead seconds under 4%.

    active second   the ball seen above the tape line for >= ACTIVE_MIN_S in it
    window          active seconds merged across gaps <= MERGE_GAP_S (a low dig, a near-court
                    ball drawn below the tape), then padded by PAD_BEFORE_S / PAD_AFTER_S
    dropped         a window with the ball above the tape under MIN_HIGH_S in all: a ball thrown
                    back to the server (full_rally_test: 11-19 frames; the shortest marked rally on
                    11 clips: 157 frames)

Recall first: a missed rally is lost for good, extra time only costs pose minutes. Needs the
camera (where the tape is in the picture); without it the caller falls back to "ball seen".
"""
from __future__ import annotations

import numpy as np

ACTIVE_MIN_S = 0.1       # [guess] ball above the tape this long in a second: that second is active
MERGE_GAP_S = 5.0        # [guess] a rally's high balls are never further apart than this
PAD_BEFORE_S = 3.0       # [guess] before the first high ball: the serve toss / the server's bounces
PAD_AFTER_S = 2.0        # [guess] after the last: the ball comes down, lands (your rally-over marks
                         #   were 0.2-0.9 s after the last high ball)
MIN_HIGH_S = 0.5         # [guess] a window needs this much ball above the tape (risk: a serve
                         #   straight into the net might have less)
NET_H_M = 2.43
BALL_R_PX_PAD = 0        # the tape line itself: no slack


def tape_y(cam, x_px: np.ndarray, net_h: float = NET_H_M) -> np.ndarray:
    """The tape's height in the picture (y px) at picture columns x_px: the net line projected,
    interpolated across the picture (the net is a straight line from post to post)."""
    pts = cam.project(np.array([[0.0, y, net_h] for y in (-5.0, 5.0)]))     # just past the posts
    (xa, ya), (xb, yb) = pts[np.argsort(pts[:, 0])]
    t = (np.asarray(x_px, float) - xa) / max(xb - xa, 1e-6)
    return ya + t * (yb - ya)


def windows(pos: np.ndarray, fps: float, cam, net_h: float = NET_H_M) -> list[tuple[int, int]]:
    """pos: (n, 2) ball in the picture per frame (NaN: not seen). Returns [(f0, f1)] frames, inclusive."""
    n = len(pos)
    ok = np.isfinite(pos[:, 0])
    high = np.zeros(n, bool)
    high[ok] = pos[ok, 1] < tape_y(cam, pos[ok, 0], net_h) - BALL_R_PX_PAD
    w = max(1, int(round(fps)))
    need = max(1, int(round(ACTIVE_MIN_S * fps)))
    active = [s for s in range(0, n, w) if high[s:s + w].sum() >= need]
    minh = int(MIN_HIGH_S * fps)
    out: list[list[int]] = []
    gap, pb, pa = int(MERGE_GAP_S * fps), int(PAD_BEFORE_S * fps), int(PAD_AFTER_S * fps)
    for s in active:
        a, b = s, min(n - 1, s + w - 1)
        if out and a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(max(0, a - pb), min(n - 1, b + pa)) for a, b in out if high[a:b + 1].sum() >= minh]


def track_positions(track: dict) -> np.ndarray:
    """track.json (tools.track2d) -> (n, 2) picture positions, NaN where no ball."""
    fr = track["frames"]
    n = max(int(r["f"]) for r in fr) + 1 if fr else 0
    P = np.full((n, 2), np.nan)
    for r in fr:
        if r.get("pos") is not None:
            P[int(r["f"])] = r["pos"]
    return P
