"""Other balls in the picture (not the game ball): spare balls, a ball resting on the sideline.

The YOLO training labels (volleyball_combined) mark EVERY volleyball in the picture; the
linker then picks the game ball. Checked clips (tools.review) only mark the game ball, so
the other balls come from here: YOLO's confident boxes away from the game ball, which you
can correct in the review tool (removed / added balls are saved per frame as "others").
"""
from __future__ import annotations

import math

CONF = 0.5          # only confident YOLO boxes count as another ball
AWAY_BW = 1.5       # ... and only farther than this (ball-widths) from the game ball


def auto(frame_boxes, game_xy, ball_px: float, conf: float = CONF, away_bw: float = AWAY_BW) -> list[list[float]]:
    """[[cx, cy, w, h], ...] in video pixels. frame_boxes: YOLO boxes [cx, cy, w, h, conf] of
    one frame. game_xy: [x, y] of the game ball (labelled or proposed), or None."""
    out = []
    for b in frame_boxes or ():
        if b[4] < conf:
            continue
        if game_xy is not None and math.hypot(b[0] - game_xy[0], b[1] - game_xy[1]) <= away_bw * ball_px:
            continue
        if any(math.hypot(b[0] - o[0], b[1] - o[1]) <= 0.5 * ball_px for o in out):
            continue                              # duplicate box on the same ball
        out.append([round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)])
    return out
