"""Volleyball court geometry in meters: the known points the camera is calibrated from.

World frame: origin at the centre of the court on the floor, under the net.
  x  along the court (end line to end line, 18 m): -9 .. +9, the net at x = 0
  y  across the court (sideline to sideline, 9 m): -4.5 .. +4.5
  z  up from the floor
+x = the FAR half (beyond the net, as seen from the camera), +y = the LEFT sideline as seen
from the camera, so x, y, z form a right-handed frame (z up). (+y = right would be a mirror
image: the camera would end up under the floor.)

Lines are measured to their outer edges in the rules; at phone resolution the 5 cm line width is
2-3 px, so clicking the middle of a line crossing is close enough.
"""
from __future__ import annotations

LENGTH, WIDTH = 18.0, 9.0
ATTACK = 3.0                       # attack line, 3 m from the centre line
NET_HEIGHT = {"men": 2.43, "women": 2.24}
NET_BAND = 1.0                     # the net is 1 m tall (top tape down to the bottom band)
ANTENNA = 0.8                      # antenna sticks up 80 cm above the net tape


def keypoints(net_height: float = NET_HEIGHT["men"]) -> dict:
    """{name: (x, y, z)} of every point that can be clicked. Names say near / far half and
    left / right sideline as seen from the camera."""
    hx, hy, h = LENGTH / 2, WIDTH / 2, net_height
    pts = {}
    for side, y in (("left", hy), ("right", -hy)):
        pts[f"far end corner {side}"] = (hx, y, 0.0)
        pts[f"far attack line {side}"] = (ATTACK, y, 0.0)
        pts[f"centre line {side}"] = (0.0, y, 0.0)
        pts[f"near attack line {side}"] = (-ATTACK, y, 0.0)
        pts[f"near end corner {side}"] = (-hx, y, 0.0)
        # the antenna stands on the sideline: these are where it crosses the net
        pts[f"net top {side}"] = (0.0, y, h)
        pts[f"net bottom {side}"] = (0.0, y, h - NET_BAND)
        pts[f"antenna top {side}"] = (0.0, y, h + ANTENNA)
    return pts


ORDER = [f"{k} {s}" for k in ("far end corner", "far attack line", "centre line", "near attack line",
                             "near end corner", "net bottom", "net top", "antenna top")
         for s in ("left", "right")]


def lines(net_height: float = NET_HEIGHT["men"]) -> list:
    """3D segments to draw the court and net back onto the picture: [((x,y,z), (x,y,z)), ...]."""
    hx, hy, h = LENGTH / 2, WIDTH / 2, net_height
    out = [((-hx, -hy, 0), (hx, -hy, 0)), ((-hx, hy, 0), (hx, hy, 0)),          # sidelines
           ((-hx, -hy, 0), (-hx, hy, 0)), ((hx, -hy, 0), (hx, hy, 0)),          # end lines
           ((0, -hy, 0), (0, hy, 0)),                                            # centre line
           ((-ATTACK, -hy, 0), (-ATTACK, hy, 0)), ((ATTACK, -hy, 0), (ATTACK, hy, 0)),
           ((0, -hy, h), (0, hy, h)), ((0, -hy, h - NET_BAND), (0, hy, h - NET_BAND)),   # net
           ((0, -hy, h - NET_BAND), (0, -hy, h + ANTENNA)), ((0, hy, h - NET_BAND), (0, hy, h + ANTENNA))]
    return out
