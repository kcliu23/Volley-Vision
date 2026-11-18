"""The serve (FIVB 12, 13.3.4, 14.5): nobody touches it before it crosses, and it can't be blocked
or attacked above the net. Works on (x, z) points, so any stage that knows where the touches were
(the 3D fit now; the rally detector later) can use it."""
from __future__ import annotations

from core.rules.fivb import BLOCK_BELOW_TAPE_M, BLOCK_NEAR_M, FLOOR_Z_M, SERVE_HIT_X_M, SERVE_LOOK, SERVE_START_Z_M


def serve_faults(start: tuple[float, float], joins: list[tuple[float, float]],
                 net_h: float | None) -> list[int]:
    """The touches the serve rules forbid, as indices into joins (empty if the rally doesn't start
    with a serve). start: (x, z) where the rally starts; joins: (x, z) of each touch, in order
    (x: metres from the net, + the far side). The serve: the first of these far back and high.

      - nobody touches the served ball before it crosses the net (FIVB 12: the serve goes over to
        the opponent): a touch on the server's side after the hit is a flight split in two. The
        server's own hit after a toss (still far back) is the serve, not a fault;
      - the serve can't be blocked or attacked above the net (FIVB 14.5, 13.3.4): the first touch
        on the receiving side isn't at the net, up at the tape.
    """
    pts = [start] + list(joins)
    # the serve: the first point far back and high among the rally's first SERVE_LOOK points (a
    # rally often starts before it: the ball held or bounced low by the server)
    s = next((i for i, (x, z) in enumerate(pts[:SERVE_LOOK])
              if abs(x) >= SERVE_HIT_X_M and z >= SERVE_START_Z_M), None)
    if s is None:
        return []
    side = 1 if pts[s][0] > 0 else -1
    bad = []
    for j, (x, z) in enumerate(joins):
        if j + 1 <= s:                                   # before the serve (or the serve itself)
            continue
        if z < FLOOR_Z_M:
            break
        if (1 if x > 0 else -1) == side:
            if not (side * x >= SERVE_HIT_X_M and not bad):
                bad.append(j)
            continue
        if net_h is not None and abs(x) < BLOCK_NEAR_M and z > net_h - BLOCK_BELOW_TAPE_M:
            bad.append(j)
        break
    return bad
