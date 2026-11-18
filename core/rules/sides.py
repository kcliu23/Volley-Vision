"""Volleyball's rules for one rally: which side of the net each touch was on, decided for the whole
rally at once.

Deciding each touch on its own (where the 3D put it, who stood there) breaks down at the net: from
one camera, depth there is weak, players of both sides are at the ball, and the 3D may put a dig on
the wrong side, or send the ball through the net. A rally has to make sense as a whole:

  - a team touches the ball at most 3 times before it goes over (a block doesn't count: the
    blocking team still has 3);
  - the ball changes side only over the net (above the tape); a flight that never gets that high
    can't change side;
  - a block is at the net, up at the tape, right after the other team's touch;
  - the serve (FIVB 12, 14.5) goes over to the other team, and can't be blocked;
  - (in the 3D fit, rally3d) a ball that changes side mid-rally crosses inside the antennas
    (FIVB 10.1.1), and the serve is hit from behind the end line (FIVB 1.4.2, 12).
FIVB Official Volleyball Rules 2025-2028.

Every touch gets a cost for each side (from where a first 3D fit puts it, and the players at the
ball), every flight a cost for changing side (how far below the tape it stays). The cheapest
sequence that keeps the rules wins (dynamic programming). A touch's side is "sure" when forcing it
to the other side costs at least MARGIN more; only sure sides are passed on to the 3D fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.rules.fivb import BLOCK_BELOW_TAPE_M, BLOCK_NEAR_M, MAX_TOUCHES

P_FLOOR = 0.02          # no single clue is ever more than 98% sure
PLAYER_P = 0.9          # one side's players at the ball (numbers or feet): 90% that side
NAMED_P = 0.999         # you said who touched it
SIDE_SIGMA_MIN_M = 0.25  # the first fit's position across the net is never trusted closer than this
BLOCK_COST = 1.0        # a block is a little less likely than an ordinary touch (nats)
CROSS_SIGMA_M = 0.15    # a flight changing side while staying this far under the tape: 1 nat
CROSS_MAX = 20.0
MARGIN = 2.0            # nats: a side is sure when the other side costs this much more


@dataclass
class Touch:
    """What is known about one touch (or the start of the rally: first=True)."""
    x: float                  # metres from the net (+ the far side), from a fit without sides
    sx: float                 # its uncertainty
    z: float                  # height
    player: int | None = None  # -1 / +1: the side of the players at the ball (None: unknown/both)
    named: bool = False       # player is your who-touched-it
    first: bool = False
    serve: bool = False       # (first only) the rally starts with a serve
    p_side: dict = field(default_factory=dict)


@dataclass
class Choice:
    side: int                 # -1 near, +1 far (the toucher's team; a block: the blocking team)
    block: bool
    margin: float             # nats to the best sequence with this touch on the other side
    sure: bool


def _phi(v: float) -> float:
    return 0.5 * (1.0 + math.erf(v / math.sqrt(2.0)))


def side_cost(t: Touch, s: int) -> float:
    p = _phi(s * t.x / max(t.sx, SIDE_SIGMA_MIN_M))
    c = -math.log(min(1 - P_FLOOR, max(P_FLOOR, p)))
    if t.player in (-1, 1):
        q = NAMED_P if t.named else PLAYER_P
        c += -math.log(q if s == t.player else 1 - q)
    return c


def can_block(t: Touch, net_h: float | None) -> bool:
    return (not t.first and abs(t.x) < BLOCK_NEAR_M and net_h is not None
            and t.z > net_h - BLOCK_BELOW_TAPE_M)


def cross_cost(zmax: float, net_h: float | None, ball_r: float = 0.105) -> float:
    """Changing side in a flight whose highest point is zmax."""
    if net_h is None:
        return 0.0
    short = net_h + ball_r - zmax
    return 0.0 if short <= 0 else min(CROSS_MAX, (short / CROSS_SIGMA_M) ** 2)


def _decode(touches: list[Touch], zmax: list[float], net_h, force: tuple[int, int] | None = None):
    """Cheapest legal sequence. zmax[i]: the flight from touch i to touch i+1. force: (i, side).
    Returns (cost, [(side, block)])."""
    INF = float("inf")
    # state: (team in possession, its touches so far, last touch was a block)
    best: dict = {}
    for s in (-1, 1):
        t0 = touches[0]
        if force and force[0] == 0 and force[1] != s:
            continue
        best[(s, 1, False)] = (side_cost(t0, s), [(s, False)])
    for i in range(1, len(touches)):
        t = touches[i]
        nxt: dict = {}
        for (team, n, was_block), (c, path) in best.items():
            for s in (-1, 1):
                if force and force[0] == i and force[1] != s:
                    continue
                for blk in ((False, True) if can_block(t, net_h) else (False,)):
                    if i == 1 and touches[0].serve and (blk or s == team):
                        continue                   # the serve goes over and can't be blocked
                    if blk:
                        # a block answers the other team's touch, at the net
                        if was_block or s == team:
                            continue
                        state, cc = (s, 0, True), 0.0
                    elif was_block:
                        # after a block either team plays it; the blocking team still has 3
                        state, cc = (s, 1, False), 0.0
                    elif s == team:
                        if n + 1 > MAX_TOUCHES:
                            continue
                        state, cc = (s, n + 1, False), 0.0
                    else:
                        state, cc = (s, 1, False), cross_cost(zmax[i - 1], net_h)
                    total = c + cc + side_cost(t, s) + (BLOCK_COST if blk else 0.0)
                    if total < nxt.get(state, (INF,))[0]:
                        nxt[state] = (total, path + [(s, blk)])
        best = nxt
        if not best:
            return INF, []
    return min(best.values(), key=lambda v: v[0]) if best else (INF, [])


def decide(touches: list[Touch], zmax: list[float], net_h: float | None) -> list[Choice]:
    """The side of every touch (touches[0]: the start of the rally) under the rules."""
    if not touches:
        return []
    cost, path = _decode(touches, zmax, net_h)
    if not path:
        return []
    out = []
    for i, (s, blk) in enumerate(path):
        alt, _ = _decode(touches, zmax, net_h, force=(i, -s))
        m = alt - cost
        out.append(Choice(s, blk, m, m >= MARGIN))
    return out
