"""Volleyball's rules on a plain list of touches: who (side, player) touched it, in order. No 3D and
no positions needed, so the 2D touch detector, the rally detector, the event builder and
tools.check_labels can all use it, on whatever they know (a side or player they don't know is None,
and the rules needing it are skipped there).

  - four_touches  a team hits the ball at most 3 times before it goes over (FIVB 9.1); a block
                  doesn't count (14.4.1)
  - double_touch  nobody touches it twice in a row (9.2.3), except after their own block (14.2)
  - serve_own     the serving team doesn't touch its own serve (12: it goes over to the opponent)
  - serve_block   a serve can't be blocked (14.5)

FIVB Official Volleyball Rules 2025-2028.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.rules.fivb import MAX_TOUCHES


@dataclass
class Contact:
    """One touch. side: -1 near, +1 far (the toucher's team; a block: the blocking team)."""
    frame: int
    side: int | None = None
    player: int | None = None
    block: bool = False
    serve: bool = False       # this touch is the serve (only the rally's first one can be)


@dataclass
class Fault:
    index: int                # into the contact list: the touch that breaks the rule
    rule: str                 # four_touches | double_touch | serve_own | serve_block
    why: str


def faults(contacts: list[Contact]) -> list[Fault]:
    """Every rule broken in one rally's touches, in order."""
    out: list[Fault] = []
    team, n = None, 0                     # who has the ball, their hits so far (unknown: None)
    for i, c in enumerate(contacts):
        prev = contacts[i - 1] if i else None
        if prev is not None and prev.serve:
            if c.block:
                out.append(Fault(i, "serve_block", f"frame {c.frame}: the serve was blocked"))
            elif c.side is not None and c.side == prev.side:
                out.append(Fault(i, "serve_own", f"frame {c.frame}: the serving team touched its own serve"))
        if (prev is not None and c.player is not None and c.player == prev.player
                and not prev.block):
            out.append(Fault(i, "double_touch", f"frame {c.frame}: #{c.player} touched it twice in a row"))
        if c.side is None:
            team, n = None, 0
            continue
        if c.block:
            team, n = c.side, 0           # the blocking team still has 3
            continue
        if c.side == team:
            n += 1
            if n == MAX_TOUCHES + 1:
                out.append(Fault(i, "four_touches", f"frame {c.frame}: the team's 4th touch"))
        else:
            team, n = c.side, 1
    return out
