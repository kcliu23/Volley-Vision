"""Which labelled frames are the ball IN PLAY: from a serve until the ball is dead.

The analyses use the ball in play. Before a serve the server holds or bounces the ball, and after
a rally it's carried around: the tracker rightly distrusts faint, still things there, and those
frames say little about tracking rallies. So the scores also report in-play frames on their own
(decided 2026-11-09).

Defined by your marks in tools.review (m + type), not by a rule on how the ball moves:
    serve   (m 1)   the rally starts here
    dead    (m 9)   the rally is over (ball on the floor / out / whistle)
In play = from each serve to the next "dead" mark (or the next serve, or the clip's end). A clip
with no serve marked has no in-play frames defined: in_play() returns None for it, and the scores
say how many clips still need their serves marked.

    from evaluation.inplay import in_play
    frames = in_play(labels_json)        # set of frame numbers, or None (no serve marked)
"""
from __future__ import annotations


def in_play(V: dict) -> set[int] | None:
    contacts = {int(k): v for k, v in (V.get("contacts") or {}).items()}
    serves = sorted(f for f, v in contacts.items() if v == "serve")
    if not serves:
        return None
    n = V.get("n") or (max(int(k) for k in V["labels"]) + 1)
    ends = sorted(f for f, v in contacts.items() if v == "dead")
    out = set()
    for i, s in enumerate(serves):
        nxt = serves[i + 1] if i + 1 < len(serves) else n
        stop = min([e for e in ends if s < e < nxt] + [nxt])
        out.update(range(s, stop + 1 if stop < nxt else stop))
    return out
