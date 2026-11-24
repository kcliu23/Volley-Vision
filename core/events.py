"""events.json: one row per touch, with its rally, team, player, type, zone (notes/ANALYSIS_DESIGN.md
section 3). Built from what the pipeline already writes: touches.json (the 3D fit's touches, their
side and toucher) and rallies.json (core.scene3d.rallies: start, end, winner, score).

The type comes from types_dp, not from a model: the cheapest touch sequence the rules allow
(serve, then per team 1st touch = reception after the serve, else dig; 2nd = set; 3rd = attack;
a touch up at the tape right after the other team's attack = block). A block doesn't count as
one of the team's touches (FIVB 14.4.1): after it, the next touch by either team starts that
team's count again (core.rules.sequence counts the same way).
"""
from __future__ import annotations

from core.rules.fivb import BLOCK_BELOW_TAPE_M, BLOCK_NEAR_M, NET_H_MEN_M, SERVE_HIT_X_M, SERVE_LOOK, SERVE_START_Z_M

TEAM = {-1: "near", 1: "far"}
ATTACK_LINE_M = 3.0
LANE_M = 1.5              # the middle lane is |y| <= 1.5 m (three 3 m lanes across 9 m)


def zone(xy, side: int) -> int | None:
    """FIVB zone 1-6 of a floor spot (court metres, +x far) on that team's side, from the team's
    view facing the net (near team: right is -y; far team: right is +y). Off the court: the zone
    of the nearest lane / row (a server behind the end line is in their back row lane)."""
    if xy is None or side not in (-1, 1):
        return None
    x, y = float(xy[0]), float(xy[1])
    right = side * y             # + = to the team's right (near: -y, far: +y)
    lane = "r" if right > LANE_M else ("l" if right < -LANE_M else "m")
    front = abs(x) <= ATTACK_LINE_M
    return {("r", True): 2, ("m", True): 3, ("l", True): 4,
            ("r", False): 1, ("m", False): 6, ("l", False): 5}[(lane, front)]


def build(touches: list[dict], rallies: list[dict], clip: str, fps: float | None = None,
          net_h: float = NET_H_MEN_M, z_model: dict | None = None) -> dict:
    """touches: touches.json["touches"]; rallies: rallies.json["rallies"]."""
    R, E = [], []
    for ri, r in enumerate(rallies, 1):
        ts = [t for t in touches if r["f0"] - 3 <= t["frame"] <= r["f1"]]
        R.append({"id": ri, "f0": r["f0"], "f1": r["f1"], "serve_team": TEAM.get(r.get("serve_side")),
                  "winner": TEAM.get(r.get("winner")), "end": r.get("end"),
                  "score": [r.get("near"), r.get("far")]})
        for seq, (t, kind) in enumerate(zip(ts, types_dp(ts, net_h, z_model, r.get("serve_side"))), 1):
            who = t.get("toucher") or {}
            feet = who.get("feet")
            E.append({"rally": ri, "seq": seq, "frame": t["frame"],
                      "t": t.get("time_s", round(t["frame"] / fps, 3) if fps else None),
                      "team": TEAM.get(t.get("side")), "player": who.get("number"), "type": kind,
                      "player_xy": feet, "zone": zone(feet, t.get("side")),
                      "ball_xyz": t.get("xyz"), "quality": None, "conf": None})
    _grade_receptions(E)
    return {"schema": 1, "clip": clip, "rallies": R, "events": E}


SETTER_A_NET_M = 2.0      # [guess, design section 4] the setter within this of the net, zone 2-3: A
SETTER_B_NET_M = 3.5      # [guess] ... within this (or front row elsewhere): B; further: C


def _grade_receptions(E: list[dict]):
    """Reception quality from the setter's spot at the next touch (design section 4). Without a
    setter yet (no lineup), the team's next touch stands in for the set."""
    for i, e in enumerate(E):
        if e["type"] != "reception":
            continue
        nxt = E[i + 1] if i + 1 < len(E) and E[i + 1]["rally"] == e["rally"] else None
        if nxt is None or nxt["team"] != e["team"]:
            e["quality"] = "error"                         # ace / shank / overpass
            continue
        xy = nxt["player_xy"]
        if xy is None:
            continue
        d = abs(float(xy[0]))
        if nxt["zone"] in (2, 3) and d <= SETTER_A_NET_M:
            e["quality"] = "A"
        elif d <= SETTER_B_NET_M:
            e["quality"] = "B"
        else:
            e["quality"] = "C"


# --- types as the cheapest legal sequence (order = hard rules, height and side = evidence) -------
import math  # noqa: E402

TYPE_Z = {"P1": (1.3, 0.45), "P2": (1.75, 0.3), "P3": (2.65, 0.4)}   # ball height at the touch (m):
                         # mean, sd per 1st / 2nd / 3rd touch [fit on holdouts 1-10, Nov 19; check
                         # with evaluation.events_eval --folds before trusting]
OVER_COST = 2.5          # [guess] nats: the ball sent over on the 1st or 2nd touch (overpass, dump)
INVENTED_COST = 3.0      # [guess] nats: a touch the 3D made up (2.2 per 10 real ones, Nov 18)
SIDE_COST_FAR = 2.5      # [guess] nats: the 3D's side overruled, away from the net ...
SIDE_COST_NET = 0.5      # ... and within NET_UNSURE_M of it (one camera: depth is weak there)
NET_UNSURE_M = 1.5


def _emit(t, kind: str, z_model: dict) -> float:
    xyz = t.get("xyz")
    if xyz is None or kind not in z_model:
        return 0.0
    mu, sd = z_model[kind]
    return 0.5 * ((xyz[2] - mu) / sd) ** 2 + math.log(sd)


def _side_cost(t, team: int) -> float:
    s, xyz = t.get("side"), t.get("xyz")
    if s not in (-1, 1) or s == team:
        return 0.0
    return SIDE_COST_NET if xyz is not None and abs(xyz[0]) < NET_UNSURE_M else SIDE_COST_FAR


def types_dp(touches: list[dict], net_h: float = NET_H_MEN_M, z_model: dict | None = None,
             serve_side: int | None = None) -> list[str]:
    """The type of each touch of ONE rally: the cheapest sequence the rules allow. serve_side: the
    serving team from the rally finder (the server behind the end line, core.scene3d.rallies):
    when no touch is far back and high (a serve the 3D didn't see: often the near server, behind
    the camera) and the first touch is on the other side, the serve isn't among the touches."""
    z_model = z_model or TYPE_Z
    if not touches:
        return []
    sv = next((i for i, t in enumerate(touches[:SERVE_LOOK]) if t.get("xyz") is not None
               and abs(t["xyz"][0]) >= SERVE_HIT_X_M and t["xyz"][2] >= SERVE_START_Z_M), None)
    if sv is None and serve_side in (-1, 1) and touches[0].get("side") == -serve_side:
        sv, st = -1, serve_side                     # the serve is not among the touches
    else:
        sv = sv or 0
        st = touches[sv].get("side")
        if st not in (-1, 1):
            st = 1 if (touches[sv].get("xyz") or [1])[0] > 0 else -1
    # state: (kind of the last real touch, its team) -> (cost, path)
    best = {("S", st): (0.0, [])}
    nxt_of = {"S": [("P1", -1, 0.0)], "P1": [("P2", 1, 0.0), ("P1", -1, OVER_COST)],
              "P2": [("P3", 1, 0.0), ("P1", -1, OVER_COST)], "P3": [("P1", -1, 0.0), ("B", -1, 0.0)],
              "B": [("P1", 1, 0.0), ("P1", -1, 0.0)]}
    for t in touches[sv + 1:]:
        new: dict = {}
        xyz = t.get("xyz")
        blockable = xyz is not None and abs(xyz[0]) < BLOCK_NEAR_M and xyz[2] > net_h - BLOCK_BELOW_TAPE_M
        for (k, team), (c, path) in best.items():
            opts = [(k, team, INVENTED_COST, "O")]                       # made up: nothing changes
            for k2, same, extra in nxt_of[k]:
                if k2 == "B" and not blockable:
                    continue
                team2 = team * same
                opts.append((k2, team2, extra + _emit(t, k2, z_model) + _side_cost(t, team2), k2))
            for k2, team2, cc, lab in opts:
                tot = c + cc
                if tot < new.get((k2, team2), (math.inf,))[0]:
                    new[(k2, team2)] = (tot, path + [(lab, team2)])
        best = new
    _, path = min(best.values(), key=lambda v: v[0])
    out = [] if sv < 0 else ["other"] * sv + ["serve"]
    prev = "S"
    for lab, _team in path:
        if lab == "O":
            out.append("other")
            continue
        out.append({"P1": "reception" if prev == "S" else "dig", "P2": "set", "P3": "attack", "B": "block"}[lab])
        prev = lab
    return out
