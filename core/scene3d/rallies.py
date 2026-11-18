"""Rallies and the score, from the 3D ball (ball3d.json) and its touches (touches.json).

A rally starts at the serve and ends when the ball first touches the floor (or is lost for good).
Who won it (rally-point scoring, FIVB 6.1):
  - the ball lands IN on one side: the other team wins;
  - it lands OUT: the team that touched it last loses;
  - and the team that wins a rally serves the next one: the next serve's side confirms (or, when
    the landing wasn't seen, decides) the winner.
Teams are "near" (the camera's side of the net) and "far".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

HALF_L, HALF_W = 9.0, 4.5      # m: the court (x across the net, y along it)
LINE = 0.10                    # m: a ball touching the line is in (FIVB 8.3); our 3D has some slack too
FLOOR_Z = 0.30                 # m: the ball's centre this low is on the floor
MAX_GAP_S = 4.0                # s: the ball unseen longer than this between two pieces: a new rally
MIN_RALLY_S = 1.0              # s: shorter, with no serve: the ball bouncing between points (a full game:
                               #   59 found -> 33 with this, of 34 real)
NET_UNSURE_M = 1.5             # m: a landing this close to the net: its side from the last flight (over or not)
CARRY_FROM_Z = 1.5             # m: a rally's last arc ending this low and falling is carried to the floor ...
CARRY_MAX_S = 0.6              # s: ... if it gets there within this


@dataclass
class Rally:
    f0: int                    # first frame (the serve, or the first frame with the ball)
    f1: int                    # last frame (the floor contact, or the last frame with the ball)
    serve_side: int | None     # -1 near, +1 far
    end: str                   # "in", "out", "lost" (no floor contact seen)
    land: list | None          # [x, y] where it landed
    last_touch_side: int | None
    winner: int | None         # -1 near, +1 far, None unknown
    why: str


def _side(x: float) -> int:
    return -1 if x < 0 else 1


def _carry_to_floor(fr: dict, f1: int, fps: float, n: int = 8, g: float = 9.81):
    """The last arc (its last n frames) carried on under gravity to the floor, when the ball is on
    its way down and low: where it lands [x, y], or None."""
    import numpy as np
    pts = [(f, fr[f]) for f in range(f1 - n + 1, f1 + 1) if f in fr]
    if len(pts) < 4:
        return None
    t = np.array([(f - f1) / fps for f, _ in pts])
    X = np.array([p for _, p in pts], float)
    vx, x0 = np.polyfit(t, X[:, 0], 1)
    vy, y0 = np.polyfit(t, X[:, 1], 1)
    vz, z0 = np.polyfit(t, X[:, 2] + 0.5 * g * t ** 2, 1)
    if X[-1, 2] > CARRY_FROM_Z or vz > 0.5:              # still high, or going up: not landing yet
        return None
    r = 0.105                                            # the ball touches the floor at z = its radius
    disc = vz * vz + 2 * g * (z0 - r)
    if disc < 0:
        return None
    tl = (vz + disc ** 0.5) / g
    if not 0 <= tl <= CARRY_MAX_S:
        return None
    return [float(x0 + vx * tl), float(y0 + vy * tl)]


def find_rallies(B: dict, touches: list) -> list[Rally]:
    """B: ball3d.json; touches: touches.json["touches"]."""
    fr = {int(k): v["xyz"] for k, v in B["frames"].items()}
    fps = float(B.get("fps") or 60.0)
    chains: dict = {}
    for F in B["flights"]:
        chains.setdefault(F["chain"], []).append(F)
    # the 3D splits where the ball is lost for a moment; a rally only ends at the floor, a new serve,
    # or a long gap: pieces in between belong to the rally before them
    groups: list = []
    for c in sorted(chains, key=lambda c: min(F["f0"] for F in chains[c])):
        fl = sorted(chains[c], key=lambda F: F["f0"])
        has_serve = any(t.get("kind") == "serve" and fl[0]["f0"] - 6 <= t["frame"] <= fl[-1]["f1"] for t in touches)
        if groups and not has_serve and not groups[-1]["landed"] and fl[0]["f0"] - groups[-1]["fl"][-1]["f1"] <= MAX_GAP_S * fps:
            groups[-1]["fl"] += fl
        else:
            groups.append({"fl": fl})
        groups[-1]["landed"] = any(fr.get(g, (0, 0, 9))[2] <= FLOOR_Z for F in groups[-1]["fl"]
                                   for g in range(F["f0"], F["f1"] + 1))
    out = []
    for G in groups:
        fl = G["fl"]
        f0, f1 = fl[0]["f0"], fl[-1]["f1"]
        tc = [t for t in touches if f0 - 6 <= t["frame"] <= f1]
        serve = next((t for t in tc if t.get("kind") == "serve"), None)
        start = serve["frame"] if serve else f0
        # the first floor contact after the serve: the lowest point of a flight that reaches the floor
        land_f, land = None, None
        for F in fl:
            if F["f1"] <= start:
                continue
            low = [g for g in range(max(F["f0"], start), F["f1"] + 1) if g in fr and fr[g][2] <= FLOOR_Z]
            if low:
                g = min(low, key=lambda g: fr[g][2])
                land_f, land = g, fr[g][:2]
                break
        extrap = False
        if land is None:                                  # the 3D stops just before the floor: carry the
            land, extrap = _carry_to_floor(fr, f1, fps), True   # last arc down to it (physics)
            if land is not None:
                land_f = f1
        end_f = land_f if land_f is not None else f1
        before = [t for t in tc if start < t["frame"] < end_f and t.get("kind") != "serve"]
        last = before[-1] if before else serve
        last_side = last["side"] if last else None
        if land is not None and abs(land[0]) < NET_UNSURE_M and last_side is not None:
            # landing by the net: from one camera its side is a guess; whether the last flight went
            # over the tape is not. Didn't go over: it fell on the last toucher's side (into the net)
            Fl = next((F for F in fl if F["f0"] <= end_f <= F["f1"]), fl[-1])
            nc = Fl.get("net_crossing")
            over = bool(nc) and nc.get("clears_net_by_m", -1) >= 0 and (last is None or Fl["f0"] >= last["frame"] - 6)
            land_side = -last_side if over else last_side
            inside = abs(land[1]) <= HALF_W + LINE
            end = "in" if inside else "out"
            if not over:
                winner, why = -last_side, f"fell on the {'near' if last_side < 0 else 'far'} side by the net (didn't go over)"
            elif inside:
                winner, why = -land_side, f"went over and landed in on the {'near' if land_side < 0 else 'far'} side by the net"
            else:
                winner, why = -last_side, f"went over and out by the net after the {'near' if last_side < 0 else 'far'} team's touch"
        elif land is not None:
            inside = abs(land[0]) <= HALF_L + LINE and abs(land[1]) <= HALF_W + LINE
            end = "in" if inside else "out"
            if inside:
                winner, why = -_side(land[0]), f"landed in on the {'near' if land[0] < 0 else 'far'} side"
                why += " (arc carried to the floor)" if extrap else ""
            elif last_side is not None:
                winner, why = -last_side, f"out after the {'near' if last_side < 0 else 'far'} team's touch"
                why += " (arc carried to the floor)" if extrap else ""
            else:
                winner, why = None, "out, last toucher unknown"
        else:
            end, winner, why = "lost", None, "no floor contact seen"
        out.append(Rally(start, end_f, serve["side"] if serve else None, end,
                         [round(v, 2) for v in land] if land is not None else None, last_side, winner, why))
    # a "rally" shorter than this with no serve is the ball bouncing or rolling between points
    out = [r for r in out if (r.f1 - r.f0) / fps >= MIN_RALLY_S or r.serve_side is not None]
    # the winner serves next: confirms the landing, or decides when the landing wasn't seen
    for a, b in zip(out[:-1], out[1:]):
        if b.serve_side is None:
            continue
        if a.winner is None:
            a.winner, a.why = b.serve_side, "the next serve is theirs"
        elif a.winner != b.serve_side:
            a.why += f" (but the next serve is the {'near' if b.serve_side < 0 else 'far'} team's: check)"
    return out


def score(rallies: list[Rally]) -> list[dict]:
    """Rallies with the running score after each: [{..., "near": n, "far": m}]."""
    near = far = 0
    rows = []
    for r in rallies:
        near += r.winner == -1
        far += r.winner == 1
        rows.append({**asdict(r), "near": near, "far": far})
    return rows
