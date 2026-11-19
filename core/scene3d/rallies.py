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
SERVE_ZONE_M = 5.0             # m: behind an end line, up to this far back: where a server stands
SERVE_LOOK_S = 6.0             # s: looked at before each rally for the server
MIN_RALLY_S = 1.0              # s: shorter, with no serve: the ball bouncing between points
PLAY_APEX_M = 2.0              # m: a flight this high starts a rally (lower: the ball bounced or held before the serve)
SERVE_BEFORE_S = 3.0           # s: a serve touch up to this before that flight is the rally's start
LAND_FROM_M = 1.5              # m: a floor contact is one when the ball came down from this high ...
LAND_LOOK_S = 1.0              # s: ... within this before it ...
UP_AFTER_M = 2.0               # m: ... and doesn't go back up this high ...
UP_AFTER_S = 1.0               # s: ... within this after it (a dig near the floor does; so does 3D noise)
START_RUN_S = 2.0              # s: ... and the 3D carries on from it this long (a lone flight is noise)
START_GAP_S = 1.0              # s: ... with no hole longer than this
BACK_GAP_S = 2.5               # s: high flights this close before a rally's start belong to it
END_GAP_S = 1.5                # s: the ball gone this long and back only low: the rally ended before the hole
DEAD_LOW_M = 1.5               # m: the ball below this ...
DEAD_LOW_S = 1.5               # s: ... for this long is dead (rolling, held), not in play
REST_S = 3.0                   # s: the next rally starts no sooner than this after one ends (the ball
                               #   thrown back to the server is not a rally)
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


def server_sides(rallies: list, poses: dict, cam, fps: float) -> list:
    """Which team served each rally, from the players (not the ball: a far serve is a few blurred
    pixels, often not tracked). Before a serve, the server stands behind their end line: one more
    person behind it than usual (a referee or a spectator may always be there). Per rally: +1 far,
    -1 near, None unknown. When neither end shows an extra person and the camera stands close
    behind the near end line (the near server is out of the picture), it's near."""
    import numpy as np
    frames = {int(k): v for k, v in poses.items() if v}
    if not frames:
        return [None] * len(rallies)

    def behind(f):
        n = {-1: 0, 1: 0}
        for p in frames.get(f, []):
            b = p["box"]
            g = cam.floor_point([((b[0] + b[2]) / 2, b[3])], 0.0)[0]
            if np.isfinite(g[0]) and HALF_L <= abs(g[0]) <= HALF_L + SERVE_ZONE_M and abs(g[1]) <= HALF_W + 1.5:
                n[1 if g[0] > 0 else -1] += 1
        return n
    every = sorted(frames)[::30]
    usual = {s: float(np.median([behind(f)[s] for f in every])) for s in (-1, 1)}
    near_hidden = abs(float(cam.position[0])) - HALF_L < 3.0
    out, prev_end = [], 0
    for r in rallies:
        lo = max(prev_end + int(0.5 * fps), r.f0 - int(SERVE_LOOK_S * fps))
        c = [behind(f) for f in range(lo, r.f0 + 10, 5) if f in frames]
        side = None
        if c:
            extra = {s: float(np.percentile([x[s] for x in c], 75)) - usual[s] for s in (-1, 1)}
            if extra[1] >= 1 and extra[1] >= extra[-1]:
                side = 1
            elif extra[-1] >= 1:
                side = -1
            elif near_hidden:
                side = -1
        out.append(side)
        prev_end = r.f1
    return out


def _rally_spans(fr: dict, flights: list, touches: list, fps: float) -> list:
    """(start, landing frame or None, last frame, flights) per rally, from the 3D alone.

    A rally starts with the first real flight (up to PLAY_APEX_M or higher) after the last one ended:
    before the serve the ball is bounced or held low, and that is not play. (A serve touch, if there
    is one, marks the start exactly.) It ends at the first floor contact that is one: the ball comes
    down from LAND_FROM_M and doesn't go back up past UP_AFTER_M (a dig near the floor does, and so
    does a frame or two of 3D noise at the start of a flight), or when the ball is gone for MAX_GAP_S.
    """
    keys = sorted(fr)
    if not keys:
        return []
    import bisect

    def zmax(a, b):
        i, j = bisect.bisect_left(keys, a), bisect.bisect_right(keys, b)
        return max((fr[keys[k]][2] for k in range(i, j)), default=-1.0)
    apex = [zmax(F["f0"], F["f1"]) for F in flights]
    serves = sorted(t["frame"] for t in touches if t.get("kind") == "serve")
    look, after = int(LAND_LOOK_S * fps), int(UP_AFTER_S * fps)
    gap, end_gap, low_for = int(MAX_GAP_S * fps), int(END_GAP_S * fps), int(DEAD_LOW_S * fps)

    def goes_on(f, run=START_RUN_S):                      # the 3D carries on from f this long
        k, last = bisect.bisect_left(keys, f), f
        while k < len(keys) and keys[k] - last <= int(START_GAP_S * fps):
            last = keys[k]
            if last - f >= int(run * fps):
                return True
            k += 1
        return False
    def seen_after(g):                                    # how far to look after a low point: UP_AFTER_S,
        n = bisect.bisect_right(keys, g + after) - bisect.bisect_right(keys, g)   # or past a hole in the
        return after if n >= after // 3 else int(BACK_GAP_S * fps)               # 3D (it may come back up)
    out, free_from, i = [], -10 ** 9, 0
    while i < len(flights):
        served = any(flights[i]["f0"] - 6 <= f <= flights[i]["f1"] for f in serves)
        if flights[i]["f0"] < free_from or not ((served and goes_on(flights[i]["f0"], START_RUN_S / 2))
                                                 or (apex[i] >= PLAY_APEX_M and goes_on(flights[i]["f0"]))):
            i += 1
            continue
        j = b = i                                         # the serve and the pass, when the 3D lost the
        while (j > 0 and flights[j - 1]["f0"] >= free_from          # ball for a moment right after them
               and flights[j]["f0"] - flights[j - 1]["f1"] <= int(BACK_GAP_S * fps)):
            j -= 1
            b = j if apex[j] >= PLAY_APEX_M else b
        F = flights[b]
        sv = [f for f in serves if max(free_from, F["f0"] - int(SERVE_BEFORE_S * fps)) <= f <= F["f1"]]
        start = sv[-1] if sv else F["f0"]
        k = bisect.bisect_left(keys, start)
        land_f, last, high = None, keys[k] if k < len(keys) else start, start
        while k < len(keys):
            g = keys[k]
            if g - last > gap or (g - last > end_gap and zmax(g, g + int(2 * fps)) < PLAY_APEX_M):
                break                                     # gone for good, or back only low (picked up)
            if g - last > end_gap:
                high = g                                  # back after a hole, and up: still in play
            if g - high > low_for:
                last = high                               # low for this long: the ball is dead; it came
                break                                     # down right after the last time it was up
            last = g
            if fr[g][2] >= DEAD_LOW_M:
                high = g
            if fr[g][2] <= FLOOR_Z and zmax(g - look, g) >= LAND_FROM_M and zmax(g + 1, g + seen_after(g)) < UP_AFTER_M:
                run = [h for h in keys[k:bisect.bisect_right(keys, g + int(0.2 * fps))] if fr[h][2] <= FLOOR_Z]
                land_f = min(run, key=lambda h: fr[h][2])
                break
            k += 1
        end = land_f if land_f is not None else last
        out.append((start, land_f, end, [G for G in flights if G["f1"] >= start and G["f0"] <= end]))
        free_from = end + int(REST_S * fps)
        while i < len(flights) and flights[i]["f0"] < free_from:
            i += 1
    return out


def find_rallies(B: dict, touches: list, servers=None) -> list[Rally]:
    """B: ball3d.json; touches: touches.json["touches"]; servers: optional (poses, cam) to find the
    serving team from the players (server_sides), which then decides each rally's winner (the
    winner serves next) ahead of where the ball landed."""
    fr = {int(k): v["xyz"] for k, v in B["frames"].items()}
    fps = float(B.get("fps") or 60.0)
    flights = sorted(B["flights"], key=lambda F: F["f0"])
    out = []
    for start, land_f, f1, fl in _rally_spans(fr, flights, touches, fps):
        tc = [t for t in touches if start - 6 <= t["frame"] <= f1]
        serve = next((t for t in tc if t.get("kind") == "serve" and t["frame"] <= start + 6), None)
        land, extrap = (fr[land_f][:2] if land_f is not None else None), False
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
    if servers is not None:
        poses, cam = servers
        for r, side in zip(out, server_sides(out, poses, cam, fps)):
            if side is not None:
                r.serve_side = side
        for a, b in zip(out[:-1], out[1:]):              # the winner serves next: from the players, that's
            if b.serve_side is not None:                 # the surest clue there is (the landing is a 3D guess)
                if a.winner is not None and a.winner != b.serve_side:
                    a.why = f"{'near' if b.serve_side < 0 else 'far'} serves next (where it landed says otherwise: {a.why})"
                else:
                    a.why = f"{'near' if b.serve_side < 0 else 'far'} serves next" + (f"; {a.why}" if a.winner is not None else "")
                a.winner = b.serve_side
        return out
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
