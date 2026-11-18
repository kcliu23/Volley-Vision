"""Is the 3D ball within reach of a player at each of your touches?

Someone touched the ball at every touch you marked (tools.review), so the 3D ball must be near at
least one of the players you numbered (tools.label_players). No volleyball rules: a dig can be a
dive 2 m from the feet, a block reaches over the net. Only a ball far from EVERY numbered player is
plainly wrong: the 3D put it at the wrong distance.

    gap  horizontal distance (m) from the 3D ball to the feet of the player you said touched it
         (tools.review: the number after the touch type), else of the nearest numbered player

A player's feet: the lowest box bottom over the half second up to the touch (a jumping hitter's
feet are in the air at the touch; the takeoff spot is on the floor), put on the floor through
the calibrated camera. Only touches where a numbered player is SEEN at the ball count: some
numbered box, grown by half its size, holds the ball in the picture. Otherwise the toucher isn't
in the picture (a far server behind the baseline is often not detected at all) and the nearest
numbered player is someone else; those touches are skipped (counted).

    python -m evaluation.reach3d holdout1 ... --output outA --name now [--output outB --name new]
    python -m evaluation.reach3d holdout1 ... --by-side

--by-side splits the touches by where the 3D ball is: the camera's half, the far half within
NET_ZONE_M of the net, and the far back court. Far players are small and mostly unnumbered (and
pose before the far court zoom barely found them), so far back court gaps mostly measure to the
wrong player (Nov 18, 10 holdouts: camera half 49 touches, 0% over 3 m; far back court 37, 54%,
8 of them far serves 0.6-2.9 m behind the end line, where the unnumbered server stands).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.common import paths
from core.geometry.camera import Camera
from evaluation.inplay import in_play

FAR_M = 3.0             # farther than this from every player: the 3D is grossly off
TAKEOFF_S = 0.5
NET_ZONE_M = 2.5        # --by-side: the far half this close to the net is its own group


def touches_all(V: dict):
    """Your touches including serves: (first, last, type)."""
    c = sorted((int(k), v) for k, v in (V.get("contacts") or {}).items() if v not in ("dead",))
    out = []
    for f, v in c:
        if out and f - out[-1][1] <= 1:
            out[-1] = (out[-1][0], f, out[-1][2])
        else:
            out.append((f, f, v))
    return out


def player_feet(ids: dict, cam, f: int, fps: float) -> dict[int, np.ndarray]:
    """number -> floor spot (x, y) at touch frame f."""
    low = {}
    for g in range(f - int(TAKEOFF_S * fps), f + 3):
        for b in (ids.get("frames") or {}).get(str(g), []):
            n = b.get("player")
            if not isinstance(n, int):
                continue
            x0, y0, x1, y1 = b["box"]
            if n not in low or y1 > low[n][1]:
                low[n] = ((x0 + x1) / 2, y1)
    return {n: cam.floor_point([uv], 0.0)[0][:2] for n, uv in low.items()}


def _seen(ids: dict, f: int, ball_uv) -> bool:
    for g in range(f - 2, f + 3):
        for b in (ids.get("frames") or {}).get(str(g), []):
            if not isinstance(b.get("player"), int):
                continue
            x0, y0, x1, y1 = b["box"]
            w, h = x1 - x0, y1 - y0
            if x0 - w / 2 <= ball_uv[0] <= x1 + w / 2 and y0 - h / 2 <= ball_uv[1] <= y1 + h / 2:
                return True
    return False


def gaps(stem: str, output: str, name: str = "ball3d.json"):
    V = json.load(open(paths.verified(stem)))
    ids = json.load(open(Path("data/player_ids") / f"{stem}.json"))
    cam = Camera.load(Path("data/cameras") / f"{stem}.json")
    B = json.load(open(Path(output) / stem / name))
    fr = {int(k): np.array(v["xyz"]) for k, v in B["frames"].items()}
    play = in_play(V)                                    # None: no serve marks: every touch counts
    fps = float(B.get("fps") or 60.0)
    from tools.ball3d_run import positions
    tr = json.load(open(Path(output) / stem / "track.json"))
    C = json.load(open(Path(output) / stem / "contacts.json"))
    uv, _ = positions(stem, tr, False, C)
    out, skipped = [], 0
    who_at = {int(k): int(v) for k, v in (V.get("touched_by") or {}).items()}
    for t0, t1, kind in touches_all(V):
        if play is not None and not (t0 in play or t1 in play or kind == "serve"):
            continue
        f = (t0 + t1) // 2
        near = [g for g in range(f - 2, f + 3) if g in fr]
        feet = player_feet(ids, cam, f, fps)
        g = min((g for g in range(f - 2, f + 3) if g < len(uv) and np.isfinite(uv[g, 0])), key=lambda g: abs(g - f), default=None)
        if not near or not feet or g is None or not _seen(ids, f, uv[g]):
            skipped += 1
            continue
        X = fr[min(near, key=lambda g: abs(g - f))]
        who = [w for g, w in who_at.items() if t0 - 1 <= g <= t1 + 1]
        if who:                                          # you said who touched it: measure to them
            if who[0] not in feet:                       # ... but they have no box here
                skipped += 1
                continue
            n, d = who[0], float(np.hypot(*(X[:2] - feet[who[0]])))
        else:
            n, d = min(((n, float(np.hypot(*(X[:2] - p)))) for n, p in feet.items()), key=lambda q: q[1])
        out.append((f, kind, d, n, float(X[2])))
    return out, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stems", nargs="+")
    ap.add_argument("--output", action="append")
    ap.add_argument("--name", action="append")
    ap.add_argument("--detail", action="store_true", help="list the far touches")
    ap.add_argument("--by-side", action="store_true", help="also split by the ball's half of the court")
    a = ap.parse_args()
    outs = a.output or [str(paths.OUTPUT)]
    names = a.name or outs
    print(f"{'version':14s} {'touches':>7s} {'median gap':>10s} {'p90':>6s} {'over 3 m':>9s} {'toucher unseen':>14s}")

    def row(label, G, sk):
        G = np.array(G, dtype=float)
        G = G[np.isfinite(G)]                       # a toucher with no feet seen (e.g. the server) can't be judged
        if not len(G):
            print(f"{label:14s} {0:7d} {'-':>10s} {'-':>6s} {'-':>9s} {sk:14d}")
            return
        print(f"{label:14s} {len(G):7d} {np.median(G):9.2f}m {np.percentile(G, 90):5.2f}m {np.mean(G > FAR_M):8.1%} {sk:14d}")

    for out, name in zip(outs, names):
        G, sk, far, side = [], 0, [], []
        for s in a.stems:
            if not (Path(out) / s / "ball3d.json").exists():
                continue
            g, k = gaps(s, out)
            G += [x[2] for x in g]; sk += k
            far += [(s,) + x for x in g if x[2] > FAR_M]
            if a.by_side:
                cam = Camera.load(Path("data/cameras") / f"{s}.json")
                fr = json.load(open(Path(out) / s / "ball3d.json"))["frames"]
                near_sign = np.sign(cam.position[0]) or -1.0
                for x in g:
                    k_ = min(fr, key=lambda q: abs(int(q) - x[0]))
                    side.append(near_sign * fr[k_]["xyz"][0])   # > 0: the camera's half
        row(name, G, sk)
        if a.by_side:
            G_, S_ = np.array(G, dtype=float), np.array(side)
            for label, m in (("  camera half", S_ > 0), ("  far, net", (S_ <= 0) & (S_ > -NET_ZONE_M)),
                             ("  far, back", S_ <= -NET_ZONE_M)):
                row(label, G_[m], 0)
        if a.detail:
            for s, f, kind, d, n, z in sorted(far, key=lambda q: -q[4]):
                print(f"{'':14s} {s} frame {f} {kind}: ball {d:.2f} m from the nearest player (#{n}), {z:.2f} m up")


if __name__ == "__main__":
    main()
