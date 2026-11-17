"""Rallies and the score (core.scene3d.rallies)."""
from core.scene3d.rallies import find_rallies, score


def B(points, chains=None, fps=60.0):
    """points: {frame: (x, y, z)}; one flight per chain (default all in chain 0)."""
    chains = chains or [(min(points), max(points))]
    fl = [{"f0": a, "f1": b, "chain": i} for i, (a, b) in enumerate(chains)]
    return {"fps": fps, "flights": fl, "frames": {str(f): {"xyz": list(p)} for f, p in points.items()}}


def arc(f0, f1, x0, x1, top=4.0, z0=2.5, z1=0.2):
    """From z0 up to about top and down to z1 (default: the floor) at x1."""
    n = f1 - f0
    return {f0 + i: (x0 + (x1 - x0) * i / n, 0.0,
                     z0 + (z1 - z0) * (i / n) + top * 4 * (i / n) * (1 - i / n)) for i in range(n + 1)}


def test_lands_in_far_side_near_wins():
    r = find_rallies(B(arc(0, 100, -10, 5)), [{"frame": 0, "kind": "serve", "side": -1}])
    assert r[0].end == "in" and r[0].winner == -1


def test_out_after_far_touch_near_wins():
    pts = arc(0, 60, -10, 3, z1=1.0) | arc(61, 120, 3, -14, z0=1.0)            # far team sends it long, past our end line
    r = find_rallies(B(pts), [{"frame": 0, "kind": "serve", "side": -1}, {"frame": 61, "kind": "touch", "side": 1}])
    assert r[0].end == "out" and r[0].winner == -1


def test_pieces_join_until_the_floor_and_next_serve_decides():
    pts = {f: (-10 + f * 0.25, 0.0, 2.5 + 0.02 * f) for f in range(0, 41)}       # served, ball lost mid-air
    pts |= {f: (1.0 + (f - 80) * 0.05, 0.0, 3.0) for f in range(80, 100)}           # seen again, no landing
    pts |= arc(400, 500, 10, -3)
    chains = [(0, 40), (80, 99), (400, 500)]
    T = [{"frame": 0, "kind": "serve", "side": -1}, {"frame": 400, "kind": "serve", "side": 1}]
    r = find_rallies(B(pts, chains), T)
    assert len(r) == 2 and r[0].f0 == 0 and r[0].winner == 1        # the far team serves next: they won it
    rows = score(r)
    assert (rows[-1]["near"], rows[-1]["far"]) == (0, 2)
