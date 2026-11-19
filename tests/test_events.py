"""events.json (core.events) and the first stats (core.stats)."""
from core.events import build, types_dp, zone
from core.stats import team_stats


def T(f, x, z, side=None, y=0.0, feet=None):
    return {"frame": f, "xyz": [x, y, z], "side": side if side is not None else (1 if x > 0 else -1),
            "time_s": f / 60, "toucher": {"number": None, "feet": feet}}


def test_zone_fivb_numbering():
    assert zone([-1.0, -3.0], -1) == 2          # near team: right is -y, front row
    assert zone([-1.0, 0.0], -1) == 3
    assert zone([-6.0, 3.0], -1) == 5
    assert zone([1.0, 3.0], 1) == 2             # far team: right is +y
    assert zone([10.0, 3.0], 1) == 1            # a server behind the end line: zone 1 lane
    assert zone(None, 1) is None


def test_typical_rally():
    ts = [T(0, 10.0, 2.4), T(60, -5.0, 1.2), T(100, -1.0, 1.8), T(130, -0.5, 2.7), T(200, 5.0, 1.0)]
    assert types_dp(ts) == ["serve", "reception", "set", "attack", "dig"]


def test_held_ball_before_serve_and_wrong_side_at_net():
    ts = [T(0, 8.0, 1.0), T(40, 10.0, 2.4), T(100, -5.0, 1.2),
          T(140, 0.6, 1.8, side=1),             # the set, put on the wrong side at the net
          T(170, -0.5, 2.7)]
    assert types_dp(ts) == ["other", "serve", "reception", "set", "attack"]


def test_unseen_serve_uses_rally_serve_side():
    ts = [T(60, 5.0, 1.2), T(100, 1.0, 1.8), T(130, 0.5, 2.7)]       # near served; the far team plays
    assert types_dp(ts, serve_side=-1) == ["reception", "set", "attack"]


def test_block_after_attack():
    ts = [T(0, 10.0, 2.4), T(60, -5.0, 1.2), T(100, -1.0, 1.8), T(130, -0.5, 2.8), T(135, 0.4, 2.6)]
    assert types_dp(ts)[-1] == "block"


def test_stats_and_reception_grade():
    ts = [T(0, 10.0, 2.4, feet=[10.5, 3.0]), T(60, -5.0, 1.2, feet=[-6.0, 0.0]),
          T(100, -1.0, 1.8, feet=[-1.0, -2.0]), T(130, -0.5, 2.7, feet=[-1.5, 2.5])]
    ev = build(ts, [{"f0": 0, "f1": 200, "serve_side": 1, "winner": -1, "end": "in", "near": 1, "far": 0}], "c")
    rec = [e for e in ev["events"] if e["type"] == "reception"][0]
    assert rec["quality"] == "A"                 # the set taken in zone 2, 1 m off the net
    st = team_stats(ev)
    assert st["teams"]["near"]["attacks"] == 1 and st["teams"]["far"]["serves"] == 1
    assert st["side_out"] == 1.0                 # the receiving team won it
    assert st["timing_s"]["set_to_attack"] == 0.5
