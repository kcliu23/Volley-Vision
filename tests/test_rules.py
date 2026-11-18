"""Volleyball's rules deciding touch sides (core.rules)."""
from core.rules.sides import Touch, decide

NET = 2.43


def T(x, z=1.0, sx=0.3, **kw):
    return Touch(x, sx, z, **kw)


def sides(ch):
    return [c.side for c in ch]


def test_clear_rally_follows_the_3d():
    ch = decide([T(-8, first=True), T(6), T(5), T(3), T(-5)], [4, 3, 4, 4], NET)
    assert sides(ch) == [-1, 1, 1, 1, -1]
    assert all(c.sure for c in ch)


def test_no_fourth_touch():
    # four touches on the far side in a row, the third close to the net: the rules move it
    ch = decide([T(-8, first=True), T(5), T(4), T(0.1), T(3)], [4, 3, 4, 4], NET)
    assert sides(ch)[1:].count(1) <= 3


def test_ball_cannot_change_side_low():
    # a dig below the tape, fitted a little past the net, then a low flight to the set on our side:
    # changing side there would mean going through the net, so the dig is ours
    ch = decide([T(6, first=True), T(-3, z=2.6), T(0.2, z=0.8, sx=0.4), T(-1.5, z=2.2)],
                [4, 3, 1.5], NET)
    assert ch[2].side == -1


def test_block_at_the_net_after_an_attack():
    ch = decide([T(-7, first=True), T(-3), T(-2, z=2.8), T(0.1, z=2.7), T(-4), T(-2)],
                [4, 4, 3, 2, 3], NET)
    assert ch[3].block and ch[3].side == 1          # the far team blocks our attack
    assert sides(ch)[4:] == [-1, -1]                 # we play it again: the block didn't count


def test_named_player_decides():
    ch = decide([T(6, first=True), T(0.3, z=0.8, player=-1, named=True)], [4], NET)
    assert ch[1].side == -1


# --- the serve rules: touches they forbid (rules.serve_faults) ---------------------------------
from core.rules.serve import serve_faults  # noqa: E402


def test_serve_split_on_the_servers_side_is_a_fault():
    # far serve; the fit split it mid-air on the far side (x 3.6, 3.6 m up), then the pass
    assert serve_faults((11.0, 2.7), [(3.6, 3.6), (-6.0, 0.9)], NET) == [0]


def test_toss_then_hit_far_back_is_the_serve():
    # the rally starts at the toss; the hit (far back) is the server's, not a fault
    assert serve_faults((11.5, 3.0), [(10.8, 2.8), (-6.0, 0.9), (-2.0, 2.3)], NET) == []


def test_serve_cannot_be_blocked():
    # the first touch on the receiving side at the net, up at the tape
    assert serve_faults((10.0, 2.5), [(-0.2, 3.3), (-6.0, 0.9)], NET) == [0]


def test_reception_at_the_net_below_the_tape_is_fine():
    assert serve_faults((10.0, 2.5), [(-0.5, 1.2)], NET) == []


def test_not_a_serve_no_serve_rules():
    # the rally starts near the net (not a serve): nothing forbidden
    assert serve_faults((2.0, 2.5), [(3.6, 3.6)], NET) == []


def test_serve_after_the_ball_was_held_low():
    # the rally starts with the server holding the ball (low), then the hit far back
    assert serve_faults((11.0, 1.3), [(10.1, 2.4), (5.3, 3.5), (-6.0, 0.9)], NET) == [1]


def test_far_back_high_touch_late_in_the_rally_is_not_a_serve():
    assert serve_faults((2.0, 1.0), [(-1.0, 2.0), (3.0, 1.0), (8.0, 2.0), (4.0, 2.0)], NET) == []


def test_after_the_reception_anything_goes():
    # two touches on the receiving side after the serve: not the serve rules' business
    assert serve_faults((-9.0, 2.4), [(6.0, 0.8), (1.0, 2.2)], NET) == []


# --- rules on a plain touch list, no 3D (core.rules.sequence) ----------------------------------
from core.rules.sequence import Contact, faults  # noqa: E402


def C(side, player=None, **kw):
    return Contact(0, side, player, **kw)


def rules_of(cs):
    return [(f.index, f.rule) for f in faults(cs)]


def test_seq_legal_rally():
    cs = [C(-1, 1, serve=True), C(1, 7), C(1, 8), C(1, 9), C(-1, 2, block=True), C(1, 10), C(1, 7), C(1, 8)]
    assert rules_of(cs) == []          # the block doesn't count: 3 more after it


def test_seq_four_touches():
    assert rules_of([C(-1, 1), C(1, 7), C(1, 8), C(1, 9), C(1, 10)]) == [(4, "four_touches")]


def test_seq_double_touch_but_not_after_own_block():
    assert rules_of([C(1, 7), C(1, 7)]) == [(1, "double_touch")]
    assert rules_of([C(-1, 1), C(1, 7, block=True), C(1, 7)]) == []


def test_seq_serve_rules():
    assert rules_of([C(-1, 1, serve=True), C(-1, 2)]) == [(1, "serve_own")]
    assert rules_of([C(-1, 1, serve=True), C(1, 7, block=True)]) == [(1, "serve_block")]


def test_seq_unknown_side_skips_counting():
    assert rules_of([C(1), C(1), C(None), C(1), C(1)]) == []
