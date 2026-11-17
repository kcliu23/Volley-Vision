"""Volleyball's rules deciding touch sides (core.scene3d.rules)."""
from core.scene3d.rules import Touch, decide

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
