"""tools.label_players marks: a fix at one frame never undoes another frame's mark.

KC's case on holdout1 (2026-11-10): one tracker track is the bench guy, then drifts onto player 1.
Before, every key named the whole track, so fixing frame 820 flipped frame 790 back.
"""
from tools.label_players import mark, piece_of

FR = list(range(700, 944))              # the track is seen in these frames


def value(names, cuts, tid, f):
    return names.get(piece_of(cuts, tid, f))


def new():
    return {}, {}, {}


def test_first_key_names_the_whole_track():
    n, c, a = new()
    mark(n, c, a, 45, 790, "x", FR)
    assert value(n, c, 45, 700) == "x" and value(n, c, 45, 943) == "x" and not c


def test_fix_later_frame_keeps_earlier():
    n, c, a = new()
    mark(n, c, a, 45, 790, "x", FR)
    mark(n, c, a, 45, 820, 1, FR)
    assert value(n, c, 45, 790) == "x" and value(n, c, 45, 819) == "x"
    assert value(n, c, 45, 820) == 1 and value(n, c, 45, 943) == 1


def test_fix_between_moves_only_the_switch():
    n, c, a = new()
    mark(n, c, a, 45, 790, "x", FR)
    mark(n, c, a, 45, 820, 1, FR)
    mark(n, c, a, 45, 805, 1, FR)          # it really switches at 805
    assert value(n, c, 45, 804) == "x" and value(n, c, 45, 805) == 1 and value(n, c, 45, 900) == 1
    assert c[45] == [805]                 # the 820 cut joined into 805's part


def test_fix_earlier_frame_keeps_later():
    n, c, a = new()
    mark(n, c, a, 45, 850, 1, FR)          # named while scanning backwards
    mark(n, c, a, 45, 774, "x", FR)
    assert value(n, c, 45, 774) == "x" and value(n, c, 45, 700) == "x"
    assert value(n, c, 45, 850) == 1 and value(n, c, 45, 943) == 1


def test_back_and_forth_does_not_flip():
    """The loop KC was stuck in: alternate fixes at two frames, each must stay as set."""
    n, c, a = new()
    for _ in range(3):
        mark(n, c, a, 45, 790, "x", FR)
        mark(n, c, a, 45, 820, 1, FR)
        assert value(n, c, 45, 790) == "x" and value(n, c, 45, 820) == 1


def test_same_frame_twice_renames():
    n, c, a = new()
    mark(n, c, a, 8, 0, 3, range(0, 300))
    mark(n, c, a, 8, 0, 4, range(0, 300))
    assert value(n, c, 8, 0) == 4 and value(n, c, 8, 299) == 4 and not c


def test_unname_from_here():
    n, c, a = new()
    mark(n, c, a, 45, 790, "x", FR)
    mark(n, c, a, 45, 900, None, FR)
    assert value(n, c, 45, 899) == "x" and value(n, c, 45, 900) is None


def test_other_tracks_untouched():
    n, c, a = new()
    mark(n, c, a, 2, 790, 2, FR)
    mark(n, c, a, 45, 790, "x", FR)
    mark(n, c, a, 45, 820, 1, FR)
    assert value(n, c, 2, 820) == 2 and 2 not in c
