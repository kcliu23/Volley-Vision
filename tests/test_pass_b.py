import numpy as np

from core.rally import pass_b


def test_window_labels_half_rule():
    lab = pass_b.window_labels(set(range(10, 40)), n=60, fps=20)          # windows of 10 frames
    assert lab.tolist() == [0, 1, 1, 1, 0, 0]


def test_to_rallies_joins_gaps_and_drops_short():
    fps = 20                                                              # 10 frames per window
    p = np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    # windows 1-4 and 6-7: gap of 1 window (0.5 s) joined -> frames 10..79 (3.5 s);
    # window 12 alone: 0.5 s < MIN_RALLY_S, dropped
    assert pass_b.to_rallies(p, fps, n=170) == [(10, 79)]


def test_features_shape_and_unknown_people():
    fps, n = 20, 100
    pos = np.full((n, 2), np.nan)
    pos[20:60] = [[500, 300]] * 40
    X, names = pass_b.features(pos, fps, None, 1080.0, None)
    assert X.shape == (10, len(names)) and len(names) == 3 * (len(pass_b.BALL) + len(pass_b.PEOPLE))
    assert X[3, names.index("seen")] == 1.0 and X[0, names.index("seen")] == 0.0
    assert np.isnan(X[:, names.index("people")]).all()                    # no pose: unknown, not zero


def test_snap_edges():
    v1 = [(100, 500), (1000, 1200)]
    rule = [(130, 470), (2000, 2100)]
    # start 100 -> 130 (0.5 s away), end 500 -> 470; the second has no rule rally: kept as is
    assert pass_b.snap_edges(v1, rule, fps=60) == [(130, 470), (1000, 1200)]
    assert pass_b.snap_edges([(100, 500)], [(400, 900)], fps=60) == [(100, 500)]   # rule edges > 2 s away
