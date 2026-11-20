from evaluation.rallies_eval import checked_span, judge, truth_rallies


def test_truth_rallies_joins_serve_runs_and_leaves_open_rally():
    C = {"100": "serve", "101": "serve", "150": "pass", "300": "dead", "500": "serve"}
    assert truth_rallies(C) == [(100, 300), (500, None)]


def test_checked_span_stops_at_last_closed_rally():
    truth = [(100, 300), (500, None)]
    assert checked_span(truth, [], fps=10, n=10_000) == (70, 330)        # 3 s pad, open rally not in it
    assert checked_span(truth, [800], fps=10, n=10_000) == (70, 830)     # a rejection (X) is checked too
    assert checked_span([(5, None)], [], fps=10, n=100) is None


def test_judge_one_to_one_and_accepted_guesses_apart():
    truth = [(100, 300), (500, 700), (900, None)]
    pred = [(100, 290), (480, 720), (490, 600), (800, 850), (2000, 2100)]
    J = judge(truth, pred, span=(0, 1000), rejected=[805])
    assert J["n_true"] == 2 and J["found"] == 2 and J["missed"] == []
    assert J["accepted_start"] == 1 and J["starts"] == [-20]              # 100 == 100: placed with P
    assert sorted(J["ends"]) == [-10, 20]
    assert sorted(J["fake"]) == [(490, 600), (800, 850)]                  # 2000-2100: past the span
    assert J["fake_rejected"] == 1
    assert J["play"] == 201 + 201 and J["play_kept"] == 191 + 201


def test_judge_missed():
    J = judge([(100, 300)], [], span=(0, 400))
    assert J["missed"] == [(100, 300)] and J["play_kept"] == 0 and J["dead_cut"] == J["dead"]


def test_edge_hits_start_strict_end_loose():
    from evaluation.rallies_eval import edge_hits
    J = {"starts": [2, -4, 30], "ends": [20, -31, 5]}
    H = edge_hits(J, 60.0)                       # start: 3 frames; end: 0.5 s = 30 frames
    assert (H["start_ok"], H["start_n"], H["end_ok"], H["end_n"]) == (1, 3, 2, 3)
