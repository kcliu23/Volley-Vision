"""evaluation.inplay: in play = from a serve mark to the rally-over mark."""
from evaluation.inplay import in_play


def clip(contacts, n=100):
    return {"n": n, "labels": {str(f): [0, 0] for f in range(n)}, "contacts": {str(k): v for k, v in contacts.items()}}


def test_no_serve_marked_means_unknown():
    assert in_play(clip({30: "set"})) is None


def test_serve_to_rally_over():
    assert in_play(clip({20: "serve", 35: "pass", 80: "dead"})) == set(range(20, 81))


def test_serve_to_clip_end_without_a_rally_over_mark():
    assert in_play(clip({20: "serve"})) == set(range(20, 100))


def test_two_rallies():
    p = in_play(clip({10: "serve", 40: "dead", 60: "serve", 90: "dead"}))
    assert p == set(range(10, 41)) | set(range(60, 91))


def test_next_serve_ends_a_rally_without_its_own_mark():
    assert in_play(clip({10: "serve", 60: "serve"})) == set(range(10, 100))
