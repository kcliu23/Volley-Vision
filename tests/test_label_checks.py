"""core.common.label_checks: frames to re-check in your labels; tools.review: who touched it."""
from core.common.label_checks import ball_flags, touch_flags
from tools.review import Review


def _line(n=40, step=10.0, start=0):
    return {f: [100 + step * (f - start), 300 + 0.5 * (f - start) ** 2 * 0.2] for f in range(start, start + n)}


def test_clean_flight_has_no_flags():
    assert ball_flags(_line()) == {}


def test_one_wrong_click_is_flagged():
    L = _line(); L[20] = [900.0, 50.0]
    F = ball_flags(L)
    assert 20 in F or 21 in F


def test_screen_recording_repeat_is_not_a_flag():
    L = _line()
    L[15] = list(L[14])                       # picture repeated ...
    for f in range(16, 40):                   # ... then a skipped frame to catch up: same path, one frame later
        L[f] = [100 + 10.0 * (f + 0), 300 + 0.5 * f ** 2 * 0.2]
    L = {f: v for f, v in L.items()}
    F = ball_flags(L)
    assert not any(14 <= f <= 17 for f in F), F


def test_blink_is_flagged():
    L = _line(); L[10] = "none"
    assert 10 in ball_flags(L)


def test_touch_gap_and_mixed_and_rally_marks():
    F = touch_flags({"10": "set", "12": "set", "30": "pass", "31": "dig", "50": "dead"})
    assert 12 in F and "gap" in F[12]
    assert 31 in F and "mixed" in F[31]
    assert 50 in F and "no serve" in F[50]
    assert touch_flags({"5": "serve", "40": "pass", "41": "pass", "90": "dead"}) == {}


def _review():
    N = 50
    P = {"n": N, "fps": 30, "width": 1920, "height": 1080,
         "frames": [{"xy": [900, 500], "box": [30, 30], "doubt": []} for _ in range(N)],
         "segments": [{"f0": 0, "f1": N - 1, "kind": "rally"}]}
    return Review(P, None, None)


def test_toucher_saved_removed_with_the_touch_and_undone():
    R = _review()
    R.set_contact(10, "set"); R.set_toucher(10, 7)
    assert R.to_json("v.mov")["touched_by"] == {"10": 7}
    R.set_contact(10, None)
    assert R.touched_by == {}
    R.undo()
    assert R.touched_by == {10: 7} and R.contacts == {10: "set"}
    assert Review(R.P, R.to_json("v.mov")).touched_by == {10: 7}
