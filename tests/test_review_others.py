"""tools.review.Review: fixing a follow that went wrong cuts the track there, it doesn't erase it."""
from tools.review import Review

N = 100


def review_with_track():
    P = {"n": N, "fps": 30, "width": 1920, "height": 1080,
         "frames": [{"xy": [900, 500], "box": [30, 30]} for _ in range(N)],
         "segments": [{"f0": 0, "f1": N - 1}]}
    R = Review(P, None, None)
    R.other_clicks = []
    for g in range(N):                       # a spare ball rolling slowly, and one resting far away
        R.others_edit[g] = [[100 + g, 200, 30, 30], [1500, 900, 30, 30]]
    return R


def frames_with(R, x_min, x_max):
    return sorted(g for g in range(N) if any(x_min <= o[0] <= x_max for o in R.others(g)))


def test_shift_click_on_orange_cuts_from_here_to_the_end():
    R = review_with_track()
    R.toggle_other(60, (160, 200))
    assert frames_with(R, 0, 400) == list(range(60))           # before frame 60 stays
    assert frames_with(R, 1400, 1600) == list(range(N))        # the other ball is untouched


def test_option_shift_click_cuts_back_to_the_start():
    R = review_with_track()
    R.toggle_other(60, (160, 200), backward=True)
    assert frames_with(R, 0, 400) == list(range(61, N))


def test_undo_brings_it_back():
    R = review_with_track()
    R.toggle_other(60, (160, 200))
    R.undo()
    assert frames_with(R, 0, 400) == list(range(N))


def test_same_with_crop_following_on():
    R = review_with_track()
    R.follow = True
    R.toggle_other(60, (160, 200))
    assert frames_with(R, 0, 400) == list(range(60))


def test_r_removes_one_box_in_this_frame_only():
    R = review_with_track()
    R.remove_other_here(60, (163, 205))
    assert frames_with(R, 0, 400) == [g for g in range(N) if g != 60]
    assert frames_with(R, 1400, 1600) == list(range(N))


def test_capital_r_clears_this_frame_only():
    R = review_with_track()
    R.clear_others_here(60)
    assert R.others(60) == [] and len(R.others(59)) == 2 and len(R.others(61)) == 2
    R.undo()
    assert len(R.others(60)) == 2


def empty_review():
    R = review_with_track()
    R.others_edit = {}
    return R


def test_shift_click_marks_this_frame_only_no_following():
    R = empty_review()
    msg = R.toggle_other(10, (500, 300))
    assert [g for g in range(N) if R.others(g)] == [10] and "keyframe 1" in msg


def test_still_ball_is_marked_where_it_stays():
    R = empty_review()
    R.still_find = lambda f, box: (5, 80)
    msg = R.toggle_other(10, (500, 300))
    assert [g for g in range(N) if R.others(g)] == list(range(5, 81)) and "still ball" in msg
    assert R.other_clicks == []                    # a still ball isn't a keyframe


def test_g_joins_all_keyframes_of_a_moving_ball():
    R = empty_review()
    for f, x in ((10, 500), (30, 700), (50, 600)):  # it changes direction at frame 30
        R.toggle_other(f, (x, 300))
    assert R.fill_others() == 19 + 19
    assert R.others(20)[0][0] == 600 and R.others(40)[0][0] == 650
    assert R.other_clicks == []                    # ready for the next ball
    R.undo()
    assert [g for g in range(N) if R.others(g)] == [10, 30, 50]
