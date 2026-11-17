"""Who touched it, from the pose players and your numbers (core.people.touchers, tools.review)."""
from core.people.touchers import Players


class Labels:                      # stands in for PlayerLabels: tracker id -> number
    def __init__(self, m):
        self.m = m

    def number(self, tid, f):
        return self.m.get(tid)


def kp(wrist):
    k = [[0.0, 0.0, 0.0]] * 17
    k = [list(x) for x in k]
    k[9] = [wrist[0], wrist[1], 0.9]
    return k


FR = {"10": [{"id": 1, "box": [100, 100, 160, 300], "kp": kp((150, 120))},     # #3, wrist at the ball
             {"id": 2, "box": [170, 100, 230, 300], "kp": kp((200, 150))},     # #9, a bit further
             {"id": 3, "box": [400, 100, 460, 300], "kp": kp((420, 150))}]}    # not numbered


def test_suggests_the_numbered_player_at_the_ball():
    P = Players(FR, Labels({1: 3, 2: 9}))
    assert P.suggest(10, (152, 118))["number"] == 3
    assert P.suggest(10, (199, 152))["number"] == 9


def test_no_suggestion_far_from_everyone_or_unnumbered():
    P = Players(FR, Labels({1: 3, 2: 9}))
    assert P.suggest(10, (800, 600)) is None
    assert P.suggest(10, (421, 151)) is None          # only the unnumbered player is there


def test_click_and_number_lookup():
    P = Players(FR, Labels({1: 3, 2: 9}))
    assert P.under(10, (180, 200))["number"] == 9
    assert P.under(10, (420, 200))["number"] is None
    assert P.with_number(10, 3)["id"] == 1


def test_review_saves_number_and_box_and_undoes():
    from tools.review import Review
    prop = {"n": 20, "fps": 60, "width": 640, "height": 360, "video": "x.mov", "segments": [{"f0": 0, "f1": 19, "kind": "rally"}],
            "frames": [{"state": "detected", "xy": [10, 10], "box": [20, 20], "doubt": []} for _ in range(20)]}
    R = Review(prop)
    R.set_contact(10, "dig")
    R.set_toucher(10, 3, [100, 100, 160, 300])
    J = R.to_json("x.mov")
    assert J["touched_by"] == {"10": 3} and J["touched_box"] == {"10": [100.0, 100.0, 160.0, 300.0]}
    R.undo()
    assert 10 not in R.touched_by and 10 not in R.touched_box
    R.set_contact(10, None)
    assert 10 not in R.contacts


def test_next_frame_joins_the_touch():
    from tools.review import Review
    prop = {"n": 20, "fps": 60, "width": 640, "height": 360, "video": "x.mov", "segments": [{"f0": 0, "f1": 19, "kind": "rally"}],
            "frames": [{"state": "detected", "xy": [10, 10], "box": [20, 20], "doubt": []} for _ in range(20)]}
    R = Review(prop)
    R.set_contact(10, "set")
    R.set_toucher(10, 2, [1, 2, 3, 4])
    assert R.extend_touch(11) == [10, 11]
    assert R.extend_touch(9) == [9, 10, 11]
    assert R.contacts[9] == "set" and R.touched_by[9] == 2 and R.touched_box[11] == [1, 2, 3, 4]
    R.set_toucher(11, 5)                                   # naming it on one frame names the whole touch
    assert all(R.touched_by[g] == 5 for g in (9, 10, 11))
    assert R.extend_touch(15) is None                      # nothing next to it
    R.undo()
    assert all(R.touched_by[g] == 2 for g in (9, 10, 11))


def test_toucher_box_follows_the_player():
    from tools.review import Review
    prop = {"n": 20, "fps": 60, "width": 640, "height": 360, "video": "x.mov", "segments": [{"f0": 0, "f1": 19, "kind": "rally"}],
            "frames": [{"state": "detected", "xy": [10, 10], "box": [20, 20], "doubt": []} for _ in range(20)]}
    R = Review(prop)
    base = {2: 90, 4: 288}                                  # each player moves 1 px a frame
    R.box_at = lambda g, n, ref: [base[n] + g, 0, base[n] + 50 + g, 200]
    R.set_contact(10, "set")
    R.set_toucher(10, 2, [100, 0, 150, 200])
    R.extend_touch(11)
    R.extend_touch(12)
    assert R.touched_box[12] == [102.0, 0.0, 152.0, 200.0] and R.touched_box[10] == [100.0, 0.0, 150.0, 200.0]
    R.set_toucher(12, 4, [300, 0, 350, 200])               # re-named on frame 12: every frame follows
    assert R.touched_box[12] == [300.0, 0.0, 350.0, 200.0] and R.touched_box[10] == [298.0, 0.0, 348.0, 200.0]
