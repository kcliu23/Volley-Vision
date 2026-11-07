from core.court import ATTACK, LENGTH, ORDER, WIDTH, keypoints, named_lines


def test_court_size():
    assert (LENGTH, WIDTH, ATTACK) == (18.0, 9.0, 3.0)


def test_keypoints_are_mirror_pairs():
    kp = keypoints()
    for n, (x, y, z) in kp.items():
        if n.endswith("left"):
            assert kp[n.replace("left", "right")] == (x, -y, z)


def test_click_order_names_exist():
    kp = keypoints()
    assert all(n in kp for n in ORDER)


def test_named_lines_lie_on_the_court():
    for a, b in named_lines().values():
        for x, y, z in (a, b):
            assert abs(x) <= LENGTH / 2 and abs(y) <= WIDTH / 2 and 0 <= z < 3
