"""tools.review.still_span: a resting spare ball is found frame by frame by its look (no YOLO)."""
import cv2
import numpy as np
import pytest

from tools.review import still_span

FPS, N, W, H = 30, 300, 640, 360
BALL = (200, 150)


def make_video(path, removed_at=250, occluded=(100, 130), moved_at=None):
    rng = np.random.default_rng(0)
    floor = (rng.integers(90, 130, (H, W, 3))).astype(np.uint8)          # textured floor
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for g in range(N):
        img = floor.copy()
        if g < removed_at:
            c = BALL if moved_at is None or g < moved_at else (400, 250)
            cv2.circle(img, c, 14, (40, 200, 240), -1)
            cv2.line(img, (c[0] - 14, c[1]), (c[0] + 14, c[1]), (20, 60, 200), 3)   # panels
            cv2.line(img, (c[0], c[1] - 14), (c[0], c[1] + 14), (230, 230, 230), 3)
        if occluded[0] <= g < occluded[1]:                                # a player walks in front
            cv2.rectangle(img, (150, 60), (260, 300), (30, 30, 80), -1)
        out.write(img)
    out.release()


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    make_video(p)
    return p


def test_finds_the_whole_resting_stretch_through_an_occlusion(video):
    a, b = still_span(video, 40, [*BALL, 28, 28], 28.0, FPS, N, log=lambda m: None)
    assert a == 0
    assert 240 <= b < 250                       # last look before it's taken away


def test_clicked_after_the_occlusion_goes_back_through_it(video):
    a, b = still_span(video, 200, [*BALL, 28, 28], 28.0, FPS, N, log=lambda m: None)
    assert a == 0 and 240 <= b < 250


def test_a_ball_that_moves_stops_where_it_left(tmp_path):
    p = tmp_path / "moved.mp4"
    make_video(p, removed_at=N, occluded=(0, 0), moved_at=180)
    a, b = still_span(p, 40, [*BALL, 28, 28], 28.0, FPS, N, log=lambda m: None)
    assert a == 0 and 170 <= b < 180

