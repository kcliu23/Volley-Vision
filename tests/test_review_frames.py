"""tools.review.FrameSource: jumping around (seek) gives the same frames as decoding in order."""
import shutil

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def _video(path, n=700):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 48))
    for i in range(n):
        img = np.zeros((48, 160, 3), np.uint8)
        for b in range(10):                       # frame number in the picture, 10 black / white bits
            if i >> b & 1:
                img[:, b * 16:(b + 1) * 16] = 255
        w.write(img)
    w.release()


def _frame_no(img):
    return sum(1 << b for b in range(10) if img[8:40, b * 16 + 4:(b + 1) * 16 - 4].mean() > 128)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_jumps_give_the_right_frames(tmp_path):
    from tools.review import FrameSource, constant_rate
    v = tmp_path / "v.mp4"
    _video(v)
    assert constant_rate(str(v))
    src = FrameSource(str(v), 1.0)
    for f in [10, 500, 480, 479, 100, 650, 3, 400]:       # forward, back one, far back, far forward
        src.load(max(0, f - 60), min(699, f + 90), pad=0)
        assert src.get(f) is not None and _frame_no(src.get(f)) == f
    assert src.seek_ok


def test_constant_rate_false_without_video(tmp_path):
    from tools.review import constant_rate
    assert not constant_rate(str(tmp_path / "missing.mp4"))
