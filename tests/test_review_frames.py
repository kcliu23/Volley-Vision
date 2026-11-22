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


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="needs ffprobe")
def test_missed_seek_falls_back_to_decoding(tmp_path, monkeypatch):
    from tools.review import FrameSource
    v = tmp_path / "v.mp4"
    _video(v)
    src = FrameSource(str(v), 1.0)
    assert src.can_seek()                                # the file itself is fine...
    monkeypatch.setattr(src.cv2, "VideoCapture", _OffByOne.wrap(src.cv2.VideoCapture))
    src.load(400, 550, pad=0)                            # ...but this capture's seeks land one frame late
    assert src.seek_ok is False
    for f in (400, 475, 550):
        assert _frame_no(src.get(f)) == f


class _OffByOne:
    """A VideoCapture whose seeks land one frame after the one asked for."""
    def __init__(self, cap):
        self.cap = cap

    @classmethod
    def wrap(cls, make):
        return lambda *a: cls(make(*a))

    def set(self, prop, value):
        return self.cap.set(prop, value + 1)

    def __getattr__(self, name):
        return getattr(self.cap, name)
