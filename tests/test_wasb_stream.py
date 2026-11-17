"""WASB heatmaps streamed (core.ball.wasb.iter_heatmaps) = all at once, frame for frame."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")


class Fake(torch.nn.Module):
    """3 frames in (B, 9, H, W) -> 3 maps out: each frame's mean colour, as the model's raw output."""
    vv = {"input_wh": [64, 36], "resize": "letterbox"}

    def forward(self, x):
        return [torch.stack([x[:, 3 * j:3 * j + 3].mean(1) for j in range(3)], 1)]


@pytest.mark.parametrize("n", [9, 10, 11, 25])
def test_stream_matches_all_at_once(tmp_path, n):
    from core.ball import wasb
    v = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"MJPG"), 30, (128, 72))
    rng = np.random.default_rng(0)
    for _ in range(n):
        vw.write(rng.integers(0, 255, (72, 128, 3), dtype=np.uint8))
    vw.release()
    m = Fake()
    got = dict(wasb.iter_heatmaps(m, "cpu", v, n, batch=2))
    assert sorted(got) == list(range(n))
    # each frame's map must come from that frame: compare with running it alone in a triplet
    cap = cv2.VideoCapture(v)
    M = wasb.input_affine(m, v)
    for f in range(n):
        ok, img = cap.read()
        x = cv2.warpAffine(img, M, (64, 36), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - wasb.MEAN) / wasb.STD
        want = 1 / (1 + np.exp(-x.mean(2)))
        assert np.allclose(got[f], want, atol=1e-5), f
