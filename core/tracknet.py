"""TrackNet ball heatmaps for a clip, turned into candidates and cached.

TrackNet (TrackNetV3, github.com/qaz812345/TrackNetV3) looks at 8 frames at once plus the
clip's background image at 512x288 and outputs one heatmap per frame: how likely the ball
centre is at each pixel. Motion makes blurred and faint balls stand out and still objects
fade into the background, so it finds balls YOLO misses (see core.tracknet_fill).

Each heatmap blob becomes a candidate [cx, cy, w, h, peak] in full-resolution pixels.
TrackNet gives no ball size, so w = h = the ball size YOLO saw nearby in time.
Cached as output/<stem>/cache/tracknet.json (core.paths), keyed by the checkpoint contents.

The network definition below is from TrackNetV3 (model.py), MIT License,
Copyright (c) 2024 qaz812345 — see the license text at the end of this file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

H, W = 288, 512
TRACKNET_CACHE_VERSION = 1
DEFAULT_CKPT = "models/tracknet/FT_TrackNet_best.pt"


# ── network (TrackNetV3, MIT) ───────────────────────────────────────────────

def _build(in_dim: int, out_dim: int):
    import torch.nn as nn
    import torch

    class Conv2DBlock(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv = nn.Conv2d(i, o, kernel_size=3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(o)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class Double2DConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv_1, self.conv_2 = Conv2DBlock(i, o), Conv2DBlock(o, o)

        def forward(self, x):
            return self.conv_2(self.conv_1(x))

    class Triple2DConv(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.conv_1, self.conv_2, self.conv_3 = Conv2DBlock(i, o), Conv2DBlock(o, o), Conv2DBlock(o, o)

        def forward(self, x):
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNet(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.down_block_1 = Double2DConv(i, 64)
            self.down_block_2 = Double2DConv(64, 128)
            self.down_block_3 = Triple2DConv(128, 256)
            self.bottleneck = Triple2DConv(256, 512)
            self.up_block_1 = Triple2DConv(768, 256)
            self.up_block_2 = Double2DConv(384, 128)
            self.up_block_3 = Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, o, (1, 1))
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            pool = nn.MaxPool2d((2, 2), stride=(2, 2))
            up = nn.Upsample(scale_factor=2)
            x1 = self.down_block_1(x)
            x2 = self.down_block_2(pool(x1))
            x3 = self.down_block_3(pool(x2))
            x = self.bottleneck(pool(x3))
            x = self.up_block_1(torch.cat([up(x), x3], dim=1))
            x = self.up_block_2(torch.cat([up(x), x2], dim=1))
            x = self.up_block_3(torch.cat([up(x), x1], dim=1))
            return self.sigmoid(self.predictor(x))

    return TrackNet(in_dim, out_dim)


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str):
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pd = ck["param_dict"]
    if pd.get("bg_mode") != "concat":
        raise ValueError(f"only bg_mode=concat checkpoints are supported (got {pd.get('bg_mode')})")
    L = int(pd["seq_len"])
    m = _build((L + 1) * 3, L)
    m.load_state_dict(ck["model"])
    dev = _device()
    return m.eval().to(dev), L, dev


# ── inference ────────────────────────────────────────────────────────────────

def read_frames(path: str, n: int | None = None) -> np.ndarray:
    """All frames as RGB 512x288 (PIL bicubic, as TrackNetV3 resizes)."""
    from PIL import Image
    cap = cv2.VideoCapture(path)
    out = []
    while n is None or len(out) < n:
        ok, img = cap.read()
        if not ok:
            break
        out.append(np.asarray(Image.fromarray(img[:, :, ::-1]).resize((W, H))))
    cap.release()
    return np.stack(out) if out else np.zeros((0, H, W, 3), np.uint8)


def heatmaps(model, L: int, dev, frames: np.ndarray, batch: int = 4, log=None) -> np.ndarray:
    """(N, H, W) heatmaps, non-overlapping 8-frame windows (each frame predicted once)."""
    import torch
    n = len(frames)
    idx = np.linspace(0, n - 1, min(n, 1800)).astype(int)
    med = np.moveaxis(np.median(frames[idx], 0).astype(np.uint8), -1, 0).astype(np.float32) / 255.0
    starts = list(range(0, n, L))
    out = np.zeros((n, H, W), np.float16)
    t0 = time.time()
    for b in range(0, len(starts), batch):
        xs, spans = [], []
        for s in starts[b:b + batch]:
            ids = [min(s + k, n - 1) for k in range(L)]          # pad the tail with the last frame
            seq = np.moveaxis(frames[ids], -1, 1).reshape(L * 3, H, W).astype(np.float32) / 255.0
            xs.append(np.concatenate([med, seq], 0))
            spans.append((s, min(L, n - s)))
        with torch.no_grad():
            y = model(torch.from_numpy(np.stack(xs)).to(dev)).float().cpu().numpy()
        for (s, k), yy in zip(spans, y):
            out[s:s + k] = yy[:k]
        if log and (b // batch) % 25 == 0:
            log(f"  TrackNet {min(n, (starts[min(b + batch, len(starts)) - 1]) + L)}/{n} frames  {time.time() - t0:.0f}s")
    return out


def blobs(hm: np.ndarray, thr: float = 0.1, top: int = 3):
    """Heatmap -> [(x, y, peak)] in heatmap px, strongest first."""
    mask = (hm >= thr).astype(np.uint8)
    k, lab, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    res = []
    for i in range(1, k):
        ys, xs = np.nonzero(lab == i)
        w = hm[ys, xs].astype(np.float64)
        res.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.max())))
    res.sort(key=lambda r: -r[2])
    return res[:top]


def ball_sizes(dets: dict, fps: float) -> list[float]:
    """Per-frame ball size: median of strong YOLO detections within ±1 s, else the clip median."""
    n = len(dets["frames"])
    pts = [(f, max(c[2], c[3])) for f, fr in enumerate(dets["frames"]) for c in fr if c[4] >= 0.4]
    med = float(np.median([s for _, s in pts])) if pts else 30.0
    if not pts:
        return [med] * n
    F = np.array([p[0] for p in pts]); S = np.array([p[1] for p in pts])
    r = max(1, int(round(fps)))
    out = []
    for f in range(n):
        m = np.abs(F - f) <= r
        out.append(float(np.median(S[m])) if m.sum() >= 3 else med)
    return out


def _ckpt_key(ckpt: str) -> str:
    """Identifies the checkpoint by content (size + hash of its first and last MB), so the
    cache stays valid when the file is copied between machines."""
    import hashlib
    size = os.path.getsize(ckpt)
    h = hashlib.sha1()
    with open(ckpt, "rb") as fh:
        h.update(fh.read(1 << 20))
        if size > (2 << 20):
            fh.seek(-(1 << 20), 2)
            h.update(fh.read(1 << 20))
    return f"{size}:{h.hexdigest()[:16]}:v{TRACKNET_CACHE_VERSION}"


def candidates(video: str, dets: dict, ckpt: str = DEFAULT_CKPT, cache_path: str | Path | None = None,
               thr: float = 0.1, log=print, redo: bool = False) -> list[list] | None:
    """TrackNet candidates per frame for this clip (cached). None if no checkpoint."""
    if not ckpt or not Path(ckpt).exists():
        if log:
            log(f"[VolleyVision] TrackNet checkpoint not found ({ckpt}); skipping TrackNet")
        return None
    from core import paths
    stem = Path(video).stem
    cp = Path(cache_path) if cache_path else paths.tracknet(stem)
    key = _ckpt_key(ckpt)
    n = len(dets["frames"])
    if cp.exists() and not redo:
        try:
            c = json.load(open(cp))
            if c.get("key") == key and len(c.get("frames", [])) == n and c.get("thr") == thr:
                return c["frames"]
        except Exception:
            pass
    t0 = time.time()
    model, L, dev = load_model(ckpt)
    if log:
        log(f"[VolleyVision] TrackNet on {dev} ({Path(ckpt).name})")
    frames = read_frames(video, n)
    hms = heatmaps(model, L, dev, frames, log=log)
    sx, sy = dets["width"] / W, dets["height"] / H
    sizes = ball_sizes(dets, dets.get("fps") or 60)
    out = []
    for f in range(n):
        if f >= len(hms):
            out.append([])
            continue
        s = round(sizes[f], 1)
        out.append([[round(x * sx, 1), round(y * sy, 1), s, s, round(p, 4)]
                    for x, y, p in blobs(hms[f].astype(np.float32), thr)])
    cp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"schema": "vv-tracknet/1", "video": str(video), "key": key, "thr": thr, "seq_len": L,
               "frames": out}, open(cp, "w"), separators=(",", ":"))
    if log:
        log(f"[VolleyVision] TrackNet: {sum(1 for f in out if f)}/{n} frames with a candidate "
            f"({time.time() - t0:.0f}s)  →  {cp}")
    return out


# TrackNetV3 license (applies to the network definition above):
#
# MIT License
#
# Copyright (c) 2024 qaz812345
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software, pretrained model checkpoints, and associated documentation
# files (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
