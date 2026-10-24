"""Run a TrackNetV3 checkpoint on a clip and cache ball candidates in YOLO's format.

    python -m tools.tracknet_dump tests/test42.MP4 --ckpt models/tracknet/TrackNet_best.pt --tag bad
    python -m tools.tracknet_dump --gt --ckpt ... --tag vb      # every clip with GT labels

TrackNet sees 8 frames (+ the clip's median background image) at 512x288 and outputs one
heatmap per frame. Each heatmap blob becomes a candidate [cx, cy, w, h, conf]:
  cx, cy  heatmap-weighted centroid, scaled to the original resolution
  conf    blob peak (0..1)
  w, h    TrackNet gives no box size, so we use the ball size YOLO saw nearby in time
          (median of strong detections within ±1 s), else the clip's typical ball size
Cached as output/<clip>/cache/tracknet_<tag>.json: {"frames": [[cand, ...], ...], ...}

Needs the TrackNetV3 code (github.com/qaz812345/TrackNetV3, MIT) on --repo.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

H, W = 288, 512


def load_model(ckpt_path, repo):
    import torch
    sys.path.insert(0, repo)
    from model import TrackNet
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pd = ck["param_dict"]
    assert pd.get("bg_mode") == "concat", f"only bg_mode=concat supported, got {pd.get('bg_mode')}"
    L = pd["seq_len"]
    m = TrackNet(in_dim=(L + 1) * 3, out_dim=L)
    m.load_state_dict(ck["model"])
    return m.eval(), L


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        rgb = Image.fromarray(img[:, :, ::-1])
        out.append(np.asarray(rgb.resize((W, H))))        # PIL default (bicubic), as TrackNetV3 does
    cap.release()
    return np.stack(out)                                    # (N, H, W, 3) uint8


def heatmaps(model, L, frames, batch=4, log=print):
    import torch
    n = len(frames)
    idx = np.linspace(0, n - 1, min(n, 1800)).astype(int)
    median = np.median(frames[idx], 0).astype(np.uint8)
    med = np.moveaxis(median, -1, 0).astype(np.float32) / 255.0
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
            y = model(torch.from_numpy(np.stack(xs))).numpy()
        for (s, k), yy in zip(spans, y):
            out[s:s + k] = yy[:k]
        if log and (b // batch) % 10 == 0:
            done = min(n, starts[min(b + batch, len(starts)) - 1] + L)
            log(f"  {done}/{n} frames  {time.time() - t0:.0f}s")
    return out


def blobs(hm, thr=0.1, top=3):
    """Heatmap -> [(x, y, peak)] in heatmap px, strongest first."""
    mask = (hm >= thr).astype(np.uint8)
    k, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    res = []
    for i in range(1, k):
        ys, xs = np.nonzero(lab == i)
        w = hm[ys, xs].astype(np.float64)
        res.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), float(w.max())))
    res.sort(key=lambda r: -r[2])
    return res[:top]


def ball_sizes(dets, fps):
    """Per-frame ball size from strong YOLO detections within ±1 s (median), else clip median."""
    n = len(dets["frames"])
    pts = [(f, max(c[2], c[3])) for f, fr in enumerate(dets["frames"]) for c in fr if c[4] >= 0.4]
    med = float(np.median([s for _, s in pts])) if pts else 30.0
    if not pts:
        return [med] * n
    F = np.array([p[0] for p in pts]); S = np.array([p[1] for p in pts])
    r = int(round(fps))
    out = []
    for f in range(n):
        m = np.abs(F - f) <= r
        out.append(float(np.median(S[m])) if m.sum() >= 3 else med)
    return out


def dump(video, ckpt, tag, repo, thr=0.1, log=print):
    stem = Path(video).stem
    from core import paths
    dpath = paths.dets(stem)
    dets = json.load(open(dpath))
    model, L = load_model(ckpt, repo)
    frames = read_frames(video)
    n = len(dets["frames"])
    if len(frames) != n:
        log(f"  note: {len(frames)} decoded frames vs {n} in dets; using min")
    hms = heatmaps(model, L, frames[:n], log=log)
    sx, sy = dets["width"] / W, dets["height"] / H
    sizes = ball_sizes(dets, dets.get("fps") or 60)
    out_frames = []
    for f in range(n):
        if f >= len(hms):
            out_frames.append([])
            continue
        s = sizes[f]
        out_frames.append([[round(x * sx, 1), round(y * sy, 1), round(s, 1), round(s, 1), round(p, 4)]
                           for x, y, p in blobs(hms[f].astype(np.float32), thr)])
    res = {"video": str(video), "ckpt": str(ckpt), "tag": tag, "thr": thr, "seq_len": L,
           "mode": "nonoverlap", "frames": out_frames}
    op = paths.tracknet(stem, tag=tag)
    json.dump(res, open(op, "w"))
    log(f"{stem}: {sum(len(f) > 0 for f in out_frames)}/{n} frames with a candidate -> {op}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="*")
    ap.add_argument("--gt", action="store_true", help="every clip with labels in data/gt")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repo", default="../tn/TrackNetV3")
    ap.add_argument("--video-dir", default="/mnt/user-data/uploads/Volley-Vision/tests")
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    vids = list(a.videos)
    if a.gt:
        for g in sorted(glob.glob("data/gt/*_gt.json")):
            stem = Path(g).name.replace("_gt.json", "")
            vids += [p for p in glob.glob(f"{a.video_dir}/{stem}.*")]
    for v in vids:
        print(f"== {v}", flush=True)
        dump(v, a.ckpt, a.tag, a.repo, a.thr, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
