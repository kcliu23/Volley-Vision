"""VballNet ball finder (github.com/asigatchov/vball-net, MIT): a drop-in alternative to TrackNet.

A volleyball-specific, lightweight TrackNetV4-style model (motion attention, depthwise
convolutions), released as ONNX files, so it runs with onnxruntime (no TensorFlow needed).

Same output as core.tracknet.candidates: per frame, [[x, y, size, size, peak], ...] in video
pixels, cached in output/<clip>/cache/tracknet_vballnet.json.

    python -m core.vballnet videos/test53.mov                          # first .onnx in models/vballnet/
    python -m core.vballnet videos/test53.mov --ckpt models/vballnet/VballNetFastV1_155_h288_w512.onnx

Weights: download the demo models from the repo's README (Google Drive or Yandex Disk) and put
the .onnx files in models/vballnet/. Needs:  pip install onnxruntime

Preprocessing follows the repo's own inference scripts: frames stretched to 512x288, /255,
- models with "seq9_grayscale" in the file name: 9 grayscale frames in, 9 heatmaps out
- the others: 3 RGB frames in, 3 heatmaps out
run on non-overlapping groups of frames; output is a probability heatmap (sigmoid).
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TAG = "vballnet"
IN_W, IN_H = 512, 288


def default_ckpt() -> str | None:
    found = sorted(glob.glob("models/vballnet/*.onnx"))
    return found[0] if found else None


def load_model(ckpt: str):
    import onnxruntime as ort
    prov = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    try:
        sess = ort.InferenceSession(ckpt, providers=prov)
    except Exception:                                   # e.g. an op CoreML can't take
        sess = ort.InferenceSession(ckpt, providers=["CPUExecutionProvider"])
    gray = "seq9_grayscale" in Path(ckpt).name
    return sess, (9 if gray else 3), gray


def heatmaps(sess, k: int, gray: bool, video: str, n: int, log=None) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    frames = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        if gray:
            x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (IN_W, IN_H)).astype(np.float32) / 255.0
            frames.append(x[None])                       # (1, H, W)
        else:
            x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IN_W, IN_H)).astype(np.float32) / 255.0
            frames.append(x.transpose(2, 0, 1))          # (3, H, W)
    cap.release()
    m = len(frames)
    out = np.zeros((m, IN_H, IN_W), np.float16)
    starts = list(range(0, max(m - k, 0) + 1, k))
    if starts and starts[-1] + k < m:
        starts.append(m - k)
    name = sess.get_inputs()[0].name
    t0 = time.time()
    for i, s in enumerate(starts):
        x = np.concatenate(frames[s:s + k], 0)[None].astype(np.float32)
        y = sess.run(None, {name: x})[0][0]              # (k, H, W)
        out[s:s + k] = y
        if log and i % 100 == 0:
            log(f"  VballNet {min(s + k, m)}/{m} frames  {time.time() - t0:.0f}s")
    return out


def candidates(video: str, dets: dict, ckpt: str | None = None, cache_path=None, thr: float = 0.1,
               log=print, redo: bool = False) -> list[list] | None:
    ckpt = ckpt or default_ckpt()
    if not ckpt or not Path(ckpt).exists():
        if log:
            log("[VolleyVision] VballNet weights not found (models/vballnet/*.onnx); see core/vballnet.py")
        return None
    from core import paths
    from core.tracknet import blobs, ball_sizes, _ckpt_key
    stem = Path(video).stem
    cp = Path(cache_path) if cache_path else paths.tracknet(stem, tag=TAG)
    key = "vballnet:" + _ckpt_key(ckpt)
    n = len(dets["frames"])
    if cp.exists() and not redo:
        try:
            c = json.load(open(cp))
            if c.get("key") == key and len(c.get("frames", [])) == n and c.get("thr") == thr:
                return c["frames"]
        except Exception:
            pass
    t0 = time.time()
    sess, k, gray = load_model(ckpt)
    if log:
        log(f"[VolleyVision] VballNet ({Path(ckpt).name}, {k} {'grayscale' if gray else 'RGB'} frames) "
            f"on {sess.get_providers()[0]}")
    hms = heatmaps(sess, k, gray, video, n, log=log)
    sx, sy = dets["width"] / IN_W, dets["height"] / IN_H
    sizes = ball_sizes(dets, dets.get("fps") or 60)
    out = []
    for f in range(n):
        if f >= len(hms):
            out.append([])
            continue
        s = round(sizes[f], 1)
        out.append([[round((x + 0.5) * sx - 0.5, 1), round((y + 0.5) * sy - 0.5, 1), s, s, round(p, 4)]
                    for x, y, p in blobs(hms[f].astype(np.float32), thr)])
    cp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"schema": "vv-tracknet/1", "model": "vballnet", "ckpt": Path(ckpt).name, "video": str(video),
               "key": key, "thr": thr, "frames": out}, open(cp, "w"), separators=(",", ":"))
    if log:
        log(f"[VolleyVision] VballNet: {sum(1 for f in out if f)}/{n} frames with a candidate "
            f"({time.time() - t0:.0f}s)  →  {cp}")
    return out


def main():
    import argparse
    from core import paths
    ap = argparse.ArgumentParser(description="VballNet ball candidates for clips (needs the YOLO cache for ball sizes)")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    for v in a.videos:
        dp = paths.dets(Path(v).stem)
        if not dp.exists():
            print(f"[vballnet] {Path(v).stem}: no YOLO cache ({dp}); run main.py or tools.dump_detections first")
            continue
        candidates(v, json.load(open(dp)), a.ckpt, redo=a.redo)


if __name__ == "__main__":
    main()
