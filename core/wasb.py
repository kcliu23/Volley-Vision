"""WASB ball finder (BMVC 2023, github.com/nttcom/WASB-SBDT, MIT): a drop-in alternative to TrackNet.

WASB is a small HRNet (~1.5M parameters) that looks at 3 consecutive frames and outputs a ball
heatmap for each. Unlike TrackNet it keeps high-resolution features all the way through, so
it places small balls more precisely; the authors' volleyball weights beat TrackNetV2 on their
volleyball benchmark at about the same speed.

Same output as core.tracknet.candidates: per frame, [[x, y, size, size, peak], ...] in video
pixels, cached in output/<clip>/cache/tracknet_wasb.json. So every tool that reads TrackNet
candidates can read these too (tools/archive/dev/eval_tracknet.py --tag wasb, the linkers, ...).

    python -m core.wasb videos/test53.mov                  # writes the cache, prints a summary
    python -m core.wasb videos/*.mov --ckpt models/wasb/wasb_volleyball_best.pth.tar

Weights: download "WASB (Ours) / Volleyball" from the WASB repo's MODEL_ZOO.md (Google Drive)
and save it as models/wasb/wasb_volleyball_best.pth.tar. Fine-tuned checkpoints from
training/wasb/train_wasb.py carry their input size and resize mode, and run as they were trained.

Preprocessing follows the WASB repo exactly: RGB, the frame scaled to fit 512x288 and centred
(letterboxed, not stretched), ImageNet mean/std normalisation; 3 frames in, 3 heatmaps out
(sigmoid), run on non-overlapping triplets ("step 3", the paper's default).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_CKPT = "models/wasb/wasb_volleyball_best.pth.tar"
TAG = "wasb"
IN_W, IN_H = 512, 288
FRAMES = 3
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# configs/model/wasb.yaml from the WASB repo
CFG = {
    "name": "hrnet", "frames_in": 3, "frames_out": 3, "inp_height": 288, "inp_width": 512,
    "out_height": 288, "out_width": 512, "rgb_diff": False, "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [1],
                       "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2],
                       "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2],
                       "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2, 2],
                       "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": "SUM"},
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}


class _Cfg(dict):
    """dict with attribute access (the vendored HRNet reads its config both ways, as OmegaConf allows)."""
    def __getattr__(self, k):
        v = self[k]
        return _Cfg(v) if isinstance(v, dict) else v

    def __getitem__(self, k):
        v = dict.__getitem__(self, k)
        return _Cfg(v) if isinstance(v, dict) and not isinstance(v, _Cfg) else v


def load_model(ckpt: str | None = DEFAULT_CKPT):
    """The WASB HRNet on the best device; weights from ckpt (None: random, for tests)."""
    import torch
    from core.tracknet import _device
    from core.wasb_hrnet import HRNet
    dev = _device()
    m = HRNet(_Cfg(CFG))
    m.vv = {"input_wh": [IN_W, IN_H], "resize": "letterbox"}     # the authors' preprocessing
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and sd.get("vv"):                 # fine-tuned here: run it as it was trained
            m.vv = dict(sd["vv"])
        sd = sd.get("model_state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"WASB checkpoint doesn't match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return m.to(dev).eval(), dev


def affine(w: int, h: int, in_w: int = IN_W, in_h: int = IN_H, resize: str = "letterbox") -> np.ndarray:
    """Frame -> network input. "letterbox" (the authors' weights): scale to fit by the longer side,
    centred. "stretch" (fine-tuned here): resize straight to in_w x in_h."""
    if resize == "stretch":
        return np.array([[in_w / w, 0, 0], [0, in_h / h, 0]], np.float32)
    s = in_w / max(w, h)
    return np.array([[s, 0, in_w / 2 - s * w / 2], [0, s, in_h / 2 - s * h / 2]], np.float32)


def heatmaps(model, dev, video: str, n: int, batch: int = 4, log=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame heatmaps (n, IN_H, IN_W) in [0, 1], and the frame->input affine matrix."""
    import torch
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vv = getattr(model, "vv", {"input_wh": [IN_W, IN_H], "resize": "letterbox"})
    in_w, in_h = vv["input_wh"]
    M = affine(w, h, in_w, in_h, vv.get("resize", "letterbox"))
    frames = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        if vv.get("resize") == "stretch":
            x = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        else:
            x = cv2.warpAffine(img, M, (in_w, in_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        frames.append(x.transpose(2, 0, 1))
    cap.release()
    m = len(frames)
    out = np.zeros((m, in_h, in_w), np.float16)
    starts = list(range(0, max(m - FRAMES, 0) + 1, FRAMES))
    if starts and starts[-1] + FRAMES < m:            # last frames: one more triplet ending at the clip end
        starts.append(m - FRAMES)
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            idx = starts[b:b + batch]
            x = torch.from_numpy(np.stack([np.concatenate(frames[s:s + FRAMES], 0) for s in idx])).to(dev)
            y = torch.sigmoid(model(x)[0]).float().cpu().numpy()          # (B, 3, H, W)
            for s, hm in zip(idx, y):
                out[s:s + FRAMES] = hm
            if log and (b // batch) % 50 == 0:
                log(f"  WASB {min(idx[-1] + FRAMES, m)}/{m} frames  {time.time() - t0:.0f}s")
    return out, M


def candidates(video: str, dets: dict, ckpt: str = DEFAULT_CKPT, cache_path=None, thr: float = 0.1,
               log=print, redo: bool = False) -> list[list] | None:
    """WASB candidates per frame for this clip (cached), in the same format as core.tracknet.candidates."""
    if not ckpt or not Path(ckpt).exists():
        if log:
            log(f"[VolleyVision] WASB weights not found ({ckpt}); see core/wasb.py for where to get them")
        return None
    from core import paths
    from core.tracknet import blobs, ball_sizes, _ckpt_key
    stem = Path(video).stem
    cp = Path(cache_path) if cache_path else paths.tracknet(stem, tag=TAG)
    key = "wasb:" + _ckpt_key(ckpt)
    n = len(dets["frames"])
    if cp.exists() and not redo:
        try:
            c = json.load(open(cp))
            if c.get("key") == key and len(c.get("frames", [])) == n and c.get("thr") == thr:
                return c["frames"]
        except Exception:
            pass
    t0 = time.time()
    model, dev = load_model(ckpt)
    if log:
        log(f"[VolleyVision] WASB on {dev} ({Path(ckpt).name})")
    hms, M = heatmaps(model, dev, video, n, log=log)
    Minv = cv2.invertAffineTransform(M)
    sizes = ball_sizes(dets, dets.get("fps") or 60)
    out = []
    for f in range(n):
        if f >= len(hms):
            out.append([])
            continue
        s = round(sizes[f], 1)
        row = []
        for x, y, p in blobs(hms[f].astype(np.float32), thr):
            X, Y = Minv @ [x, y, 1.0]
            row.append([round(float(X), 1), round(float(Y), 1), s, s, round(p, 4)])
        out.append(row)
    cp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"schema": "vv-tracknet/1", "model": "wasb", "video": str(video), "key": key, "thr": thr,
               "frames": out}, open(cp, "w"), separators=(",", ":"))
    if log:
        log(f"[VolleyVision] WASB: {sum(1 for f in out if f)}/{n} frames with a candidate "
            f"({time.time() - t0:.0f}s)  →  {cp}")
    return out


def main():
    import argparse
    from core import paths
    ap = argparse.ArgumentParser(description="WASB ball candidates for clips (needs the YOLO cache for ball sizes)")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    for v in a.videos:
        stem = Path(v).stem
        dp = paths.dets(stem)
        if not dp.exists():
            print(f"[wasb] {stem}: no YOLO cache ({dp}); run main.py or tools.dump_detections first")
            continue
        candidates(v, json.load(open(dp)), a.ckpt, redo=a.redo)


if __name__ == "__main__":
    main()
