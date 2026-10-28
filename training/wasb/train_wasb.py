"""Fine-tune WASB (core/wasb_hrnet.py, MIT) on a TrackNet-format export of your checked clips.

    python training/wasb/train_wasb.py --data /content/data/tracknet_v2 --init wasb_volleyball_best.pth.tar \
        --out /content/drive/MyDrive/VolleyVision/wasb_ft_v2 --epochs 15
    ... --resume            # continue from <out>/last.pth.tar after a disconnect

Data: the TrackNet export from tools.export_training (train|val/match<k>/frame/<rally>/<i>.jpg and
csv/<rally>_ball.csv with Frame,Visibility,X,Y in the stored frame's pixels). Old (512x288) and
new (e.g. 640x288) frames can be mixed: every frame is resized to the training input size.

Recipe (the WASB paper / repo, adapted for fine-tuning):
  3 consecutive frames in, 3 heatmaps out; ImageNet normalisation
  target: binary disk of radius 2.5 px at the ball centre (none if not visible)
  loss: quality focal loss (beta 2) on the sigmoid output
  Adam, lr 1e-4 (1/10 of training from scratch), cosine decay; batch 8
  augmentation: colour jitter (p 0.5, +-10%), horizontal flip (p 0.5; a court mirrors fine)
  input 640x288 by default: your clips' shape, same 288 px height WASB was trained at

Fewer false balls (optional; WASB fine-tuned on 16 clips fired on heads, hands and fixtures in
~15% of no-ball frames, because ~85% of training windows contain a ball):
  --neg-weight W     windows with no visible ball are drawn W times as often
  --mine-from E      hard-negative mining (as in the WASB paper): from epoch E on, before each
                     epoch, find the training windows where the model fires on something that
                     isn't the ball (no-ball frame, or >8 px from it) and draw them --hard-weight
                     times as often in that epoch

Each epoch: validation F1 (strongest blob >= 0.5 within 4 px of the label, as in the paper),
<out>/last.pth.tar (for --resume) and <out>/best.pth.tar (best val F1). Checkpoints carry
"vv": {"input_wh": [W, H], "resize": "stretch"}, so core.wasb runs them the way they were trained.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve()
for p in (HERE.parents[2], HERE.parent, Path.cwd()):       # find core/wasb_hrnet.py (repo or next to this file)
    sys.path.insert(0, str(p))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FRAMES = 3
RADIUS = 2.5


# ── data ─────────────────────────────────────────────────────────────────────

def read_rallies(root: Path) -> list[list[tuple[str, tuple[float, float] | None]]]:
    """[[(jpg path, (x, y) in stored-frame px or None), ...] per rally] under root/match*/."""
    out = []
    for m in sorted(root.glob("match*")):
        for c in sorted((m / "csv").glob("*_ball.csv")):
            rid = c.name[:-len("_ball.csv")]
            rows = list(csv.DictReader(open(c)))
            run = []
            for r in rows:
                img = m / "frame" / rid / f"{int(r['Frame'])}.jpg"
                if not img.exists():
                    break
                vis = int(float(r["Visibility"])) > 0
                run.append((str(img), (float(r["X"]), float(r["Y"])) if vis else None))
            if len(run) >= FRAMES:
                out.append(run)
    return out


class Triplets(Dataset):
    def __init__(self, rallies, wh, train: bool, stride: int = 1, zoom_prob: float = 0.0,
                 zoom_range: tuple = (1.5, 3.0)):
        self.wh, self.train = wh, train
        self.zoom_prob, self.zoom_range = zoom_prob, zoom_range
        self.items = [(r, i) for r in rallies for i in range(0, len(r) - FRAMES + 1, stride)]
        W, H = wh
        yy, xx = np.mgrid[0:H, 0:W]
        self.grid = (xx.astype(np.float32), yy.astype(np.float32))

    def __len__(self):
        return len(self.items)

    def ball_free(self) -> list[bool]:
        """Per window: True if none of its frames shows the ball."""
        return [all(xy is None for _, xy in run[i:i + FRAMES]) for run, i in self.items]

    def __getitem__(self, k):
        run, i = self.items[k]
        W, H = self.wh
        flip = self.train and random.random() < 0.5
        jitter = None
        if self.train and random.random() < 0.5:                       # same jitter for all 3 frames
            jitter = [random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1),
                      random.uniform(-0.1, 0.1)]
        # zoom augmentation: the same zoomed-in crop for all 3 frames, so the ball appears
        # 1.5-3x bigger (near-camera balls are rare in the data and WASB misses them)
        box = None
        if self.train and self.zoom_prob > 0 and random.random() < self.zoom_prob:
            box = random.uniform(*self.zoom_range)
        imgs, hms, xys = [], [], []
        for j, (path, xy) in enumerate(run[i:i + FRAMES]):
            img = cv2.imread(path)
            h0, w0 = img.shape[:2]
            if isinstance(box, float):                     # pick the crop once, from the first frame's size
                z = box
                cw, ch = int(w0 / z), int(h0 / z)
                mid = run[i + FRAMES // 2][1]
                if mid is not None and random.random() < 0.8:  # mostly around the ball, anywhere in the crop
                    cx0 = mid[0] - random.uniform(0.15, 0.85) * cw
                    cy0 = mid[1] - random.uniform(0.15, 0.85) * ch
                else:
                    cx0, cy0 = random.uniform(0, w0 - cw), random.uniform(0, h0 - ch)
                box = (int(min(max(cx0, 0), w0 - cw)), int(min(max(cy0, 0), h0 - ch)), cw, ch)
            if box is not None:
                x0, y0, cw, ch = box
                img = img[y0:y0 + ch, x0:x0 + cw]
                if xy is not None:
                    xy = (xy[0] - x0, xy[1] - y0)
                    if not (0 <= xy[0] < cw and 0 <= xy[1] < ch):
                        xy = None                          # the ball is outside the crop
                h0, w0 = ch, cw
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if jitter is not None:
                b, c, s, hue = jitter
                img = np.clip(img * b, 0, 1)
                m = img.mean()
                img = np.clip((img - m) * c + m, 0, 1)
                g = img.mean(2, keepdims=True)
                img = np.clip((img - g) * s + g, 0, 1)
                if abs(hue) > 1e-3:
                    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.int16)
                    hsv[..., 0] = (hsv[..., 0] + int(hue * 180)) % 180
                    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            if flip:
                img = img[:, ::-1]
            imgs.append(((img - MEAN) / STD).transpose(2, 0, 1))
            hm = np.zeros((H, W), np.float32)
            if xy is not None:
                cx, cy = xy[0] * W / w0, xy[1] * H / h0
                if flip:
                    cx = W - 1 - cx
                cxi, cyi = int(cx), int(cy)                              # as WASB: integer centre, disk r=2.5
                hm[(self.grid[0] - cxi) ** 2 + (self.grid[1] - cyi) ** 2 <= RADIUS ** 2] = 1.0
                xys.append((cx, cy))
            else:
                xys.append((-1.0, -1.0))
            hms.append(hm)
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate(imgs, 0)))
        return x, torch.from_numpy(np.stack(hms)), torch.tensor(xys, dtype=torch.float32)


# ── model, loss, metric ──────────────────────────────────────────────────────

def quality_focal_loss(logits, target, beta: float = 2.0):
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    loss = (p - target).abs() ** beta * ((1 - target) * torch.log(1 - p) + target * torch.log(p))
    return -loss.mean()


def top_blob(hm: np.ndarray, thr: float = 0.5):
    m = (hm >= thr).astype(np.uint8)
    k, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    best = None
    for j in range(1, k):
        ys, xs = np.nonzero(lab == j)
        w = hm[ys, xs]
        if best is None or w.max() > best[2]:
            best = ((xs * w).sum() / w.sum(), (ys * w).sum() / w.sum(), w.max())
    return best


def evaluate(model, loader, dev, dist: float = 4.0) -> dict:
    tp = fp = fn = tn = fp_nb = 0
    model.eval()
    with torch.no_grad():
        for x, _, xys in loader:
            hm = torch.sigmoid(model(x.to(dev))[0]).cpu().numpy()
            for b in range(hm.shape[0]):
                for t in range(FRAMES):
                    pred = top_blob(hm[b, t])
                    gx, gy = xys[b, t].tolist()
                    if gx < 0:
                        fp += pred is not None
                        fp_nb += pred is not None
                        tn += pred is None
                    elif pred is None:
                        fn += 1
                    elif math.hypot(pred[0] - gx, pred[1] - gy) <= dist:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "false_on_no_ball": fp_nb / max(fp_nb + tn, 1), "no_ball_frames": fp_nb + tn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def build_model(init: str | None, dev):
    from core.wasb import CFG, _Cfg
    from core.wasb_hrnet import HRNet
    m = HRNet(_Cfg(CFG))
    if init:
        sd = torch.load(init, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        m.load_state_dict(sd)
    return m.to(dev)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="TrackNet export folder (with train/ and val/)")
    ap.add_argument("--init", default=None, help="starting weights (WASB volleyball checkpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--train-stride", type=int, default=1, help="1 = every window of 3 consecutive frames")
    ap.add_argument("--max-steps", type=int, default=0, help="stop each epoch after this many steps (smoke test)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (on by default on CUDA)")
    ap.add_argument("--neg-weight", type=float, default=1.0, help="draw ball-free windows this many times as often")
    ap.add_argument("--mine-from", type=int, default=0, help="hard-negative mining from this epoch on (0 = off)")
    ap.add_argument("--hard-weight", type=float, default=3.0, help="draw mined hard negatives this many times as often")
    ap.add_argument("--mine-every", type=int, default=1, help="re-mine hard negatives every N epochs (mining costs ~half an epoch)")
    ap.add_argument("--zoom-prob", type=float, default=0.0,
                    help="share of training windows zoomed in 1.5-3x (bigger balls; 0.3 recommended)")
    a = ap.parse_args()
    assert a.width % 8 == 0 and a.height % 8 == 0, "width/height must be multiples of 8"
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wh = (a.width, a.height)
    tr, va = read_rallies(root / "train"), read_rallies(root / "val")
    dtr, dva = Triplets(tr, wh, True, a.train_stride, zoom_prob=a.zoom_prob), Triplets(va, wh, False, FRAMES)
    if a.zoom_prob:
        print(f"zoom augmentation: {a.zoom_prob:.0%} of training windows zoomed in 1.5-3x")
    print(f"device {dev} | input {wh[0]}x{wh[1]} | train {len(tr)} rallies, {len(dtr)} windows | "
          f"val {len(va)} rallies, {len(dva)} windows")
    free = np.array(dtr.ball_free())
    base_w = np.where(free, a.neg_weight, 1.0)
    print(f"ball-free training windows: {free.mean():.1%}" + (f" (drawn {a.neg_weight:g}x as often)" if a.neg_weight != 1 else ""))
    dmine = Triplets(tr, wh, False, a.train_stride)            # same windows, no augmentation, for mining

    def train_loader(weights):
        if np.allclose(weights, 1.0):
            return DataLoader(dtr, a.batch, shuffle=True, drop_last=True, num_workers=a.workers,
                              pin_memory=dev.type == "cuda")
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(dtr))
        return DataLoader(dtr, a.batch, sampler=sampler, drop_last=True, num_workers=a.workers,
                          pin_memory=dev.type == "cuda")

    def mine_hard(model) -> np.ndarray:
        """Windows where the model's strongest blob (>= 0.5) is not the ball: on a no-ball frame,
        or more than 8 px (input resolution) from it."""
        hard = np.zeros(len(dmine), bool)
        loader = DataLoader(dmine, a.batch, shuffle=False, num_workers=a.workers)
        model.eval()
        k = 0
        with torch.no_grad():
            for x, _, xys in loader:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    hm = torch.sigmoid(model(x.to(dev))[0]).float().cpu().numpy()
                for b in range(hm.shape[0]):
                    for t in range(FRAMES):
                        pred = top_blob(hm[b, t])
                        if pred is None:
                            continue
                        gx, gy = xys[b, t].tolist()
                        if gx < 0 or math.hypot(pred[0] - gx, pred[1] - gy) > 8:
                            hard[k + b] = True
                            break
                k += hm.shape[0]
        return hard

    lva = DataLoader(dva, a.batch, shuffle=False, num_workers=a.workers)

    model = build_model(None if a.resume else a.init, dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    meta = {"input_wh": [wh[0], wh[1]], "resize": "stretch", "init": Path(a.init).name if a.init else None}
    amp = dev.type == "cuda" and not a.no_amp          # mixed precision: less GPU memory, faster
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best_f1, hist = 0, -1.0, []
    if a.resume:
        ck = torch.load(out / "last.pth.tar", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start, best_f1, hist = ck["epoch"], ck["best_f1"], ck.get("history", [])
        print(f"resumed after epoch {start} (best val F1 {best_f1:.3f})")
    elif a.init:
        r = evaluate(model, lva, dev)
        print(f"before fine-tuning: val F1 {r['f1']:.3f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
              f"false on no-ball frames {r['false_on_no_ball']:.1%} (of {r['no_ball_frames']})")
        hist.append({"epoch": 0, **r})

    hard = None
    for ep in range(start, a.epochs):
        w = base_w
        if a.mine_from and ep + 1 >= a.mine_from:
            t1 = time.time()
            if (ep + 1 - a.mine_from) % max(a.mine_every, 1) == 0 or hard is None:
                hard = mine_hard(model)
            w = base_w * np.where(hard, a.hard_weight, 1.0)
            print(f"  epoch {ep + 1}: mined {hard.sum()} hard negatives ({hard.mean():.1%} of windows, "
                  f"drawn {a.hard_weight:g}x as often)  [{time.time() - t1:.0f}s]", flush=True)
        ltr = train_loader(w)
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, (x, hm, _) in enumerate(ltr):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(x.to(dev, non_blocking=True))[0]
            loss = quality_focal_loss(logits.float(), hm.to(dev, non_blocking=True))   # loss in float32
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item(), n + 1
            if step % 200 == 0:
                print(f"  epoch {ep + 1} step {step}/{len(ltr)}  loss {tot / n:.5f}  [{time.time() - t0:.0f}s]", flush=True)
            if a.max_steps and step + 1 >= a.max_steps:
                break
        sched.step()
        r = evaluate(model, lva, dev)
        hist.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), **r})
        better = r["f1"] > best_f1
        best_f1 = max(best_f1, r["f1"])
        ck = {"model_state_dict": model.state_dict(), "vv": meta, "epoch": ep + 1, "val": r}
        if better:
            torch.save(ck, out / "best.pth.tar")
        torch.save({**ck, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best_f1": best_f1, "history": hist}, out / "last.pth.tar")
        json.dump(hist, open(out / "history.json", "w"), indent=1)
        print(f"epoch {ep + 1}/{a.epochs}: loss {tot / max(n, 1):.5f} | val F1 {r['f1']:.3f} precision "
              f"{r['precision']:.3f} recall {r['recall']:.3f} false-on-no-ball {r['false_on_no_ball']:.1%}"
              f"{'  <- best' if better else ''}  [{time.time() - t0:.0f}s]",
              flush=True)
    print(f"done: best val F1 {best_f1:.3f}  →  {out / 'best.pth.tar'}")


if __name__ == "__main__":
    main()
