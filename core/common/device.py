"""Which device YOLO runs on, and how many images per forward pass.

Ultralytics picks an NVIDIA GPU (CUDA) by itself, but never the Apple-silicon GPU (MPS): on a
Mac it silently runs on the CPU. So we pick the device ourselves:
CUDA if present, else MPS, else CPU. Override with the environment variables
    VV_DEVICE=cpu|mps|cuda|0      VV_BATCH=8
"""
from __future__ import annotations

import os

BATCH = int(os.environ.get("VV_BATCH", 8))


def yolo_device() -> str:
    d = os.environ.get("VV_DEVICE")
    if d:
        return d
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_fallback = {"used": False}


def predict(model, images: list, device: str | None = None, **kw):
    """model.predict on a list of images in one batch, on the chosen device. If the GPU path
    fails (e.g. an op MPS doesn't support), retry on the CPU and stay there for this run."""
    dev = "cpu" if _fallback["used"] else (device or yolo_device())
    try:
        return model.predict(images, device=dev, verbose=False, **kw)
    except Exception as e:
        if dev == "cpu":
            raise
        print(f"[VolleyVision] YOLO on {dev} failed ({type(e).__name__}: {str(e)[:80]}); using the CPU instead")
        _fallback["used"] = True
        return model.predict(images, device="cpu", verbose=False, **kw)
