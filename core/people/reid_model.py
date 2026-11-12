"""Person look vectors from a pretrained re-identification model (LightMBN, Market-1501), used by
core.people.reid to tell players apart. Optional: without the package or the weights, nothing
breaks; the joiner uses colours only.

    pip install boxmot
    curl -L -o models/lmbn_n_market.pt \\
        https://github.com/mikel-brostrom/boxmot/releases/download/v21.0.0/lmbn_n_market.pt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "lmbn_n_market.pt"


def load_embedder():
    """embed(img_bgr, boxes (n, 4) xyxy) -> (n, d) unit vectors; None if the model isn't there."""
    if not WEIGHTS.exists():
        return None
    try:
        import torch
        from boxmot.reid import create_reid_encoder
        from boxmot.structures.detections import Detections
        from boxmot.structures.frame import Frame
        from boxmot.structures.geometry import Boxes
    except ImportError:
        return None
    enc = create_reid_encoder(WEIGHTS, device="cpu", allow_download=False)

    def embed(img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if not len(boxes):
            return np.zeros((0, enc.embedding_dim), np.float32)
        rgb = torch.from_numpy(np.ascontiguousarray(img_bgr[..., ::-1].transpose(2, 0, 1)))
        det = Detections(geometry=Boxes(torch.tensor(np.asarray(boxes, np.float32))),
                         scores=torch.ones(len(boxes)), class_ids=torch.zeros(len(boxes), dtype=torch.int64),
                         sample_id="f")
        e = enc.encode([Frame(image=rgb, sample_id="f")], [det])[0].numpy().astype(np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
    return embed
