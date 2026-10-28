"""Learned probability that a candidate is the ball, per source (YOLO, zoom find, TrackNet).

Raw scores aren't comparable across sources: YOLO 0.3 and a TrackNet peak 0.3 mean very
different things. Each source gets a small logistic model, fit on labelled frames
(tools.calibrate), from its score:

    p(ball | candidate) = sigmoid(w · logit(score) + b)

Monotonic per source: it keeps each source's own ranking and puts all sources on one
probability scale, which is what the joint linker needs to weigh them against each other.
Stored as plain numbers in models/score_calibration.json (no pickles). This is only about
interpreting model scores; clip normalization (time, units) is core.profile, and model
defects (e.g. TrackNet's position offset) are handled next to the model (core.tracknet).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

SOURCES = ("yolo", "zoom", "tracknet")   # + one model per extra finder (e.g. "wasb"), see Calibration.for_finder
DEFAULT_PATH = Path("models/score_calibration.json")


def features(source: str, conf, size_ratio=None, aspect=None) -> np.ndarray:
    """Feature matrix (N, 1): logit of the source's own score (Platt scaling).
    Size and shape features were tried and hurt on held-out clips (ball size varies too
    much across clips and positions), so each source maps its score alone to a probability."""
    c = np.clip(np.asarray(conf, float), 1e-4, 1 - 1e-4)
    return np.log(c / (1 - c))[:, None]


class Calibration:
    def __init__(self, models: dict):
        self.models = models            # source -> {"coef": [...], "intercept": b}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PATH) -> "Calibration":
        return cls(json.load(open(path))["models"])

    def save(self, path: str | Path = DEFAULT_PATH, meta: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"schema": "vv-calibration/1", "models": self.models, **(meta or {})}, open(path, "w"), indent=1)

    def logit(self, source: str, conf, size_ratio=None, aspect=None) -> np.ndarray:
        m = self.models[source]
        X = features(source, conf, size_ratio, aspect)
        return X @ np.asarray(m["coef"], float) + float(m["intercept"])

    def prob(self, source: str, conf, size_ratio=None, aspect=None) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(source, conf, size_ratio, aspect)))

    def for_finder(self, name: str) -> "Calibration":
        """The same calibration with the finder slot ("tracknet" in the linkers) scored by the
        model fitted for finder `name` (e.g. "wasb"). Unchanged if name is "tracknet"."""
        if name == "tracknet":
            return self
        if name not in self.models:
            raise KeyError(f"no score calibration for finder '{name}': run  python -m tools.calibrate --finder <ckpt>")
        return Calibration({**self.models, "tracknet": self.models[name]})


def fit(samples: dict, C: float = 1.0) -> Calibration:
    """samples[source] = (X, y) with X from features(); returns a fitted Calibration."""
    from sklearn.linear_model import LogisticRegression
    models = {}
    for s, (X, y) in samples.items():
        if len(y) == 0 or y.min() == y.max():
            models[s] = {"coef": [1.0], "intercept": -2.0, "n": int(len(y))}
            continue
        lr = LogisticRegression(C=C, max_iter=2000).fit(X, y)
        models[s] = {"coef": [float(v) for v in lr.coef_[0]], "intercept": float(lr.intercept_[0]),
                     "n": int(len(y)), "pos": int(y.sum())}
    return Calibration(models)
