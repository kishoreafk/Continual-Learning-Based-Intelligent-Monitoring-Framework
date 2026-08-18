"""Dataset loading and windowing utilities.

Primary benchmark: NASA SMAP/MSL telemetry (real spacecraft/rover data).
The Colab notebook downloads `train.npy`, `test.npy` and `labeled_anomalies.csv`
from Kaggle or NASA's telemanom repository and converts them to parquet.
"""

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


def load_channel_series(parquet_dir: str, channel: str) -> np.ndarray:
    """Load a single telemetry channel time series from the parquet dump."""
    path = os.path.join(parquet_dir, f"{channel}.parquet")
    df = pd.read_parquet(path)
    values = df.drop(columns=[c for c in df.columns if c in ("label", "timestamp", "time")],
                     errors="ignore").to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy().astype(np.int64) if "label" in df.columns else None
    return values, labels


def make_windows(values: np.ndarray, labels: Optional[np.ndarray],
                 window: int, stride: int = 1,
                 horizon: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sliding windows with a per-window anomaly label (any-step anomaly)."""
    n = values.shape[0]
    W = []
    L = []
    for start in range(0, n - window - horizon + 1, stride):
        W.append(values[start:start + window])
        if labels is not None:
            seg = labels[start:start + window]
            L.append(int(seg.max()))
    X = torch.from_numpy(np.stack(W)).float()
    Y = torch.tensor(L, dtype=torch.long) if labels is not None else None
    return X, Y


def split_train_test(labels: Optional[np.ndarray], ratio: float = 0.7):
    """Chronological split; if labels exist, keep the split on a clean boundary."""
    if labels is None:
        return ratio
    # find the split point closest to `ratio` that is not inside an anomaly run
    n = len(labels)
    target = int(n * ratio)
    best = target
    for off in range(0, n // 4):
        for cand in (target - off, target + off):
            if 0 < cand < n and labels[cand] == 0 and labels[cand - 1] == 0:
                return cand / n
    return best / n


def standardize(train: np.ndarray, test: np.ndarray):
    """Fit on train only, transform both (no test leakage)."""
    mu = train.mean(axis=0, keepdims=True)
    sd = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mu) / sd, (test - mu) / sd


def synth_multivariate(n_steps: int, n_channels: int, seed: int = 0):
    """Synthetic but realistic telemetry: smooth channels + shared regime driver."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    base = np.zeros((n_steps, n_channels), dtype=np.float32)
    for c in range(n_channels):
        freq = 0.01 + 0.02 * rng.random()
        phase = 2 * np.pi * rng.random()
        amp = 0.5 + rng.random()
        trend = 1e-4 * (rng.random() - 0.5) * t
        noise = 0.05 * rng.standard_normal(n_steps)
        base[:, c] = amp * np.sin(2 * np.pi * freq * t + phase) + trend + noise
    return base.astype(np.float32)


def inject_regime_shift(values: np.ndarray, start: int, end: int, scale: float,
                        bias: float):
    out = values.copy()
    out[start:end] = out[start:end] * scale + bias
    return out


def inject_fault(values: np.ndarray, start: int, end: int, channels=None,
                 spike: float = 4.0):
    """Spike/breakdown fault on a subset of channels."""
    out = values.copy()
    ch = channels or list(range(min(3, values.shape[1])))
    n = end - start
    ramp = np.linspace(0, 1, n)[:, None]
    out[start:end, ch] += spike * ramp
    out[start:end, ch] += 0.5 * spike * np.random.RandomState(0).standard_normal((n, len(ch)))
    return out
