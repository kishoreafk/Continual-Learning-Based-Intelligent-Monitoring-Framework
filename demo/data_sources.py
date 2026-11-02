"""Data sources for the SAGE-Monitor demo.

Two dataset families:

1. ``scripted``  — the repository's synthetic drift/fault/recall/OOD stream
   (build_stream). Ground truth is fully known; supports live injection of
   fault / drift / OOD events at the playhead. This is the primary demo.

2. ``smap``      — real NASA SMAP/MSL telemetry channels downloaded as parquet
   (train/test) plus labelled anomaly intervals from labeled_anomalies.csv.
   Train on the clean train split, stream the labelled test split.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sage_monitor.datasets import make_windows, standardize, synth_multivariate
from sage_monitor.streams import StreamMeta, build_stream

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "smap_msl"

WINDOW = 60
STRIDE = 4


# ----------------------------------------------------------------------
# Scripted scenario
# ----------------------------------------------------------------------
def load_scripted(steps: int = 4200, channels: int = 12, seg_len: int = 600,
                  seed: int = 42) -> Dict:
    """Build the scripted A -> drift -> B -> C -> FAULT -> B-return -> OOD stream."""
    base = synth_multivariate(steps, channels, seed=seed)
    X, Y, meta = build_stream(base, window=WINDOW, stride=STRIDE,
                              seg_len=seg_len, seed=seed)
    first_end = meta.segments[0]["win_end"]
    train_X = X[:first_end][Y[:first_end] == 0]
    return {
        "kind": "scripted",
        "X": X, "Y": Y, "meta": meta,
        "train_X": train_X,
        # pristine copy so live "clean" injection can restore the stream
        "orig_X": X.clone(), "orig_Y": Y.clone(),
        "n_channels": channels,
        "channel_names": [f"ch{i}" for i in range(channels)],
    }


def inject(source: Dict, at_window: int, kind: str, duration: int = 40,
           seed: int = 7) -> Dict:
    """Inject an event into the remaining (unstreamed) part of a scripted stream.

    Modifies source["X"] / source["Y"] from ``at_window`` onward and returns a
    segment record describing the injected range.

    kind:
      fault — sustained step + jitter burst on 3 channels (must NOT be learned)
      drift — coherent affine scale/bias ramp (learnable -> ADAPT path)
      ood   — novel sinusoidal structure across all channels (unlearnable)
      clean — revert the remaining stream to the original scripted data
    """
    X, Y = source["X"], source["Y"]
    n = X.size(0)
    at = max(0, min(at_window, n))
    end = min(n, at + duration)
    if end <= at:
        return {"kind": kind, "win_start": at, "win_end": at, "name": f"{kind}_empty"}

    if kind == "clean":
        X[at:end] = source["orig_X"][at:end]
        Y[at:end] = source["orig_Y"][at:end]
        return {"kind": "clean", "win_start": at, "win_end": end,
                "name": f"clean_restored_{at}_{end}"}

    seg = X[at:end].clone()  # (n_windows, window, channels)
    ch = min(3, X.size(2))
    if kind == "fault":
        # Differential stuck-sensor fault on 3 channels: one sensor stuck
        # HIGH (+3.5 sigma), one stuck LOW (-3.5 sigma), one pure jitter
        # burst (2 sigma). Mean-neutral BY DESIGN — a net offset on only a
        # few channels drags the anomaly head's global affine bias and makes
        # every clean channel exceed (a "widespread novel" signature), while
        # a balanced fault leaves the coherent fit anchored and presents the
        # concentrated component-fault signature the gate is specified for:
        # >= 3 exceeding channels, max aligned-residual z >= 10.5.
        rng = np.random.default_rng(seed)
        jit = torch.from_numpy(rng.standard_normal(
            (end - at, ch)).astype(np.float32)).unsqueeze(1)  # (n,1,ch)
        signs = torch.tensor([1.0, -1.0, 1.0]).view(1, 1, ch)
        amp = torch.tensor([3.5, 3.5, 0.0]).view(1, 1, ch)
        seg[:, :, :ch] = seg[:, :, :ch] + signs * amp + jit * 2.0
        Y[at:end] = 1
    elif kind == "drift":
        ramp = torch.linspace(0, 1, end - at).view(-1, 1, 1)
        seg = seg * (1 + 0.6 * ramp) + 0.4 * ramp
        Y[at:end] = 0
    elif kind == "ood":
        t = torch.arange(end - at, dtype=torch.float32).view(-1, 1, 1)
        freqs = torch.linspace(0.07, 0.17, X.size(2)).view(1, 1, -1)
        novel = 1.3 * torch.sin(2 * np.pi * freqs * t)  # (n,1,c) -> broadcasts
        seg = 0.25 * seg + novel
        Y[at:end] = 0
    else:
        raise ValueError(f"unknown inject kind: {kind}")

    X[at:end] = seg.float()
    return {"kind": kind, "win_start": at, "win_end": end,
            "name": f"injected_{kind}_{at}_{end}"}


# ----------------------------------------------------------------------
# Real SMAP/MSL channels
# ----------------------------------------------------------------------
def _read_anomaly_sequences(channel: str) -> List[Tuple[int, int]]:
    """Parse labelled anomaly intervals for a channel from labeled_anomalies.csv."""
    path = DATA_DIR / "labeled_anomalies.csv"
    if not path.exists():
        return []
    seqs: List[Tuple[int, int]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["chan_id"] == channel:
                seqs.extend(json.loads(row["anomaly_sequences"]))
    return sorted(seqs)


def smap_available() -> List[str]:
    if not DATA_DIR.exists():
        return []
    out = []
    for f in sorted((DATA_DIR / "train").glob("*.parquet")):
        ch = f.stem
        if (DATA_DIR / "test" / f"{ch}.parquet").exists():
            out.append(ch)
    return out


def load_smap(channel: str, window: int = WINDOW, stride: int = STRIDE,
              max_windows: Optional[int] = None) -> Dict:
    """Load one real SMAP/MSL channel: train on clean split, stream labelled test."""
    import pandas as pd

    train_path = DATA_DIR / "train" / f"{channel}.parquet"
    test_path = DATA_DIR / "test" / f"{channel}.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"SMAP channel {channel} not present under {DATA_DIR}. "
            "Download data/smap_msl/{train,test}/<channel>.parquet first.")

    def series(path: Path) -> np.ndarray:
        df = pd.read_parquet(path)
        drop = [c for c in df.columns
                if c.lower() in ("label", "timestamp", "time", "timestep")]
        return df.drop(columns=drop, errors="ignore").to_numpy(np.float32)

    train_raw, test_raw = series(train_path), series(test_path)
    train_raw, test_raw = standardize(train_raw, test_raw)

    # per-step labels from the labelled anomaly intervals
    labels = np.zeros(test_raw.shape[0], dtype=np.int64)
    for s, e in _read_anomaly_sequences(channel):
        labels[s:min(e, len(labels))] = 1

    train_X, _ = make_windows(train_raw, None, window, stride)
    X_test, Y_test = make_windows(test_raw, labels, window, stride)
    if max_windows is not None and X_test.size(0) > max_windows:
        X_test, Y_test = X_test[:max_windows], Y_test[:max_windows]

    # contiguous label runs -> evaluation segments
    segments, start, cur = [], 0, int(Y_test[0]) if Y_test.numel() else 0
    for i in range(1, Y_test.numel() + 1):
        if i == Y_test.numel() or int(Y_test[i]) != cur:
            segments.append({
                "name": f"{'ANOMALY' if cur else 'normal'}_{len(segments) + 1}",
                "kind": "anomaly" if cur else "normal",
                "win_start": start, "win_end": i,
            })
            if i < Y_test.numel():
                start, cur = i, int(Y_test[i])

    return {
        "kind": "smap",
        "channel": channel,
        "X": X_test, "Y": Y_test,
        "meta": StreamMeta(segments=segments),
        "train_X": train_X,
        "n_channels": X_test.size(2),
        "channel_names": (["value"] + [f"cmd{i}" for i in range(X_test.size(2) - 1)]),
        "anomaly_windows": int(Y_test.sum()),
    }


def catalog() -> List[Dict]:
    """Dataset catalogue served by /api/datasets."""
    out = [{
        "id": "scripted",
        "name": "Scripted Drift/Fault/OOD Scenario",
        "kind": "synthetic",
        "description": ("Regime A -> gradual drift -> regime B -> sudden drift -> "
                        "regime C -> sustained FAULT -> regime B recall -> novel OOD. "
                        "Ground-truth labels known; supports live event injection."),
        "channels": 12, "supports_injection": True, "available": True,
        "recommended": True,
    }]
    for ch in smap_available():
        seqs = _read_anomaly_sequences(ch)
        craft = None
        try:
            with open(DATA_DIR / "labeled_anomalies.csv", newline="",
                      encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if row["chan_id"] == ch:
                        craft = row["spacecraft"]
        except OSError:
            pass
        test_steps = 0
        try:
            import pandas as pd
            test_steps = int(len(pd.read_parquet(DATA_DIR / "test" / f"{ch}.parquet")))
        except Exception:
            pass
        out.append({
            "id": f"smap:{ch}",
            "name": f"NASA {craft} channel {ch} (real telemetry)",
            "kind": "real",
            "description": (f"Real {craft} telemetry, {test_steps} test steps, "
                            f"{len(seqs)} labelled anomaly interval(s): {seqs}. "
                            "Train on clean split, stream labelled test split."),
            "channels": 25, "supports_injection": False, "available": True,
            "recommended": False,
        })
    return out
