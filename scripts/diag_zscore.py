"""Residual-structure probe: per-channel z-scores of the aligned residual.

For B (learnable shift), FAULT (3-channel step+jitter) and novel_OOD, show:
  - exceed_count (# channels with z > exceed_z_thr)
  - max channel z-score in the window
  - energy concentration: fraction of total residual energy on the exceeding channels

If faults concentrate energy on a few channels while shifts spread it, the
energy-concentration or max-z statistic is the geometry-robust discriminator.

Usage: python -m scripts.diag_zscore
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path("E:/Continual Monitoring Framework")
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from sage_monitor.config import SAGEConfig
from sage_monitor.controller import SAGEController
from sage_monitor.datasets import synth_multivariate
from sage_monitor.streams import build_stream
from sage_monitor.heads import AnomalyHead

torch.manual_seed(42)
np.random.seed(42)

cfg = SAGEConfig(n_channels=12, window=60, train_epochs=6, device="cpu",
                 gate_min_obs=32, seed=42)
base = synth_multivariate(4200, 12, seed=42)
X, Y, meta = build_stream(base, window=60, stride=4, seg_len=600, seed=42)
first_end = meta.segments[0]["win_end"]
train_X = X[:first_end][Y[:first_end] == 0]

ctrl = SAGEController(cfg)
ctrl.train_base(train_X, verbose=False)


def zstats(i):
    x = X[i].unsqueeze(0).float()
    with torch.no_grad():
        out = ctrl.encoder(x)
        r = AnomalyHead._aligned_residual(x, out["recon"])  # (1, T, C)
    ch_sd = ctrl.anomaly_head._ch_sd
    if ch_sd is None:
        return None
    z = (r.abs().mean(dim=1) / (ch_sd + 1e-8))[0]  # (C,)
    exc = int((z > cfg.exceed_z_thr).sum().item())
    maxz = float(z.max().item())
    e2 = (r[0] ** 2).sum(dim=0)  # per-channel energy (T,)
    top = e2[z > cfg.exceed_z_thr].sum() / (e2.sum() + 1e-12)
    return exc, maxz, float(top.item())


for name in ("B_normal", "FAULT", "novel_OOD", "C_normal"):
    seg = next(s for s in meta.segments if s["name"] == name)
    ws, we = seg["win_start"], seg["win_end"]
    print(f"\n--- {name} [{ws}..{we}) ---")
    print("  idx   exc  maxz   energy_on_exc_ch")
    for i in range(ws, we, 8):
        st = zstats(i)
        if st is None:
            continue
        exc, maxz, top = st
        print(f"  {i:4d}  {exc:3d}  {maxz:6.1f}  {top:6.2f}")
