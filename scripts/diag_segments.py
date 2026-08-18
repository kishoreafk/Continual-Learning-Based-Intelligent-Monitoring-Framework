"""Per-window diagnostic replay: dump anomaly / action / matched regime by segment.

Usage: python -m scripts.diag_segments
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

torch.manual_seed(42)
np.random.seed(42)

cfg = SAGEConfig(n_channels=8, window=40, train_epochs=3, device="cpu",
                 gate_min_obs=32, seed=42)
base = synth_multivariate(3600, 8, seed=42)
X, Y, meta = build_stream(base, window=40, stride=6, seg_len=450, seed=42)
first_end = meta.segments[0]["win_end"]
train_X = X[:first_end][Y[:first_end] == 0]

ctrl = SAGEController(cfg)
ctrl.train_base(train_X, verbose=False)
records = ctrl.run_stream(X, labels=Y, log_every=0)

print("== regime table ==")
for r in ctrl.memory.regimes:
    print(f"  regime {r.regime_id}: n_seen={r.n_seen} windows={len(r.windows)} "
          f"band=(mu={r.err_mu:.4f} sd={r.err_sd:.4f}) thr={r.threshold:.4f}")

print("\n== per-segment decision trace ==")
for s in meta.segments:
    ws, we, name = s["win_start"], s["win_end"], s["name"]
    print(f"\n--- {name}  [{ws}..{we}) ---")
    for i in range(ws, min(we, ws + 120), 6):  # sample every 6th window
        r = records[i]
        print(f"  t={i:3d} a={r['action']:9s} anom={r['anomaly']:.3f} "
              f"drift={r['drift']:.3f} sim={r['regime_sim']:.3f} "
              f"exc={r['exceed_count']} :: {r['reason']}")
