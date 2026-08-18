"""List every ADAPT/refine/trial event with its window index and fault label."""
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

# instrument adaptation outcomes
orig_try = ctrl._try_adapt
orig_refine = ctrl._refine_young_regime
events = []

def try_adapt(x, s):
    tail = ctrl._recent_exceed[-ctrl.cfg.coherent_shift_persist:]
    events.append(("ADAPT_PROBE", ctrl._i, f"exc_tail={tail}"))
    ok = orig_try(x, s)
    events.append(("ADAPT_RESULT", ctrl._i, "committed" if ok else "FAILED"))
    return ok

def refine(regime, x):
    events.append(("REFINE_IN", ctrl._i, f"regime={regime.regime_id}"))
    orig_refine(regime, x)
    events.append(("REFINE_OUT", ctrl._i, f"band=({regime.err_mu:.3f},{regime.err_sd:.3f})"))

ctrl._try_adapt = try_adapt
ctrl._refine_young_regime = refine

ctrl._i = 0
records = []
for i in range(X.size(0)):
    ctrl._i = i
    rec = ctrl.step(X[i], true_fault=bool(Y[i].item()))
    records.append(rec)

for name, t, extra in events:
    print(f"t={t:3d} {name:12s} {extra}")

print("\nunsafe adaptation check:")
for rec, y in zip(records, Y.numpy()):
    if rec["action"] == "ADAPT" and y == 1:
        print(f"  t={rec.get('t','?')} ADAPT on fault-labeled window!")

print("\nregime table:")
for r in ctrl.memory.regimes:
    print(f"  id={r.regime_id} n_seen={r.n_seen} band=({r.err_mu:.3f},{r.err_sd:.3f})")
