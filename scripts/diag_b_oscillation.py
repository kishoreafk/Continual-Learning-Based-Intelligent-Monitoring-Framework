"""Targeted probe: why does B's anomaly oscillate after verified recalls?

Replays the stream to t=178 and, at selected B windows, prints the error of
the SAME window under each regime's adapter, scored against each band and the
current global band. Also shows which regime _verify_return_best picks.
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

PROBE_AT = {141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
            158, 159, 160, 161, 165, 170, 171, 172, 176, 177}

for i in range(X.size(0)):
    if i > 178:
        break
    x = X[i].float().unsqueeze(0)
    rec = ctrl.step(x, true_fault=bool(Y[i].item()))
    if i in PROBE_AT:
        buf = ctrl.anomaly_head._errors
        mu = float(np.mean(buf)) if len(buf) >= 8 else None
        sd = float(np.std(buf)) if len(buf) >= 8 else None
        line = (f"t={i:3d} {rec['action']:9s} anom={rec['anomaly']:.3f} "
                f"reason={rec['reason']!r} | buf n={len(buf)} mu={mu} sd={sd}")
        print(line)
        # error of this window under each regime's adapter + band
        with torch.no_grad():
            saved = ctrl.encoder.get_adapter_state()
            for r in ctrl.memory.regimes:
                if not r.adapter_state:
                    print(f"    regime {r.regime_id}: no adapter")
                    continue
                ctrl.encoder.set_adapter_state(r.adapter_state)
                out = ctrl.encoder(x)
                err = float(ctrl.anomaly_head.window_error(x, out["recon"]).mean())
                s_band = ctrl.anomaly_head.score_errors(
                    torch.tensor([err]), r.err_mu, r.err_sd).item()
                s_glob = (ctrl.anomaly_head.score_errors(
                    torch.tensor([err]), mu, sd).item() if mu else None)
                print(f"    regime {r.regime_id} (n={r.n_seen}, band={r.err_mu:.3f}/{r.err_sd:.3f}): "
                      f"err={err:.4f} score_own_band={s_band:.3f} score_global={s_glob}")
            # which regime does verify-return-best pick?
            s = ctrl._score_window(x)
            vrb = ctrl._verify_return_best(s["z"], x)
            print(f"    verify_return_best -> {vrb.regime_id if vrb else None} "
                  f"(nearest sim={s['sim']:.3f}, nearest={s['regime'].regime_id})")
            ctrl.encoder.set_adapter_state(saved)
