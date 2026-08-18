"""Band-selection diagnostic on the DEFAULT geometry (12ch / 60w / stride 4).

Dumps, for B_normal and B_return: the regime table, the window's own
reconstruction error, every regime band and whether it is band-compatible,
the band actually chosen by _score_window (None = global), the current global
calibration stats, and the resulting anomaly score. This shows exactly why B
scores saturate at 1.0.

Usage: python -m scripts.diag_band
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

cfg = SAGEConfig(n_channels=12, window=60, train_epochs=6, device="cpu",
                 gate_min_obs=32, seed=42)
base = synth_multivariate(4200, 12, seed=42)
X, Y, meta = build_stream(base, window=60, stride=4, seg_len=600, seed=42)
first_end = meta.segments[0]["win_end"]
train_X = X[:first_end][Y[:first_end] == 0]

ctrl = SAGEController(cfg)
ctrl.train_base(train_X, verbose=False)


def snapshot(i):
    """Replay scoring for window i with full band-selection visibility."""
    x = X[i].unsqueeze(0).float()
    with torch.no_grad():
        out_s = ctrl.encoder(x)
        z = out_s["z"]
        err = float(ctrl.anomaly_head.window_error(x, out_s["recon"]).mean().item())
    errs = ctrl.anomaly_head._errors
    gm = float(np.mean(errs)) if len(errs) >= 8 else None
    gs = float(np.std(errs)) if len(errs) >= 8 else None
    rows = []
    chosen = None
    best_sim = -1.0
    for r in ctrl.memory.regimes:
        if r.err_mu is None:
            rows.append((r.regime_id, None, None, None, None))
            continue
        sim_r = ctrl.memory._similarity(z, r.prototype)
        sd_eff = max(r.err_sd, 0.15 * r.err_mu)
        fit = abs(err - r.err_mu) <= 2.0 * sd_eff
        compat = sim_r >= cfg.regime_recall_similarity and fit
        rows.append((r.regime_id, r.err_mu, r.err_sd, round(sim_r, 3), compat))
        if compat and sim_r > best_sim:
            chosen = (r.err_mu, r.err_sd)
            best_sim = sim_r
    with torch.no_grad():
        anom = float(ctrl.anomaly_head(z, x, out_s["recon"], band=chosen).mean().item())
    return err, rows, chosen, (gm, gs, len(errs)), anom


records = ctrl.run_stream(X, labels=Y, log_every=0)

print("== regime table ==")
for r in ctrl.memory.regimes:
    print(f"  regime {r.regime_id}: n_seen={r.n_seen} windows={len(r.windows)} "
          f"band=(mu={r.err_mu if r.err_mu else 'None'} "
          f"sd={r.err_sd if r.err_sd else 'None'}) thr={r.threshold:.4f}")

for name in ("B_normal", "B_return", "C_normal", "FAULT"):
    seg = next(s for s in meta.segments if s["name"] == name)
    ws, we = seg["win_start"], seg["win_end"]
    print(f"\n--- {name}  [{ws}..{we}) ---")
    print("  idx  err     chosen band            global (mu,sd,n)   anom")
    for i in range(ws, min(we, ws + 240), 12):
        err, rows, chosen, glob, anom = snapshot(i)
        ch = ("None" if chosen is None
              else f"({chosen[0]:.3f},{chosen[1]:.3f})")
        gm, gs, n = glob
        gl = "None" if gm is None else f"({gm:.3f},{gs:.3f},{n})"
        bands = " ".join(
            f"R{rid}:{'' if mu is None else f'{mu:.3f}±{sd:.3f}'}"
            f"{'<+' if c else ''}" for rid, mu, sd, _, c in rows)
        print(f"  {i:4d} {err:.3f}  {ch:20s} {gl:18s} {anom:.3f}  [{bands}]")
    # action histogram over the segment
    acts = {}
    for i in range(ws, we):
        acts[records[i]["action"]] = acts.get(records[i]["action"], 0) + 1
    print(f"  actions: {acts}")
    # reason histogram + exceedance distribution
    reasons = {}
    excs = {}
    for i in range(ws, we):
        rr = records[i]["reason"]
        reasons[rr] = reasons.get(rr, 0) + 1
        ec = records[i]["exceed_count"]
        excs[ec] = excs.get(ec, 0) + 1
    print(f"  reasons: {reasons}")
    print(f"  exceed_count dist: {dict(sorted(excs.items()))}")
