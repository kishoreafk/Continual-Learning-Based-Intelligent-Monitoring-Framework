"""Local end-to-end run of SAGE-Monitor on a synthetic drift/fault stream.

Usage (from the project root):
    python -m scripts.run_local --steps 4200 --window 60 --stride 4

Runs: base training -> streaming through A -> drift -> B -> fault -> B-return
-> novel OOD, then prints the full metric report. All outputs are written to
the project's artifacts/ directory.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Fixed project layout: this script lives in <project>/scripts and writes
# only to <project>/artifacts.
PROJECT_ROOT = Path("E:/Continual Monitoring Framework")
ARTIFACTS = PROJECT_ROOT / "artifacts"
REPORT_PATH = ARTIFACTS / "run_report.json"
MODEL_PATH = ARTIFACTS / "model.pt"
sys.path.insert(0, str(PROJECT_ROOT))

from sage_monitor.config import SAGEConfig
from sage_monitor.controller import SAGEController
from sage_monitor.datasets import synth_multivariate
from sage_monitor.streams import build_stream
from sage_monitor import metrics as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4200)
    ap.add_argument("--channels", type=int, default=12)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seg-len", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = SAGEConfig(
        n_channels=args.channels, window=args.window,
        train_epochs=args.epochs, device=args.device,
        gate_min_obs=32, seed=args.seed,
    )

    print("== building synthetic stream ==")
    base = synth_multivariate(args.steps, args.channels, seed=args.seed)
    X, Y, meta = build_stream(base, window=args.window, stride=args.stride,
                              seg_len=args.seg_len, seed=args.seed)
    print(f"   windows={X.size(0)} shape={tuple(X.shape)} fault_windows={int(Y.sum())}")
    for s in meta.segments:
        print(f"   [{s['win_start']:4d}..{s['win_end']:4d}] {s['name']} ({s['kind']})")

    # base training on the first clean regime only (chronological discipline)
    first_end = meta.segments[0]["win_end"]
    train_X = X[:first_end][Y[:first_end] == 0]

    print("== training base model on regime A ==")
    ctrl = SAGEController(cfg)
    ctrl.train_base(train_X, verbose=True)

    print("== streaming ==")
    records = ctrl.run_stream(X, labels=Y, log_every=100)

    print("== evaluating ==")
    report = M.summarise(ctrl, records, meta.segments, Y.numpy())

    print("\n--- detection ---")
    for k, v in report["detection"].items():
        print(f"  {k}: {v}")
    print("--- point-adjust ---")
    for k in ("precision", "recall", "f1"):
        print(f"  {k}: {report['detection_point_adjust'][k]:.3f}")
    print(f"  missed intervals: {report['missed']}  delays: {report['delays']}")
    print(f"  false alarm rate: {report['false_alarm_rate']:.4f}")
    print("--- continual / safety ---")
    print(f"  regimes discovered: {report['n_regimes']}")
    print(f"  adaptations: {report['stats']['adaptations']} "
          f"(unsafe: {report['stats']['unsafe_adaptations']})")
    print(f"  recalls: {report['stats']['recalls']}  "
          f"rejects: {report['stats']['rejects']}  alerts: {report['stats']['alerts']}")
    print(f"  unsafe adaptation rate: {report['unsafe_adaptation_rate']:.3f}")
    print("--- segment anomaly scores ---")
    for name, score in report["segment_anomaly_score"].items():
        print(f"  {name}: {score:.3f}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=float),
                           encoding="utf-8")
    print(f"\nreport saved -> {REPORT_PATH}")

    ctrl.save(str(MODEL_PATH))
    print(f"model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
