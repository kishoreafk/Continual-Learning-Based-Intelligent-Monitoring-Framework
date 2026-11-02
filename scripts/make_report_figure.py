"""Render the SAGE-Monitor decision-stream figure for the demo artifact pack.

Re-runs the deterministic scripted pipeline (same seed as run_local) and
saves artifacts/decision_stream.png with ground truth, scores and gate
actions aligned on one timeline.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

for f in Path("/usr/share/fonts/truetype/chinese").glob("*.ttf"):
    try:
        fm.fontManager.addfont(str(f))
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

from sage_monitor.config import SAGEConfig          # noqa: E402
from sage_monitor.controller import SAGEController  # noqa: E402
from sage_monitor.datasets import synth_multivariate  # noqa: E402
from sage_monitor.streams import build_stream        # noqa: E402
from sage_monitor import metrics as M                # noqa: E402

OUT = ROOT / "artifacts" / "decision_stream.png"

ACT_COLORS = {"NO_ACTION": "#c8cdd3", "ADAPT": "#1f77b4", "RECALL": "#2ca02c",
              "REJECT": "#ff7f0e", "ALERT": "#d62728"}


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    base = synth_multivariate(4200, 12, seed=42)
    X, Y, meta = build_stream(base, window=60, stride=4, seg_len=600, seed=42)
    first_end = meta.segments[0]["win_end"]
    train_X = X[:first_end][Y[:first_end] == 0]
    cfg = SAGEConfig(n_channels=12, window=60, train_epochs=6,
                     gate_min_obs=32, device="cpu", seed=42)
    ctrl = SAGEController(cfg)
    ctrl.train_base(train_X, verbose=False)
    records = ctrl.run_stream(X, labels=Y, log_every=0)
    rep = M.summarise(ctrl, records, meta.segments, Y.numpy())
    print("detection:", {k: round(v, 3) for k, v in rep["detection"].items()})
    print("pa:", {k: round(v, 3) for k, v in rep["detection_point_adjust"].items()})

    t = np.arange(len(records))
    anomaly = np.array([r["anomaly"] for r in records])
    drift = np.array([r["drift"] for r in records])
    ood = np.array([r["ood"] for r in records])
    labels = Y.numpy()

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.5), sharex=True,
                             constrained_layout=True)
    fig.suptitle("SAGE-Monitor — scripted drift / fault / recall / OOD stream "
                 "(12 channels, seed 42)", fontsize=13, fontweight="bold")

    # 1. ground truth
    ax = axes[0]
    ax.fill_between(t, 0, labels, step="mid", color="#d62728", alpha=0.55,
                    label="fault ground truth")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_ylabel("fault label")
    for s in meta.segments:
        if s["kind"] != "normal":
            ax.axvspan(s["win_start"], s["win_end"], alpha=0.10,
                       color={"drift": "#1f77b4", "fault": "#d62728",
                              "ood": "#9467bd", "recall": "#2ca02c",
                              "anomaly": "#d62728"}.get(s["kind"], "grey"))
    for s in meta.segments:
        ax.text((s["win_start"] + s["win_end"]) / 2, 1.06, s["name"],
                ha="center", fontsize=7.5, color="#444444")
    ax.legend(loc="upper left", fontsize=8)

    # 2. scores
    ax = axes[1]
    ax.plot(t, anomaly, color="#d62728", lw=1.0, label="anomaly score")
    ax.plot(t, drift, color="#1f77b4", lw=1.0, label="drift score")
    ax.plot(t, ood, color="#9467bd", lw=1.0, alpha=0.8, label="OOD score")
    ax.axhline(0.5, color="#888888", ls="--", lw=0.8)
    ax.axhline(0.85, color="#d62728", ls=":", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.legend(loc="center left", fontsize=8, ncol=3)

    # 3. actions as a colored strip
    ax = axes[2]
    for i, r in enumerate(records):
        ax.axvspan(i, i + 1, color=ACT_COLORS[r["action"]], alpha=0.85)
    ax.set_yticks([])
    ax.set_ylabel("gate action")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in ACT_COLORS.values()]
    ax.legend(handles, ACT_COLORS.keys(), loc="center left",
              fontsize=8, ncol=5)

    # 4. exceedance / max z-score (residual shape evidence)
    ax = axes[3]
    exc = np.array([r["exceed_count"] for r in records])
    mz = np.array([r["max_zscore"] for r in records])
    ax.plot(t, mz, color="#ff7f0e", lw=0.9, label="max channel z-score")
    ax.axhline(10.5, color="#d62728", ls=":", lw=0.9,
               label="fault threshold (10.5)")
    ax.fill_between(t, 0, exc, step="mid", alpha=0.30, color="#1f77b4",
                    label="exceeding channels")
    ax.set_ylabel("residual shape")
    ax.set_xlabel("stream window")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.set_xlim(0, len(records))

    fig.savefig(OUT, dpi=150)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
