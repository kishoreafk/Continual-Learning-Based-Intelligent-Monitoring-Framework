"""Runtime / GPU requirement analysis for SAGE-Monitor.

Measures: parameter counts, model disk size, base-training throughput and
per-window streaming latency on the available device (CPU here). Produces
artifacts/runtime_analysis.json + a human-readable summary.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_monitor.config import SAGEConfig          # noqa: E402
from sage_monitor.controller import SAGEController  # noqa: E402
from sage_monitor.datasets import synth_multivariate  # noqa: E402
from sage_monitor.streams import build_stream        # noqa: E402

OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_used": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "platform": sys.platform,
    }

    # --- parameter accounting (default config, and a smaller demo config) --
    for tag, kwargs in (("default_25ch", {}), ("demo_12ch", {"n_channels": 12}),
                        ("demo_8ch", {"n_channels": 8})):
        cfg = SAGEConfig(**kwargs)
        ctrl = SAGEController(cfg)
        enc_p = count_params(ctrl.encoder)
        head_p = count_params(ctrl.anomaly_head)
        total = enc_p + head_p
        info[f"params_{tag}"] = {
            "encoder_total": enc_p,
            "encoder_base": sum(p.numel() for p in ctrl.encoder.base_parameters()),
            "encoder_adaptive": sum(p.numel() for p in ctrl.encoder.adaptive_parameters()),
            "anomaly_head": head_p,
            "total": total,
            "total_MB_fp32": round(total * 4 / 1024 / 1024, 3),
        }
        del ctrl

    # --- disk size of a saved model ---------------------------------------
    cfg = SAGEConfig(n_channels=12, window=60, train_epochs=1, device="cpu")
    ctrl = SAGEController(cfg)
    tmp = OUT / "_tmp_model.pt"
    ctrl.save(str(tmp))
    info["model_checkpoint_KB"] = round(tmp.stat().st_size / 1024, 1)
    tmp.unlink()

    # --- benchmark: base training + streaming, 12 channels -----------------
    steps, ch, win, stride, seg_len = 4200, 12, 60, 4, 600
    base = synth_multivariate(steps, ch, seed=42)
    X, Y, meta = build_stream(base, window=win, stride=stride,
                              seg_len=seg_len, seed=42)
    first_end = meta.segments[0]["win_end"]
    train_X = X[:first_end][Y[:first_end] == 0]

    cfg = SAGEConfig(n_channels=ch, window=win, train_epochs=6,
                     gate_min_obs=32, device="cpu", seed=42)
    ctrl = SAGEController(cfg)

    t0 = time.perf_counter()
    ctrl.train_base(train_X, verbose=False)
    train_s = time.perf_counter() - t0
    info["train_base"] = {"windows": int(train_X.size(0)), "epochs": 6,
                          "seconds": round(train_s, 2),
                          "windows_per_s": round(train_X.size(0) / train_s, 1)}

    t0 = time.perf_counter()
    records = ctrl.run_stream(X, labels=Y, log_every=0)
    stream_s = time.perf_counter() - t0
    n = X.size(0)
    info["stream"] = {"windows": n, "seconds": round(stream_s, 2),
                      "ms_per_window": round(stream_s * 1000 / n, 2),
                      "realtime_factor_x": round(n / stream_s, 1)}
    from sage_monitor import metrics as M
    rep = M.summarise(ctrl, records, meta.segments, Y.numpy())
    info["sanity_metrics"] = {
        "precision": rep["detection"]["precision"],
        "recall": rep["detection"]["recall"],
        "f1": rep["detection"]["f1"],
        "point_adjust_recall": rep["detection_point_adjust"]["recall"],
        "false_alarm_rate": rep["false_alarm_rate"],
        "unsafe_adaptation_rate": rep["unsafe_adaptation_rate"],
        "n_regimes": rep["n_regimes"],
    }

    path = OUT / "runtime_analysis.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
