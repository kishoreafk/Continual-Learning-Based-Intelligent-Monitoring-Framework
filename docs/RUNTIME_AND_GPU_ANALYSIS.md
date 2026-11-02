# Runtime & GPU Requirement Analysis

**Question:** does this framework need to run on a higher GPU runtime?

**Answer: No — the demo and the CPU training runs are fully functional
without a GPU.** The model is deliberately small (parameter-efficient
continual learning is a design goal), and every reported metric is
reproducible on CPU. A GPU only shortens the optional full SMAP/MSL
sweep in Colab; even there a free T4 is far beyond what the model needs.

## Measured evidence (`artifacts/runtime_analysis.json`,
produced by `scripts/runtime_analysis.py`, Python 3.12, torch 2.15+cpu,
2 CPU cores / 4 GB RAM, no CUDA device present)

### Model size

| Config | Encoder params | Adaptive (LoRA) params | Total | fp32 size |
|---|---|---|---|---|
| 25 channels (default / SMAP) | 347 091 | 138 643 | **347 091** | **1.32 MB** |
| 12 channels (demo) | 336 964 | 129 348 | 336 964 | 1.29 MB |
| 8 channels | 333 848 | 126 488 | 333 848 | 1.27 MB |

Saved checkpoint (encoder + teacher + heads + calibration buffer):
**2.7 MB**. For comparison: ResNet-18 ≈ 11 M params, GPT-2 ≈ 124 M.
This model is ~3 orders of magnitude below anything that needs GPU
acceleration to stay interactive.

### Throughput on 2 CPU cores (no GPU)

| Phase | Measurement |
|---|---|
| Base training (12 ch, 135 windows, 6 epochs) | **4.35 s** |
| Streaming (886 windows through the full gate loop, incl. adaptation probes) | **25.5 s → 28.8 ms/window → 34.8× real time** |
| Session creation in the demo server (train + seed statistics) | ~5 s wall time |

The demo streams at 5–25 windows/s by default, i.e. the framework runs at
least 30× faster than the presentation needs. Interactive pause / step /
injection stays responsive because every window is a single 3 ms
forward pass plus occasional sub-second adaptation probes.

### Quality parity

The CPU run reproduces the README's representative results exactly
(`artifacts/run_report.json`): precision 1.000, point-adjust F1 1.000,
false-alarm rate 0.000, unsafe adaptations 0. There is no numeric reason
to add a GPU to the demo path.

## When a GPU *would* matter

1. **Full SMAP/MSL sweep in the Colab notebook** — all 55 channels ×
   10 epochs. Each channel is still a ~350 K-param model; the bottleneck
   is I/O and orchestration, not FLOPs. A free Colab T4 finishes it
   comfortably; CPU also works, just slower.
2. **Future scaling experiments** (larger `d_model`, more channels,
   longer windows): GPU wins grow with `d_model²` attention cost. The
   config exposes `device: "auto" | "cpu" | "cuda"` — the controller
   already uses CUDA transparently when present (`torch.cuda.is_available()`).

## Recommendation

* **Present the demo on CPU** (any modern laptop or a small cloud VM,
  2 vCPU / 4 GB is enough). No GPU runtime required.
* If the presentation machine must also train the full SMAP/MSL set, use
  the bundled Colab notebook on a **free T4** — no paid GPU runtime needed.

## Demo packaging notes

* `requirements.txt` — framework (torch, numpy, pandas, scikit-learn,
  matplotlib, pyarrow). CPU torch wheels are fine:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
* `demo_requirements.txt` — adds fastapi + uvicorn for the demo server.
* The demo server pins `device="auto"`: it will pick up CUDA automatically
  if the host has it, so the same command works on both runtimes.
