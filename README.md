# SAGE-Monitor — Safety-Gated Continual Monitoring Framework

**Development of a Continual-Learning-Based Intelligent Monitoring Framework for
Systems Operating Under Dynamically Changing Conditions**

A streaming autonomous health-monitoring framework that separates
environmental/operational **drift** from genuine **faults**, gates learning
behind a safety check, performs **parameter-efficient continual adaptation** on
trusted normal data only, and uses **regime-aware memory** to recall previously
learned operating conditions instead of relearning them.

The core research idea:

```
Detect Drift -> Verify Normality -> Safety Gate -> Adapt Selectively
   -> Store Regime -> Recall When Needed
```

rather than the naive `Drift -> Retrain`, which lets faults be learned as a
new normal.

## Project layout

```
Continual Monitoring Framework/
├── sage_monitor/          # the framework package
│   ├── config.py          # all hyper-parameters (SAGEConfig)
│   ├── encoder.py         # TCN + attention encoder, LoRA-style adapters
│   ├── heads.py           # drift detector, anomaly head, OOD head
│   ├── gate.py            # the safety gate (central algorithm)
│   ├── memory.py          # regime-aware memory (prototypes + replay)
│   ├── controller.py      # the full observe/detect/gate/adapt loop
│   ├── losses.py          # multi-objective training losses
│   ├── streams.py         # chronological continual-learning stream builder
│   ├── datasets.py        # SMAP/MSL loading + windowing utilities
│   └── metrics.py         # detection / temporal / continual / SAFETY metrics
├── demo/                  # working project demo (REST API + dashboard)
│   ├── server.py          # FastAPI app: sessions, live streaming, reports
│   ├── sessions.py        # session manager + streaming threads
│   ├── data_sources.py    # scripted scenario + real SMAP/MSL loaders
│   └── static/            # Mission Control dashboard (no build step)
├── tests/                 # framework + API e2e tests (pytest)
├── scripts/
│   ├── run_local.py       # end-to-end CPU/GPU run on the synthetic stream
│   ├── build_notebook.py  # assembles the Colab notebook from cells + package
│   ├── runtime_analysis.py# parameter counts + CPU/GPU benchmarks
│   ├── verify_endpoints.py# live endpoint verification (34 checks)
│   └── make_report_figure.py  # decision-stream figure
├── notebooks/
│   ├── cells/             # individual Colab cells (markdown + code)
│   └── SAGE_Monitor_Colab_Training.ipynb   # GitHub-backed cloud training
├── docs/
│   ├── DEMO_GUIDE.md      # datasets, demo script, expected outputs
│   ├── RUNTIME_AND_GPU_ANALYSIS.md
│   ├── E2E_TEST_REPORT.md
│   └── FACTCHECK.md
├── artifacts/             # run reports, saved models, screenshots (generated)
├── data/                  # downloaded datasets (git-ignored by convention)
└── requirements.txt
```

## Live demo (dashboard + REST API)

```bash
pip install -r requirements.txt -r demo_requirements.txt
python -m demo.server          # http://127.0.0.1:8000
```

One process serves the Mission Control dashboard **and** the REST API that
wraps `sage_monitor`: session creation & base training, live streaming gate
decisions, fault/drift/OOD injection, regime memory status, the full safety
report and JSON report download. Two demo datasets are bundled (the scripted
drift/fault/recall/OOD scenario and real NASA SMAP P-1/T-1 channels).
See **docs/DEMO_GUIDE.md** for the presentation walkthrough and
**docs/E2E_TEST_REPORT.md** for the endpoint/E2E verification evidence.
The framework runs on CPU (~337 K params, 34.8× real time on 2 cores);
a GPU is not required — see **docs/RUNTIME_AND_GPU_ANALYSIS.md**.

## How it works

**Layers**

1. **Continuous monitoring** — telemetry ingestion, windowing, temporal
   encoding (causal TCN + lightweight attention) to a latent health state `z`.
2. **Environmental intelligence** — three parallel signals:
   - *drift score*: distance of `z` from the trusted-normal reference
     (Mahalanobis-style) + Page-Hinkley change detection on the distance stream;
   - *anomaly score*: reconstruction error made **global-affine-invariant**
     (the coherent scale/bias of a regime shift is removed by a least-squares
     fit), calibrated against trusted-normal errors;
   - *OOD score*: prototype + Mahalanobis distance calibrated against the
     trusted-normal cloud.
3. **Continual-learning controller** — the **safety gate** combines confidence,
   normality, drift and teacher-student agreement, plus two fault
   discriminators that are robust to the sensor count:
   - **residual shape** — the maximum aligned-residual z-score identifies
     concentrated extreme fault energy, while the fraction of exceeding
     channels identifies a widespread novel condition. Raw exceedance count
     alone is not used because a stale decoder can make every channel exceed
     on a learnable regime shift, and
   - **persistence + verify-before-alarm** — sustained fault evidence first
     triggers a *trial recall* of the nearest known regime (restoring its
     adapter and calibration). If the exceedance resolves, it was a returning
     regime; if it persists, the fault is confirmed and ALERT fires.

   The gate emits one of: `NO_ACTION / ADAPT / RECALL / REJECT / ALERT`.
4. **Autonomous health intelligence** — alerts, adaptation decisions, regime
   recall, and the safety metrics below.

**Parameter-efficient adaptation** — the temporal backbone stays frozen;
continual updates touch only a low-rank (LoRA-style) adapter plus the
reconstruction/forecast heads. A frozen EMA **teacher** and an embedding
**replay** memory protect older regimes from catastrophic forgetting.

**Safety invariant** — observations judged anomalous are never absorbed into
regime memory, the drift reference, or the anomaly calibration. This prevents
the catastrophic failure mode `fault -> model learns fault -> fault becomes
normal`.

## Run it

```bash
pip install -r requirements.txt

# end-to-end run on the scripted drift/fault/recall/OOD stream (CPU works)
python -m scripts.run_local --steps 4200 --channels 12 --window 60 --stride 4

# rebuild the Colab notebook after changing the package
python scripts/build_notebook.py
```

The run trains a base model on the first clean regime, streams every window
through the safety gate, and writes `artifacts/run_report.json`.

## Train in the cloud (Google Colab)

Open `notebooks/SAGE_Monitor_Colab_Training.ipynb`. The notebook clones the
public `sage_monitor` package from
`https://github.com/kishoreafk/Continual-Learning-Based-Intelligent-Monitoring-Framework`
(branch `main`) at runtime and prints the checked-out commit. It then downloads
the NASA SMAP/MSL telemetry dataset from Kaggle
(`patrickfleith/nasa-anomaly-detection-dataset-smap-msl`), recursively loads
both bundled and per-channel split layouts, builds clean training and labelled
test windows, and streams the Kaggle test data by default. Set
`STREAM_MODE = "synthetic_gate"` in the stream cell for the controlled
B/FAULT/OOD gate regression. The notebook then reports detection, temporal,
continual-learning and safety metrics and plots the decision stream.

Kaggle credentials: add `KAGGLE_USERNAME` / `KAGGLE_KEY` under
*Colab → Secrets* (or upload `kaggle.json` when prompted).

## Key metrics

- **Detection**: precision / recall / F1, plus the point-adjust variant used
  in the SMAP/MSL literature; detection delay and missed intervals.
- **Continual learning**: number of regimes discovered, adaptations, recalls.
- **Safety** (the research centrepiece):
  - `UnsafeAdaptationRate = adaptation events on true anomalies / all adaptations`
  - false-alarm rate on normal windows.

## Representative results (CPU, scripted drift/fault/recall/OOD stream)

| sensor count | precision | recall | F1 | point-adjust F1 | false-alarm rate | unsafe adaptations |
|---|---|---|---|---|---|---|
| 8  | 1.000 | 0.591 | 0.743 | 1.000 | 0.000 | 0 |
| 12 | 1.000 | 0.761 | 0.864 | 1.000 | 0.000 | 0 |
| 25 | 1.000 | 0.690 | 0.817 | 1.000 | 0.000 | 0 |

Every fault interval is detected (point-adjust recall = 1.0 at all sensor
counts) with **zero false alarms** and **zero unsafe adaptations**: the gate
never learns a fault as a new normal. The windowed recall reflects the
deliberate confirmation delay (persistence streak + verify-before-alarm trial)
that keeps a streaming monitor from crying wolf at every regime boundary.

## Research positioning

The framework is a **system-level synthesis** of 2026 continual test-time
adaptation and continual anomaly-detection research, not a claim that each
component is individually novel. See `docs/FACTCHECK.md` for the verified
citation list and dataset facts.
