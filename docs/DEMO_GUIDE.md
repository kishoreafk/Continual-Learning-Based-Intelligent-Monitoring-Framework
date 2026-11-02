# SAGE-Monitor — Demo & Presentation Guide

This guide explains how to run the working project demo, which datasets to
present, what the framework outputs, and what to say at each step.

---

## 1. One-command start

```bash
# from the repository root
pip install -r requirements.txt -r demo_requirements.txt
python -m demo.server                       # serves http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000** in a browser — the Mission Control dashboard
is served by the same process that wraps the `sage_monitor` framework.

* Interactive API docs: `http://127.0.0.1:8000/docs`
* Device badge in the header shows what the framework is running on
  (`cpu` here — see `docs/RUNTIME_AND_GPU_ANALYSIS.md`: a GPU is **not**
  required for the demo; the whole model is ~337 K parameters / 1.3 MB).

## 2. The demo datasets (and when to use them)

| # | Dataset (`/api/datasets`) | Kind | Why it is used | Expected outcome |
|---|---|---|---|---|
| 1 | **Scripted Drift/Fault/OOD Scenario** (`scripted`) ★ demo pick | Synthetic, 12 channels, 886 windows, ground truth known | Exercises **every** gate mechanism in order: learn A → gradual drift → learn B → sudden drift → learn C → sustained FAULT → B returns → novel OOD. Also supports live event injection. | ADAPT on drifts, **REJECT/ALERT on the fault**, RECALL when B returns, NO_ACTION ("novel condition: monitoring only") on OOD. Zero unsafe adaptations, zero false alarms, point-adjust F1 = 1.0 |
| 2 | **NASA SMAP P-1** (`smap:P-1`) | Real spacecraft telemetry, 8 505 test steps, 3 labelled contextual anomalies | Shows the framework on **real, imperfect** data (already bundled under `data/smap_msl/`; Kaggle credentials not required for the demo) | Anomaly score rises in the labelled intervals; metrics computed against NASA's manual labels. Real-data detection is intentionally less clean than the scripted stream — this is a discussion point |
| 3 | **NASA SMAP T-1** (`smap:T-1`) | Real, 8 612 test steps, 2 labelled anomalies (point + contextual) | Second real channel, includes a *point* anomaly | Same pipeline, different failure mode |

Full SMAP/MSL (all 55 channels) can still be downloaded with Kaggle
credentials as described in the README and used through the Colab notebook —
the demo ships with two channels so the presentation works offline.

## 3. Recommended demo script (≈ 6 minutes)

1. **Create the session** — panel *1*: keep ★ *Scripted Drift/Fault/OOD
   Scenario*; panel *2*: defaults (window 60, stride 4, epochs 6, seed 42) →
   **Create session & train** (~5 s on CPU: trains the frozen TCN+attention
   encoder on the clean regime only).
2. **Step into the stream** — click *⏭ +50* twice; the telemetry and safety
   signals appear. Point out the warm-up: `gate_min_obs = 32` windows of
   statistics before the gate acts.
3. **Play** — panel *3*, speed 25 win/s. Watch:
   * **gradual drift (~w135–210)** → drift score climbs, then blue **ADAPT**
     ticks: "shift candidate: probing learnability" → the LoRA-style adapter
     absorbs the new regime (verify-before-adapt: a probe must reduce
     held-out error, otherwise it is rolled back exactly).
   * **FAULT (~w510–585)** → anomaly saturates at 1.0, max channel z-score
     crosses 10.5 → red **ALERT** ticks: "emergency: persistent component
     fault". The fault is **never** learned (UNSAFE ADAPT stays 0).
   * **B return (~w585–735)** → dense green **RECALL** ticks: "known regime
     returned (verified)" — the regime memory restores B's adapter instead
     of relearning.
   * **novel OOD (~w735+)** → OOD = 1.0 but the gate stays NO_ACTION:
     "novel condition: monitoring only" — *unknown ≠ fault*.
4. **Live injection (the wow moment)** — pause at any point, press
   **⚡ Inject FAULT** (or DRIFT / OOD), then resume. Within ~15 windows the
   gate ALERTs on the injected fault and refuses to adapt to it.
5. **Report** — press *Compute metrics*: detection, point-adjust,
   false-alarm rate, continual stats and the safety invariant. **⬇ JSON**
   downloads `sage_report_<session>.json` (same schema as
   `artifacts/run_report.json`).
6. **Real data (optional)** — create a second session with *NASA SMAP P-1*;
   watch the labelled anomaly intervals cross the stream.

## 4. What the framework outputs

**Per window** (`GET /api/sessions/{id}/stream`, the charts on the dashboard):

| Field | Meaning |
|---|---|
| `action` | gate decision: `NO_ACTION / ADAPT / RECALL / REJECT / ALERT` |
| `reason` | human-readable justification of the decision |
| `anomaly` | global-affine-invariant reconstruction error, calibrated 0–1 |
| `drift` | latent Mahalanobis distance + Page-Hinkley change detector |
| `ood` | prototype/Mahalanobis distance of the latent (monitoring only) |
| `gate` | the weighted gate score G_t |
| `exceed_count`, `max_zscore` | residual-shape evidence (fault vs novel band) |
| `regime_sim` | cosine similarity to the nearest known regime |

**Run report** (`GET /api/sessions/{id}/report`, also `artifacts/run_report.json`):

* Detection: precision / recall / F1 on the gate decisions, plus the
  **point-adjust** variant used in the SMAP/MSL literature, detection delays,
  missed intervals, false-alarm rate on normal windows.
* Continual learning: regimes discovered, adaptations, recalls, rejects,
  alerts.
* Safety (research centrepiece): `unsafe_adaptation_rate` — adaptations that
  absorbed true anomalies over all adaptations. The demo stream gives
  **0** with the default seed.
* Per-segment mean anomaly scores (drift recovery analysis).

**Reference numbers to quote (12 channels, seed 42, CPU):**
precision 1.000 · windowed recall ≈ 0.66–0.83 (deliberate persistence +
verify-before-alarm delay) · **point-adjust F1 1.000** · false-alarm rate
0.000 · unsafe adaptations 0 · regimes discovered 2–3 · recalls 17–68.

## 5. Headless / API-only demo

Everything above also works without a browser:

```bash
python scripts/verify_endpoints.py --base http://127.0.0.1:8000  # endpoint E2E
python -m scripts.run_local                                      # CLI run + report
python -m pytest tests/                                          # 36 e2e tests
```

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `SMAP channel not present` | only `P-1`/`T-1` are bundled; add parquet files to `data/smap_msl/{train,test}/` or use the scripted dataset |
| Training seems slow | 2-core laptop ≈ 5 s; do not run two sessions at once on the same machine |
| Session says `error` | the status field carries the exception; check the server console |
| Port busy | `python -m demo.server --port 8001` |
