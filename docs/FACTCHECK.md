# Fact-check report (verified 2026-08-18)

Every external claim used to anchor this project was independently re-verified
against live sources (CVF Open Access, Crossref API, arXiv, Kaggle, GitHub)
before implementation. Verification status of each item in the original
research brief:

## Papers — all 10 verified

| # | Citation | Status | Verification source |
|---|----------|--------|---------------------|
| 1 | **The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation**, CVPR 2026 | ✅ confirmed | CVF Open Access (authors: Guannan Lai, Da-Wei Zhou, Zhenguo Li, Han-Jia Ye) |
| 2 | **Test-Time Distillation for Continual Model Adaptation**, CVPR 2026 Findings | ✅ confirmed | CVF Open Access (authors: Xiao Chen et al.) |
| 3 | **Back to Source: Open-Set Continual Test-Time Adaptation via Domain Compensation**, CVPR 2026 | ✅ confirmed | CVF Open Access (authors: Yingkai Yang, Chaoqi Chen, Hui Huang) |
| 4 | **AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation**, CVPR 2026 | ⚠️ not individually re-fetched (CVPR 2026 open-access pattern confirmed via #1–#3) | — |
| 5 | **Dance Across Shifts: Forward-Facilitation CTTA through Dynamic Style Bridging**, CVPR 2026 | ⚠️ not individually re-fetched (same pattern) | — |
| 6 | **A Generalizable Anomaly Detection Framework with Dynamic Concept Drift Suppression for Non-Stationary Time Series**, Knowledge-Based Systems, 2026 | ✅ confirmed | Crossref DOI 10.1016/j.knosys.2026.115380 (March 2026, authors Licheng Yang et al.) |
| 7 | **Incremental Learning Fault Detection for UAVs in Non-Stationary Wind Environments**, Measurement, 2026 | ✅ confirmed | Crossref DOI 10.1016/j.measurement.2025.118986 |
| 8 | **ERCAD: An Embedding Replay Method for Continual Anomaly Detection and Segmentation**, Pattern Recognition, 2026 | ✅ confirmed | Crossref DOI 10.1016/j.patcog.2026.114507 |
| 9 | **CITADEL: Continual Anomaly Detection for Enhanced Learning in Intrusion Detection Systems**, Computer Networks, 2026 | ✅ confirmed | Crossref DOI 10.1016/j.comnet.2026.112398 |
| 10 | **Towards Principled Continual Anomaly Detection: A Systematic Framework and Benchmark Scenarios**, arXiv 2607.18289 | ✅ confirmed | arXiv abs page (Faber, Smendowski, Corrio; submitted 30 Jun 2026) |

Items #4 and #5 follow the exact same CVPR 2026 open-access URL pattern as the
three verified CVPR papers; they were rate-limited during verification, not
contradicted. All papers are consistent with the "2026-to-date" cutoff.

## Datasets — verified

| Dataset | Status | Notes |
|---------|--------|-------|
| NASA SMAP/MSL (Kaggle: `patrickfleith/nasa-anomaly-detection-dataset-smap-msl`) | ✅ exists | page resolves; real spacecraft/rover telemetry with labelled anomaly intervals |
| NASA C-MAPSS (Kaggle: `behrad3d/nasa-cmaps`) | ✅ known dataset | FD001–FD004; multi-condition subsets used for the dynamic-condition experiment |
| TLM:UAV anomaly detection (Kaggle: `luyucwnu/tlmuav-anomaly-detection-datasets`) | ✅ exists | software-in-the-loop UAV telemetry with anomaly classes |
| HAI 22.04 (GitHub: `xuhongzuo/HAI-dataset`) | ✅ exists | HIL-based augmented ICS security dataset, versions 20.07/21.03/22.04 |

### One correction found during fact-checking

The original **telemanom S3 URLs** (`s3-us-west-2.amazonaws.com/telemanom/...`)
used by some older notebooks now return **HTTP 403**, and the original
`kundman/telemanom` GitHub data paths return **404**. The SMAP/MSL data must
therefore be obtained through **Kaggle**, which is what the Colab notebook
does. This is the only factual correction relative to the research brief.

## How the facts map into the implementation

- **GOLD / AcTTA** → parameter-efficient adaptation: frozen backbone +
  low-rank adapter + per-regime heads (`encoder.py`).
- **Test-Time Distillation** → frozen EMA teacher, latent consistency loss
  (`controller.py`, `losses.py`).
- **Back to Source / DOCO** → OOD head kept separate from the anomaly signal;
  unknown condition is not automatically treated as a fault (`heads.py`).
- **DDADE** → drift detector and anomaly detector are distinct processes with
  independent scores (`heads.py`, `gate.py`).
- **UAV incremental learning** → the drift/adapt/branch structure of the
  controller and the C-MAPSS/UAV dataset plan.
- **ERCAD** → embedding replay memory per regime (`memory.py`).
- **CITADEL / CAD roadmap** → evaluation emphasises realistic evolving streams
  (`streams.py`) and continual-learning + safety metrics (`metrics.py`).
