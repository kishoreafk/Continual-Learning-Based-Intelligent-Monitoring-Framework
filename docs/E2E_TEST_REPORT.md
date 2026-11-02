# E2E Verification & Test Report

**Scope:** the SAGE-Monitor framework (upstream package), the new demo
backend (`demo/`), and the Mission Control dashboard (`demo/static/`).
**Date:** 2026-11-03 · **Platform:** Linux, Python 3.12, torch 2.15+cpu,
2 vCPU / 4 GB RAM (no GPU) · **pytest JSON:** `artifacts/pytest_report.json`
· **Live API verification:** `artifacts/endpoint_verification.json`

---

## 1. Unit / framework E2E tests — `tests/test_framework_e2e.py` (21 tests)

| Area | Tests | Result |
|---|---|---|
| Stream construction | segments tile the timeline contiguously; scripted order A→drift→B→C→FAULT→B-return→OOD; labels mark only the fault segment (± window-overlap tail); window tensor shapes | ✅ PASS |
| Config | `to_dict`/`from_dict` roundtrip; unknown keys dropped | ✅ PASS |
| Safety gate (unit) | stable → NO_ACTION · persistent fault → REJECT then ALERT · fault spike → gather evidence · novel condition → monitoring only · RECALL requires verification · cold warm-up never acts | ✅ PASS |
| Metrics | exact precision/recall/F1 arithmetic · point-adjust credits whole intervals · detection delay · false-alarm rate | ✅ PASS |
| **Full pipeline (train → stream 886 windows → report)** | every window processed · precision = 1.0 · point-adjust recall = 1.0 · zero false alarms · **zero unsafe adaptations** · regimes learned & recalled · alerts fired on the fault · save/load roundtrip reproduces latents | ✅ PASS |
| Demo data sources | scripted load · fault injection changes labels+data, leaves history untouched · drift labelled normal · clean restores · real SMAP P-1 loads with 220 anomaly windows | ✅ PASS |

**21/21 passed.**

## 2. API E2E tests — `tests/test_api_e2e.py` (15 tests, FastAPI TestClient)

| Area | Tests | Result |
|---|---|---|
| Meta | health · info actions · dataset catalogue · dashboard + assets served · OpenAPI documents all 13+ paths | ✅ PASS |
| Errors | unknown session 404 (GET/POST/DELETE) · bad dataset 400 · bad inject kind 400 | ✅ PASS |
| **Full lifecycle (scripted)** | create → trains to ready → list/detail → step 120 → stream schema → chart payload → **inject FAULT(60)** → start/pause/speed/resume → streams to done (886) → ALERTs on the injected fault → **safety invariants: 0 unsafe adaptations, precision 1.0, point-adjust recall 1.0, FAR 0.0** → events logged → report download attachment → delete → 404 | ✅ PASS |
| Real data | SMAP P-1 session trains and streams labelled windows | ✅ PASS |

**15/15 passed — combined pytest: 36/36 (74 s).**

## 3. Live endpoint verification — `scripts/verify_endpoints.py` (34 checks
against a real uvicorn server on 127.0.0.1:8000)

All 15 documented API paths exercised with real HTTP:

```
GET  /api/health              GET  /api/info                GET  /api/datasets
GET  /                        GET  /static/app.js           GET  /static/style.css
GET  /docs                    GET  /openapi.json            GET  /api/sessions
POST /api/sessions            GET  /api/sessions/{sid}      DELETE /api/sessions/{sid}
POST /api/sessions/{sid}/start | /pause | /step | /speed | /inject
GET  /api/sessions/{sid}/stream | /chart | /events
GET  /api/sessions/{sid}/report | /report/download
```

**34/34 checks passed.** Highlights from the live run:

* Injected fault (60 windows) during a live stream → **22 ALERTs**, gate
  actions `{NO_ACTION: 828, RECALL: 20, ADAPT: 16, ALERT: 22}`
* **Safety invariants on the live server:** unsafe adaptations 0 ·
  detection precision 1.000 · point-adjust recall 1.000 · false-alarm
  rate 0.000
* SMAP P-1 (real NASA telemetry, 1200-window subset): trains, streams,
  220 anomaly windows covered

## 4. Browser (UI) verification

The dashboard was driven end-to-end in a headless browser
(screenshots in `artifacts/screenshots/`):

1. `01_dashboard_idle.png` — catalogue, training panel, device badge
2. `02_streaming_fault_alert.png` — live streaming, injected fault ALERTing,
   anomaly saturating, timeline ticks
3. `03_scripted_fault_alerts.png` — scripted FAULT block (red) and the
   B-return RECALL storm (green)
4. `04_final_report.png` — finished session with computed report
   (precision 1.000 · point-adjust recall 1.000 · FAR 0.0000 · 0 unsafe)

One UI bug was found and fixed during this pass (frontend polled
`/stream?limit=5000` while the API capped `limit` at 2000 → 422).

## 5. Defects found & fixed during verification

| # | Defect | Fix |
|---|---|---|
| 1 | `scripts/run_local.py` hard-coded Windows path `E:/Continual Monitoring Framework` — repo unusable off that machine | resolved from `__file__` (`Path(__file__).resolve().parents[1]`) |
| 2 | `SAGEController.load()` left the encoder in train mode (dropout active) — loaded models scored differently than the saved ones | `self.encoder.eval()` after loading |
| 3 | Demo sessions didn't seed torch RNG → gate behaviour varied between processes | `Session.train()` seeds torch/numpy from the session seed (same discipline as `run_local.py`) |
| 4 | Bad inject kind raised 500 instead of 400 | validated in `Session.inject`, surfaced as 400 |
| 5 | Frontend/API stream-limit mismatch → dashboard polls 422 | API `limit` cap raised to 5000 |

## 6. Reproduce

```bash
python -m pytest tests/ -q                       # 36 tests
python scripts/verify_endpoints.py               # 34 live checks (server running)
python -m scripts.run_local                      # CLI pipeline + report
```
