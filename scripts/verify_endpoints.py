"""Live end-to-end verification of every SAGE-Monitor demo endpoint.

Runs against a REAL running server (default http://127.0.0.1:8000), exercises
the full session lifecycle and writes artifacts/endpoint_verification.json.

Usage:
    python scripts/verify_endpoints.py --base http://127.0.0.1:8000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append({"endpoint": name, "ok": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main(base: str):
    c = httpx.Client(base_url=base, timeout=120.0)
    print(f"== verifying SAGE-Monitor demo API at {base} ==")

    # ---- meta endpoints ------------------------------------------------
    r = c.get("/api/health")
    ok = r.status_code == 200 and r.json()["status"] == "ok"
    check("GET /api/health", ok, json.dumps(r.json())[:100])

    r = c.get("/api/info")
    info = r.json() if r.status_code == 200 else {}
    ok = (r.status_code == 200 and "default_config" in info
          and set(info["actions"]) == {"NO_ACTION", "ADAPT", "RECALL", "REJECT", "ALERT"})
    check("GET /api/info", ok, f"actions={info.get('actions')}")

    r = c.get("/api/datasets")
    dsets = r.json() if r.status_code == 200 else []
    ok = r.status_code == 200 and any(d["id"] == "scripted" for d in dsets)
    check("GET /api/datasets", ok, f"{len(dsets)} datasets: {[d['id'] for d in dsets]}")

    r = c.get("/")
    check("GET / (dashboard)", r.status_code == 200 and b"Mission Control" in r.content,
          f"{len(r.content)} bytes html")
    for asset in ("/static/app.js", "/static/style.css"):
        r = c.get(asset)
        check(f"GET {asset}", r.status_code == 200 and len(r.content) > 500,
              f"{len(r.content)} bytes")
    r = c.get("/docs")
    check("GET /docs (OpenAPI UI)", r.status_code == 200)
    r = c.get("/openapi.json")
    paths = list(r.json()["paths"].keys()) if r.status_code == 200 else []
    check("GET /openapi.json", len(paths) >= 15, f"{len(paths)} documented paths")

    # ---- error handling ------------------------------------------------
    r = c.get("/api/sessions/doesnotexist")
    check("GET /api/sessions/{bad} -> 404", r.status_code == 404)
    r = c.post("/api/sessions", json={"dataset": "nope"})
    check("POST /api/sessions bad dataset -> 400", r.status_code == 400)
    r = c.post("/api/sessions/doesnotexist/step")
    check("POST step on bad session -> 404", r.status_code == 404)

    # ---- full session lifecycle (scripted, fast settings) ---------------
    r = c.post("/api/sessions", json={
        "dataset": "scripted", "epochs": 6, "speed": 1000, "channels": 12})
    ok = r.status_code == 200
    sid = r.json().get("session_id") if ok else None
    check("POST /api/sessions (create+train)", ok, f"session={sid}")
    if not sid:
        return finish()

    deadline = time.time() + 300
    status = None
    while time.time() < deadline:
        s = c.get(f"/api/sessions/{sid}").json()
        status = s["status"]
        if status in ("ready", "error"):
            break
        time.sleep(1.0)
    check("session trains to 'ready'", status == "ready", f"status={status}")

    r = c.post(f"/api/sessions/{sid}/step", params={"n": 120})
    ok = r.status_code == 200 and r.json()["playhead"] == 120
    check("POST /step?n=120", ok, f"playhead={r.json().get('playhead')}")

    r = c.get(f"/api/sessions/{sid}/stream", params={"since": 0, "limit": 50})
    snap = r.json()
    ok = (r.status_code == 200 and len(snap["records"]) == 50
          and {"action", "anomaly", "drift", "ood", "gate", "reason",
               "exceed_count", "max_zscore"} <= set(snap["records"][0]))
    check("GET /stream records schema", ok, f"{len(snap['records'])} records")
    ok = r.status_code == 200 and len(snap["segments"]) >= 7
    check("GET /stream segment map", ok,
          f"{len(snap['segments'])} scripted segments")

    r = c.get(f"/api/sessions/{sid}/chart", params={"since": 0, "limit": 120})
    cd = r.json()
    ok = (r.status_code == 200 and len(cd["series"]) == 120
          and len(cd["series"][0]) == 8)
    check("GET /chart series", ok, f"{len(cd['channels'])} channels x {len(cd['series'])} windows")

    # live injection: fault must be REJECT/ALERT'd and NEVER learned
    r = c.post(f"/api/sessions/{sid}/inject", json={"kind": "fault", "duration": 60})
    ok = r.status_code == 200 and r.json()["injected"]["kind"] == "fault"
    check("POST /inject fault", ok, str(r.json().get("injected"))[:80])

    r = c.post(f"/api/sessions/{sid}/start")
    check("POST /start", r.status_code == 200 and r.json()["status"] == "streaming")
    r = c.post(f"/api/sessions/{sid}/pause")
    check("POST /pause", r.status_code == 200 and r.json()["status"] == "paused")
    r = c.post(f"/api/sessions/{sid}/speed", json={"speed": 1000})
    check("POST /speed", r.status_code == 200 and r.json()["speed"] == 1000)
    r = c.post(f"/api/sessions/{sid}/start")
    check("POST /start (resume)", r.status_code == 200)

    # stream to completion (MAX speed) — up to 10 min on 2 CPU cores
    deadline = time.time() + 600
    final = None
    while time.time() < deadline:
        s = c.get(f"/api/sessions/{sid}").json()
        final = s["status"]
        if final == "done":
            break
        time.sleep(2.0)
    check("stream reaches 'done'", final == "done", f"status={final}")

    snap = c.get(f"/api/sessions/{sid}/stream", params={"since": 0, "limit": 1}).json()
    n = snap["total_windows"]
    snap = c.get(f"/api/sessions/{sid}/stream", params={"since": 0, "limit": n}).json()
    acts = {}
    for rec in snap["records"]:
        acts[rec["action"]] = acts.get(rec["action"], 0) + 1
    check("gate produced REJECT or ALERT on injected fault",
          acts.get("REJECT", 0) + acts.get("ALERT", 0) > 0, f"actions={acts}")

    r = c.get(f"/api/sessions/{sid}/report")
    rep = r.json()
    ok = r.status_code == 200 and {"detection", "detection_point_adjust",
                                   "false_alarm_rate", "stats",
                                   "unsafe_adaptation_rate", "n_regimes"} <= set(rep)
    check("GET /report schema", ok)
    check("safety invariant: zero unsafe adaptations",
          rep["stats"]["unsafe_adaptations"] == 0,
          f"adaptations={rep['stats']['adaptations']}, unsafe={rep['stats']['unsafe_adaptations']}")
    check("injected fault detected (point-adjust recall > 0)",
          rep["detection_point_adjust"]["recall"] > 0,
          f"pa_recall={rep['detection_point_adjust']['recall']:.3f}, "
          f"precision={rep['detection']['precision']:.3f}")
    check("zero false alarms on clean windows",
          rep["false_alarm_rate"] == 0.0, f"far={rep['false_alarm_rate']}")

    r = c.get(f"/api/sessions/{sid}/events", params={"limit": 20})
    evs = r.json()["events"]
    check("GET /events", r.status_code == 200 and len(evs) > 0, f"{len(evs)} events")
    r = c.get(f"/api/sessions/{sid}/report/download")
    ok = (r.status_code == 200 and "attachment" in r.headers.get("content-type", "")
          or "attachment" in r.headers.get("content-disposition", ""))
    check("GET /report/download (attachment)", ok)

    r = c.delete(f"/api/sessions/{sid}")
    check("DELETE /api/sessions/{sid}", r.status_code == 200)
    r = c.get(f"/api/sessions/{sid}")
    check("deleted session -> 404", r.status_code == 404)

    # ---- SMAP real-telemetry session (subset for speed) ------------------
    # P-1's labelled anomalies start at raw step 2149; with window=60/stride=4
    # we need >= 1200 windows to cover the first two anomaly intervals.
    r = c.post("/api/sessions", json={
        "dataset": "smap:P-1", "epochs": 4, "speed": 1000, "max_windows": 1200})
    if r.status_code == 200:
        sid2 = r.json()["session_id"]
        deadline = time.time() + 300
        while time.time() < deadline:
            s = c.get(f"/api/sessions/{sid2}").json()
            if s["status"] in ("ready", "error"):
                break
            time.sleep(1.0)
        check("SMAP P-1 session trains", s["status"] == "ready", f"status={s['status']}")
        c.post(f"/api/sessions/{sid2}/step", params={"n": 150})
        snap = c.get(f"/api/sessions/{sid2}/stream", params={"since": 0, "limit": 1}).json()
        check("SMAP real telemetry streams",
              snap["total_windows"] == 1200 and snap["anomaly_windows"] > 0,
              f"windows={snap['total_windows']}, anomaly_windows={snap['anomaly_windows']}")
        c.delete(f"/api/sessions/{sid2}")
    else:
        check("SMAP P-1 session", False, r.text[:120])

    return finish()


def finish():
    n_ok = sum(1 for r in RESULTS if r["ok"])
    print(f"\n== {n_ok}/{len(RESULTS)} checks passed ==")
    out = ROOT / "artifacts" / "endpoint_verification.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "passed": n_ok, "total": len(RESULTS), "results": RESULTS,
    }, indent=2), encoding="utf-8")
    print(f"saved -> {out}")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    sys.exit(main(args.base))
