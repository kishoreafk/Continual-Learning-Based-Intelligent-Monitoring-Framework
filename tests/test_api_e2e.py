"""E2E tests for the demo API layer (FastAPI TestClient, no live server).

Mirrors the live verification but as repeatable pytest checks, including a
full session lifecycle with an injected fault and the safety invariants.
"""

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from demo.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def wait_status(client, sid, statuses, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = client.get(f"/api/sessions/{sid}").json()
        if s["status"] in statuses:
            return s
        time.sleep(0.5)
    return s


class TestMeta:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["device"] in ("cpu", "cuda")

    def test_info_lists_actions(self, client):
        r = client.get("/api/info")
        assert r.status_code == 200
        assert set(r.json()["actions"]) == {
            "NO_ACTION", "ADAPT", "RECALL", "REJECT", "ALERT"}

    def test_datasets_include_scripted(self, client):
        ids = [d["id"] for d in client.get("/api/datasets").json()]
        assert "scripted" in ids

    def test_dashboard_and_assets_served(self, client):
        r = client.get("/")
        assert r.status_code == 200 and b"Mission Control" in r.content
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200

    def test_dashboard_values_are_typable(self, client):
        html = client.get("/").text
        for field in ("p-window", "p-stride", "p-epochs", "p-seed",
                      "p-channels", "p-max-windows", "p-speed", "p-step",
                      "p-inj-duration"):
            assert f'id="{field}"' in html, f"missing input {field}"
        # no control that only offers a fixed set of values
        assert '<select id="p-speed"' not in html
        assert '<select id="p-step"' not in html
        assert '<select id="p-inj-duration"' not in html
        # every numeric control is a free-typed number input
        for field in ("p-speed", "p-step", "p-inj-duration"):
            assert f'id="{field}" type="number"' in html
        js = client.get("/static/app.js").text
        for field in ("p-speed", "p-step", "p-inj-duration"):
            assert f'"{field}"' in js
        for helper in ("readNumber", "readInt", "formError"):
            assert helper in js

    def test_openapi_documents_all_paths(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for p in ("/api/health", "/api/sessions", "/api/sessions/{sid}",
                  "/api/sessions/{sid}/start", "/api/sessions/{sid}/pause",
                  "/api/sessions/{sid}/step", "/api/sessions/{sid}/speed",
                  "/api/sessions/{sid}/inject", "/api/sessions/{sid}/stream",
                  "/api/sessions/{sid}/chart", "/api/sessions/{sid}/events",
                  "/api/sessions/{sid}/report",
                  "/api/sessions/{sid}/report/download"):
            assert p in paths, f"missing {p}"


class TestErrors:
    def test_unknown_session_404(self, client):
        assert client.get("/api/sessions/zzz").status_code == 404
        assert client.post("/api/sessions/zzz/step").status_code == 404
        assert client.delete("/api/sessions/zzz").status_code == 404

    def test_bad_dataset_400(self, client):
        assert client.post("/api/sessions", json={"dataset": "nope"}).status_code == 400

    def test_bad_inject_kind_400(self, client):
        r = client.post("/api/sessions", json={
            "dataset": "scripted", "epochs": 2, "speed": 1000})
        sid = r.json()["session_id"]
        wait_status(client, sid, ("ready", "error"))
        r = client.post(f"/api/sessions/{sid}/inject", json={"kind": "volcano"})
        assert r.status_code == 400
        client.delete(f"/api/sessions/{sid}")


class TestSessionLifecycle:
    def test_typed_custom_params_are_applied(self, client):
        params = {"dataset": "scripted", "window": 30, "stride": 3,
                  "epochs": 1, "channels": 8, "seed": 7, "speed": 500}
        r = client.post("/api/sessions", json=params)
        assert r.status_code == 200
        sid = r.json()["session_id"]
        s = wait_status(client, sid, ("ready", "error"))
        assert s["status"] == "ready", s.get("error")
        assert s["params"]["window"] == 30
        assert s["params"]["stride"] == 3
        assert s["params"]["epochs"] == 1
        assert s["params"]["channels"] == 8
        assert s["params"]["seed"] == 7
        # windowing follows the typed values: the scripted stream is 3600
        # raw steps, so window 30 / stride 3 yields 1191 windows.
        assert s["total_windows"] == 1191
        chart = client.get(f"/api/sessions/{sid}/chart",
                           params={"since": 0, "limit": 10}).json()
        assert len(chart["channels"]) == 8
        client.delete(f"/api/sessions/{sid}")

    def test_full_lifecycle_with_injected_fault(self, client):
        # create + wait for training
        r = client.post("/api/sessions", json={
            "dataset": "scripted", "epochs": 6, "speed": 1000, "channels": 12})
        assert r.status_code == 200
        sid = r.json()["session_id"]

        s = wait_status(client, sid, ("ready", "error"))
        assert s["status"] == "ready", s.get("error")

        # list + detail
        assert any(x["session_id"] == sid for x in client.get("/api/sessions").json())
        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["total_windows"] > 800
        assert len(detail["segments"]) == 8

        # manual stepping
        r = client.post(f"/api/sessions/{sid}/step", params={"n": 120})
        assert r.json()["playhead"] == 120

        # stream snapshot schema
        snap = client.get(f"/api/sessions/{sid}/stream",
                          params={"since": 100, "limit": 20}).json()
        assert len(snap["records"]) == 20
        rec = snap["records"][0]
        for key in ("t", "action", "gate", "reason", "anomaly", "drift",
                    "ood", "regime_sim", "exceed_count", "max_zscore"):
            assert key in rec

        # chart payload
        cd = client.get(f"/api/sessions/{sid}/chart",
                        params={"since": 0, "limit": 100}).json()
        assert len(cd["series"]) == 100 and len(cd["series"][0]) == 8

        # inject a fault, then stream everything at MAX speed
        r = client.post(f"/api/sessions/{sid}/inject",
                        json={"kind": "fault", "duration": 60})
        assert r.status_code == 200
        assert r.json()["injected"]["win_end"] - r.json()["injected"]["win_start"] == 60

        assert client.post(f"/api/sessions/{sid}/start").status_code == 200
        assert client.post(f"/api/sessions/{sid}/pause").status_code == 200
        assert client.post(f"/api/sessions/{sid}/speed",
                           json={"speed": 1000}).status_code == 200
        assert client.post(f"/api/sessions/{sid}/start").status_code == 200

        s = wait_status(client, sid, ("done",), timeout=600)
        assert s["status"] == "done"

        # full records + safety invariants
        total = client.get(f"/api/sessions/{sid}/stream",
                           params={"since": 0, "limit": 1}).json()["total_windows"]
        snap = client.get(f"/api/sessions/{sid}/stream",
                          params={"since": 0, "limit": total}).json()
        acts = {}
        for rec in snap["records"]:
            acts[rec["action"]] = acts.get(rec["action"], 0) + 1
        assert acts.get("ALERT", 0) + acts.get("REJECT", 0) > 0, \
            f"injected fault must be flagged, got {acts}"

        rep = client.get(f"/api/sessions/{sid}/report").json()
        assert rep["stats"]["unsafe_adaptations"] == 0, \
            "THE SAFETY INVARIANT: a fault must never be adapted on"
        assert rep["detection"]["precision"] == 1.0
        assert rep["detection_point_adjust"]["recall"] == 1.0
        assert rep["false_alarm_rate"] == 0.0
        assert rep["stats"]["recalls"] >= 1

        # events + download + delete
        evs = client.get(f"/api/sessions/{sid}/events").json()["events"]
        assert any(e["action"] in ("ALERT", "REJECT") for e in evs)
        dl = client.get(f"/api/sessions/{sid}/report/download")
        assert dl.status_code == 200
        assert "attachment" in dl.headers.get("content-disposition", "")
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
        assert client.get(f"/api/sessions/{sid}").status_code == 404

    def test_smap_session_streams_real_data(self, client):
        ids = [d["id"] for d in client.get("/api/datasets").json()]
        if "smap:P-1" not in ids:
            pytest.skip("SMAP data not downloaded")
        r = client.post("/api/sessions", json={
            "dataset": "smap:P-1", "epochs": 4, "speed": 1000, "max_windows": 600})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        s = wait_status(client, sid, ("ready", "error"))
        assert s["status"] == "ready", s.get("error")
        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["anomaly_windows"] > 0
        client.post(f"/api/sessions/{sid}/step", params={"n": 80})
        snap = client.get(f"/api/sessions/{sid}/stream",
                          params={"since": 0, "limit": 10}).json()
        assert len(snap["records"]) == 10
        client.delete(f"/api/sessions/{sid}")
