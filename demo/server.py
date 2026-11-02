"""SAGE-Monitor demo server.

Serves the REST API that wraps the sage_monitor framework AND the static
mission-control dashboard (demo/static). One process, one port:

    python -m demo.server            # http://localhost:8000
    uvicorn demo.server:app --port 8000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sage_monitor.config import SAGEConfig

from demo import data_sources as DS
from demo.sessions import MANAGER, SessionError

app = FastAPI(
    title="SAGE-Monitor Demo API",
    description=("REST API over the Safety-Gated Continual Monitoring "
                 "Framework: sessions, live streaming gate decisions, "
                 "safety metrics and dataset catalogue."),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class CreateSession(BaseModel):
    dataset: str = Field("scripted", examples=["scripted", "smap:P-1"])
    window: int = Field(60, ge=20, le=240)
    stride: int = Field(4, ge=1, le=64)
    epochs: int = Field(6, ge=1, le=50)
    speed: float = Field(25.0, ge=1.0, le=1000.0, description="windows/second")
    channels: int = Field(12, ge=2, le=64)
    seed: int = 42
    max_windows: int | None = Field(None, ge=50, description="cap for SMAP test windows")


class InjectEvent(BaseModel):
    kind: str = Field("fault", examples=["fault", "drift", "ood", "clean"])
    duration: int = Field(40, ge=1, le=2000, description="windows the event lasts")


class SpeedUpdate(BaseModel):
    speed: float = Field(25.0, ge=1.0, le=1000.0)


# ----------------------------------------------------------------------
# Meta / health
# ----------------------------------------------------------------------
@app.get("/api/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "torch": torch.__version__,
        "active_sessions": len([s for s in MANAGER.sessions.values()
                                if s.status not in ("done", "error")]),
    }


@app.get("/api/info", tags=["meta"])
def info():
    cfg = SAGEConfig()
    return {
        "framework": "SAGE-Monitor — Safety-Gated Continual Monitoring Framework",
        "actions": ["NO_ACTION", "ADAPT", "RECALL", "REJECT", "ALERT"],
        "pipeline": ("Detect Drift -> Verify Normality -> Safety Gate -> "
                     "Adapt Selectively -> Store Regime -> Recall When Needed"),
        "default_config": cfg.to_dict(),
        "demo_datasets": DS.catalog(),
    }


@app.get("/api/datasets", tags=["meta"])
def datasets():
    return DS.catalog()


# ----------------------------------------------------------------------
# Sessions
# ----------------------------------------------------------------------
@app.post("/api/sessions", tags=["session"])
def create_session(body: CreateSession):
    try:
        sess = MANAGER.create(dataset_id=body.dataset, window=body.window,
                              stride=body.stride, epochs=body.epochs,
                              speed=body.speed, channels=body.channels,
                              seed=body.seed, max_windows=body.max_windows)
    except SessionError as exc:
        raise HTTPException(400, str(exc))
    return {"session_id": sess.id, "status": sess.status,
            "dataset": sess.dataset_id, "params": sess.params}


@app.get("/api/sessions", tags=["session"])
def list_sessions():
    out = []
    for s in MANAGER.sessions.values():
        out.append({"session_id": s.id, "dataset": s.dataset_id,
                    "status": s.status, "playhead": s.playhead,
                    "total_windows": s.source["X"].size(0),
                    "created_at": s.created_at})
    return out


@app.get("/api/sessions/{sid}", tags=["session"])
def session_detail(sid: str):
    try:
        s = MANAGER.get(sid)
    except SessionError as exc:
        raise HTTPException(404, str(exc))
    return {"session_id": s.id, "dataset": s.dataset_id, "status": s.status,
            "error": s.error, "params": s.params, "speed": s.speed,
            "playhead": s.playhead, "total_windows": s.source["X"].size(0),
            "windows_streamed": len(s.records),
            "n_regimes": (len(s.controller.memory) if s.controller else 0),
            "stats": dict(s.controller.stats) if s.controller else {},
            "segments": s.source["meta"].segments,
            "anomaly_windows": int(s.source["Y"].sum())}


@app.post("/api/sessions/{sid}/start", tags=["session"])
def session_start(sid: str):
    try:
        MANAGER.get(sid).start()
    except SessionError as exc:
        code = 404 if "not found" in str(exc) else 400
        raise HTTPException(code, str(exc))
    return {"session_id": sid, "status": MANAGER.get(sid).status}


@app.post("/api/sessions/{sid}/pause", tags=["session"])
def session_pause(sid: str):
    try:
        MANAGER.get(sid).pause()
    except SessionError as exc:
        raise HTTPException(404, str(exc))
    return {"session_id": sid, "status": MANAGER.get(sid).status}


@app.post("/api/sessions/{sid}/step", tags=["session"])
def session_step(sid: str, n: int = Query(1, ge=1, le=500)):
    try:
        s = MANAGER.get(sid)
        s.step(n)
    except SessionError as exc:
        code = 404 if "not found" in str(exc) else 400
        raise HTTPException(code, str(exc))
    return {"session_id": sid, "status": s.status, "playhead": s.playhead}


@app.post("/api/sessions/{sid}/speed", tags=["session"])
def session_speed(sid: str, body: SpeedUpdate):
    try:
        MANAGER.get(sid).set_speed(body.speed)
    except SessionError as exc:
        raise HTTPException(404, str(exc))
    return {"session_id": sid, "speed": MANAGER.get(sid).speed}


@app.post("/api/sessions/{sid}/inject", tags=["session"])
def session_inject(sid: str, body: InjectEvent):
    try:
        seg = MANAGER.get(sid).inject(body.kind, body.duration)
    except SessionError as exc:
        code = 404 if "not found" in str(exc) else 400
        raise HTTPException(code, str(exc))
    return {"session_id": sid, "injected": seg,
            "playhead": MANAGER.get(sid).playhead}


@app.get("/api/sessions/{sid}/stream", tags=["streaming"])
def session_stream(sid: str, since: int = Query(0, ge=0),
                   limit: int = Query(400, ge=1, le=5000)):
    try:
        return MANAGER.get(sid).snapshot(since=since, limit=limit)
    except SessionError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/sessions/{sid}/chart", tags=["streaming"])
def session_chart(sid: str, since: int = Query(0, ge=0),
                  limit: int = Query(600, ge=10, le=5000)):
    try:
        return MANAGER.get(sid).chart_data(since=since, limit=limit)
    except SessionError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/sessions/{sid}/events", tags=["streaming"])
def session_events(sid: str, limit: int = Query(100, ge=1, le=1000)):
    try:
        s = MANAGER.get(sid)
        return {"session_id": sid, "events": s.events[-limit:]}
    except SessionError as exc:
        raise HTTPException(404, str(exc))


@app.get("/api/sessions/{sid}/report", tags=["evaluation"])
def session_report(sid: str):
    try:
        return MANAGER.get(sid).report()
    except SessionError as exc:
        code = 404 if "not found" in str(exc) else 400
        raise HTTPException(code, str(exc))


@app.get("/api/sessions/{sid}/report/download", tags=["evaluation"])
def session_report_download(sid: str):
    try:
        rep = MANAGER.get(sid).report()
    except SessionError as exc:
        code = 404 if "not found" in str(exc) else 400
        raise HTTPException(code, str(exc))
    return JSONResponse(rep, headers={
        "Content-Disposition": f'attachment; filename="sage_report_{sid}.json"'})


@app.delete("/api/sessions/{sid}", tags=["session"])
def session_delete(sid: str):
    try:
        MANAGER.delete(sid)
    except SessionError as exc:
        raise HTTPException(404, str(exc))
    return {"deleted": sid}


# ----------------------------------------------------------------------
# Static dashboard
# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main():  # pragma: no cover
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="SAGE-Monitor demo server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
