"""Session manager: owns SAGE-Monitor sessions and their streaming threads.

A session = one trained SAGEController + one window stream + a playhead.
A background thread feeds windows to controller.step() at a configurable
rate so the dashboard shows live gate decisions. All mutating calls are
guarded by a per-session lock; the REST API only touches state under that
lock, so the demo is safe to pause / step / inject at any moment.
"""

import threading
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import torch

from sage_monitor.config import SAGEConfig
from sage_monitor.controller import SAGEController
from sage_monitor import metrics as M

from demo import data_sources as DS

MAX_SESSIONS = 3


class SessionError(Exception):
    pass


class Session:
    def __init__(self, dataset_id: str, window: int = 60, stride: int = 4,
                 epochs: int = 6, speed: float = 25.0, channels: int = 12,
                 seed: int = 42, max_windows: Optional[int] = None):
        self.id = uuid.uuid4().hex[:12]
        self.dataset_id = dataset_id
        self.params = {"window": window, "stride": stride, "epochs": epochs,
                       "speed": speed, "channels": channels, "seed": seed}
        self.lock = threading.RLock()
        self.status = "loading"          # loading -> ready -> streaming -> paused -> done
        self.error: Optional[str] = None
        self.created_at = time.time()

        self.records: List[Dict] = []
        self.events: List[Dict] = []
        self.injections: List[Dict] = []
        self.playhead = 0
        self.speed = speed
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # ---- build the data source -------------------------------------
        if dataset_id == "scripted":
            self.source = DS.load_scripted(steps=4200, channels=channels,
                                           seg_len=600, seed=seed)
        elif dataset_id.startswith("smap:"):
            ch = dataset_id.split(":", 1)[1]
            self.source = DS.load_smap(ch, window=window, stride=stride,
                                       max_windows=max_windows)
        else:
            raise SessionError(f"unknown dataset '{dataset_id}'")
        # pristine copy for live-injection restore
        self.source["orig_X"] = self.source["X"].clone()
        self.source["orig_Y"] = self.source["Y"].clone()

        # precompute per-window channel means for the telemetry chart
        with torch.no_grad():
            means = self.source["X"][:, :, :8].mean(dim=1)  # (n_windows, <=8 ch)
        self.chart_series = means.numpy().astype(float).tolist()
        self.chart_channels = self.source["channel_names"][:8]
        self.labels = self.source["Y"].numpy().astype(int).tolist()

        self.cfg = SAGEConfig(
            n_channels=self.source["n_channels"], window=window,
            stride=stride, train_epochs=epochs, gate_min_obs=32,
            device="auto", seed=seed)
        self.controller: Optional[SAGEController] = None

    # ------------------------------------------------------------------
    def train(self):
        """Base-train the controller on the clean training split (blocking).

        Seeds torch/numpy from the session's seed so every demo session with
        the same parameters reproduces the same model and gate behaviour
        (the same discipline as scripts/run_local.py).
        """
        with self.lock:
            try:
                self.status = "training"
                torch.manual_seed(self.params["seed"])
                np.random.seed(self.params["seed"])
                ctrl = SAGEController(self.cfg)
                ctrl.train_base(self.source["train_X"], verbose=False)
                self.controller = ctrl
                self.status = "ready"
            except Exception as exc:  # pragma: no cover
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
                raise

    # ------------------------------------------------------------------
    def _loop(self):
        n = self.source["X"].size(0)
        while not self._stop.is_set():
            with self.lock:
                if self.status == "done":
                    return
                if self._pause.is_set():
                    pass  # fall through to the sleep below
                else:
                    i = self.playhead
                    if i >= n:
                        self.status = "done"
                        return
                    rec = self.controller.step(
                        self.source["X"][i],
                        true_fault=bool(self.source["Y"][i].item()))
                    rec["t"] = i
                    self.records.append(rec)
                    if rec["action"] not in ("NO_ACTION",):
                        self.events.append({
                            "t": i, "action": rec["action"],
                            "reason": rec["reason"], "anomaly": rec["anomaly"],
                            "drift": rec["drift"], "ood": rec["ood"],
                            "gate": rec["gate"],
                        })
                    self.playhead += 1
                    if self.playhead >= n:
                        self.status = "done"
                        return
            if self._pause.is_set():
                time.sleep(0.05)
            else:
                time.sleep(1.0 / max(self.speed, 1e-6))

    def start(self):
        with self.lock:
            if self.controller is None:
                raise SessionError("session is not trained yet")
            if self.status == "done":
                raise SessionError("stream already finished — reset to rerun")
            self._pause.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
            self.status = "streaming"

    def pause(self):
        with self.lock:
            self._pause.set()
            if self.status == "streaming":
                self.status = "paused"

    def step(self, n_steps: int = 1):
        """Manually advance n windows (used for stepping through in demos)."""
        with self.lock:
            if self.controller is None:
                raise SessionError("session is not trained yet")
            n = self.source["X"].size(0)
            for _ in range(max(1, min(n_steps, 500))):
                if self.playhead >= n:
                    self.status = "done"
                    break
                i = self.playhead
                rec = self.controller.step(
                    self.source["X"][i],
                    true_fault=bool(self.source["Y"][i].item()))
                rec["t"] = i
                self.records.append(rec)
                if rec["action"] != "NO_ACTION":
                    self.events.append({
                        "t": i, "action": rec["action"], "reason": rec["reason"],
                        "anomaly": rec["anomaly"], "drift": rec["drift"],
                        "ood": rec["ood"], "gate": rec["gate"]})
                self.playhead += 1
            else:
                if self.status in ("ready", "paused", "streaming"):
                    self.status = "paused" if self.playhead < n else "done"

    def set_speed(self, speed: float):
        self.speed = max(1.0, min(float(speed), 1000.0))

    def inject(self, kind: str, duration: int = 40) -> Dict:
        with self.lock:
            if self.source["kind"] != "scripted":
                raise SessionError("injection is only supported on the scripted dataset")
            if kind not in ("fault", "drift", "ood", "clean"):
                raise SessionError(
                    f"unknown inject kind '{kind}' — use fault | drift | ood | clean")
            seg = DS.inject(self.source, self.playhead, kind, duration)
            self.injections.append(seg)
            return seg

    # ------------------------------------------------------------------
    def snapshot(self, since: int = 0, limit: int = 400) -> Dict:
        with self.lock:
            recs = self.records[since:since + limit]
            return {
                "session_id": self.id,
                "dataset": self.dataset_id,
                "status": self.status,
                "error": self.error,
                "params": self.params,
                "speed": self.speed,
                "playhead": self.playhead,
                "total_windows": self.source["X"].size(0),
                "records": recs,
                "events": self.events[-60:],
                "injections": self.injections,
                "stats": dict(self.controller.stats) if self.controller else {},
                "n_regimes": len(self.controller.memory) if self.controller else 0,
                "segments": self.source["meta"].segments,
                "anomaly_windows": int(self.source["Y"].sum()),
            }

    def chart_data(self, since: int = 0, limit: int = 600) -> Dict:
        with self.lock:
            end = min(self.playhead, since + limit)
            return {
                "channels": self.chart_channels,
                "series": self.chart_series[since:end],
                "labels": self.labels[since:end],
                "playhead": self.playhead,
            }

    def report(self) -> Dict:
        with self.lock:
            if self.controller is None:
                raise SessionError("session is not trained yet")
            recs = self.records or self.controller.score_log
            rep = M.summarise(self.controller, recs,
                              self.source["meta"].segments,
                              self.source["Y"].numpy())
            rep["session_id"] = self.id
            rep["dataset"] = self.dataset_id
            rep["params"] = self.params
            rep["windows_streamed"] = len(self.records)
            rep["injections"] = self.injections
            return rep


class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, background_train: bool = True, **kwargs) -> Session:
        with self._lock:
            active = [s for s in self.sessions.values()
                      if s.status not in ("done", "error")]
            if len(active) >= MAX_SESSIONS:
                raise SessionError(
                    f"too many active sessions (max {MAX_SESSIONS}) — delete one first")
            sess = Session(**kwargs)
            self.sessions[sess.id] = sess
        if background_train:
            threading.Thread(target=_safe_train, args=(sess,), daemon=True).start()
        else:
            sess.train()
        return sess

    def get(self, sid: str) -> Session:
        sess = self.sessions.get(sid)
        if sess is None:
            raise SessionError(f"session '{sid}' not found")
        return sess

    def delete(self, sid: str):
        sess = self.get(sid)
        sess._stop.set()
        sess._pause.set()
        with self._lock:
            self.sessions.pop(sid, None)


def _safe_train(sess: Session):
    try:
        sess.train()
    except Exception:
        pass  # status/error recorded by Session.train


MANAGER = SessionManager()
