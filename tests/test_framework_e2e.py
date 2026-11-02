"""E2E tests for the SAGE-Monitor framework itself (no HTTP layer).

Covers: stream construction integrity, config roundtrip, gate decisions,
the full train->stream->report pipeline with safety invariants, and
model save/load reproducibility.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sage_monitor.config import SAGEConfig
from sage_monitor.controller import SAGEController
from sage_monitor.datasets import make_windows, synth_multivariate
from sage_monitor.gate import Action, SafetyGate
from sage_monitor import metrics as M
from sage_monitor.streams import build_stream

from demo import data_sources as DS


# ----------------------------------------------------------------------
# Stream construction
# ----------------------------------------------------------------------
class TestStream:
    def test_segments_are_contiguous_and_cover_stream(self):
        base = synth_multivariate(4200, 12, seed=42)
        X, Y, meta = build_stream(base, window=60, stride=4, seg_len=600, seed=42)
        segs = meta.segments
        assert segs[0]["start"] == 0
        for a, b in zip(segs, segs[1:]):
            assert a["end"] == b["start"], "segments must tile the timeline"
        assert [s["name"] for s in segs] == [
            "A_normal", "gradual_drift", "B_normal", "sudden_drift_C",
            "C_normal", "FAULT", "B_return", "novel_OOD"]

    def test_labels_only_mark_fault(self):
        base = synth_multivariate(4200, 12, seed=42)
        X, Y, meta = build_stream(base, window=60, stride=4, seg_len=600, seed=42)
        fault = [s for s in meta.segments if s["name"] == "FAULT"][0]
        lab = Y.numpy()
        # every labelled window lies inside the FAULT segment ± the overlap
        # tail: a window covering the last fault step starts up to
        # (window-1)/stride ≈ 15 windows after the segment's nominal end
        for i in np.where(lab == 1)[0]:
            assert fault["win_start"] <= i < fault["win_end"] + 16
        # and there ARE labelled windows (the injected jitter half)
        assert lab.sum() > 0

    def test_window_shapes(self):
        base = synth_multivariate(1000, 5, seed=0)
        X, Y, meta = build_stream(base, window=60, stride=4, seg_len=250, seed=0)
        assert X.dim() == 3 and X.size(2) == 5 and X.size(1) == 60
        assert Y.size(0) == X.size(0)


# ----------------------------------------------------------------------
# Config + gate unit behaviour
# ----------------------------------------------------------------------
class TestConfigAndGate:
    def test_config_roundtrip(self):
        cfg = SAGEConfig(n_channels=9, window=45)
        d = cfg.to_dict()
        cfg2 = SAGEConfig.from_dict(d)
        assert cfg2 == cfg
        # unknown keys are dropped, not fatal
        cfg3 = SAGEConfig.from_dict({**d, "bogus_key": 1})
        assert cfg3 == cfg

    def _gate(self):
        return SafetyGate(SAGEConfig(gate_min_obs=32))

    def test_gate_stable_distribution_is_no_action(self):
        g = self._gate()
        a, score, reason = g.decide(
            confidence=0.9, anomaly=0.1, drift=0.1, agreement=0.9,
            ood=0.1, regime_sim=0.9, warm=True)
        assert a == Action.NO_ACTION and "stable" in reason

    def test_gate_blocks_persistent_fault(self):
        g = self._gate()
        kw = dict(confidence=0.9, drift=0.2, agreement=0.9, ood=0.6,
                  regime_sim=0.4, warm=True, exceed=5, exceed_frac=0.4,
                  max_zscore=12.0)
        a, _, reason = g.decide(anomaly=0.6, fault_confirmed=True, **kw)
        assert a == Action.REJECT
        a, _, reason = g.decide(anomaly=0.9, fault_confirmed=True, **kw)
        assert a == Action.ALERT

    def test_gate_fault_spike_gathers_evidence_not_learn(self):
        g = self._gate()
        a, _, reason = g.decide(
            confidence=0.9, anomaly=0.6, drift=0.2, agreement=0.9, ood=0.5,
            regime_sim=0.4, warm=True, exceed=5, exceed_frac=0.4,
            max_zscore=12.0, fault_confirmed=False)
        assert a == Action.NO_ACTION and "gathering" in reason

    def test_gate_novel_condition_monitored_only(self):
        g = self._gate()
        a, _, reason = g.decide(
            confidence=0.9, anomaly=0.8, drift=0.4, agreement=0.9, ood=0.95,
            regime_sim=0.3, warm=True, exceed=1, exceed_frac=0.9,
            max_zscore=9.0)
        assert a == Action.NO_ACTION and "novel" in reason

    def test_gate_recall_requires_verification(self):
        g = self._gate()
        a, _, reason = g.decide(
            confidence=0.9, anomaly=0.6, drift=0.2, agreement=0.9, ood=0.5,
            regime_sim=0.9, warm=True, verified_return=True,
            exceed=0, exceed_frac=0.0, max_zscore=1.0)
        assert a == Action.RECALL and "verified" in reason
        a, _, _ = g.decide(
            confidence=0.9, anomaly=0.6, drift=0.2, agreement=0.9, ood=0.5,
            regime_sim=0.9, warm=True, verified_return=None,
            exceed=0, exceed_frac=0.0, max_zscore=1.0)
        assert a != Action.RECALL

    def test_gate_cold_warmup_never_acts(self):
        g = self._gate()
        a, _, reason = g.decide(
            confidence=0.9, anomaly=0.9, drift=0.9, agreement=0.9, ood=0.9,
            regime_sim=0.1, warm=False)
        assert a == Action.NO_ACTION and "warm-up" in reason


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
class TestMetrics:
    def test_detection_metrics_exact(self):
        y = np.array([0, 0, 1, 1, 1, 0])
        p = np.array([0, 1, 1, 0, 1, 0])
        m = M.detection_metrics(y, p)
        assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1
        assert abs(m["precision"] - 2 / 3) < 1e-9
        assert abs(m["recall"] - 2 / 3) < 1e-9

    def test_point_adjust_credits_whole_interval(self):
        y = np.array([0, 1, 1, 1, 0])
        p = np.array([0, 1, 0, 0, 0])
        m = M.point_adjust_f1(y, p)
        assert m["recall"] == 1.0

    def test_delay_and_false_alarm(self):
        y = np.array([0, 0, 1, 1, 0])
        p = np.array([0, 0, 0, 1, 1])
        assert M.detection_delay(y, p) == [1]
        # one false positive on the last normal window out of 3 normal windows
        assert abs(M.false_alarm_rate(y, p) - 1 / 3) < 1e-9


# ----------------------------------------------------------------------
# Full pipeline (small, deterministic)
# ----------------------------------------------------------------------
class TestPipelineE2E:
    @pytest.fixture(scope="class")
    def run(self):
        torch.manual_seed(42)
        np.random.seed(42)
        base = synth_multivariate(4200, 12, seed=42)
        X, Y, meta = build_stream(base, window=60, stride=4, seg_len=600, seed=42)
        first_end = meta.segments[0]["win_end"]
        train_X = X[:first_end][Y[:first_end] == 0]
        cfg = SAGEConfig(n_channels=12, window=60, train_epochs=6,
                         gate_min_obs=32, device="cpu", seed=42)
        ctrl = SAGEController(cfg)
        ctrl.train_base(train_X, verbose=False)
        records = ctrl.run_stream(X, labels=Y, log_every=0)
        report = M.summarise(ctrl, records, meta.segments, Y.numpy())
        return ctrl, records, report, X, Y

    def test_every_window_processed(self, run):
        ctrl, records, report, X, Y = run
        assert len(records) == X.size(0)

    def test_point_adjust_recall_is_one(self, run):
        _, _, report, _, _ = run
        assert report["detection_point_adjust"]["recall"] == 1.0

    def test_precision_is_one(self, run):
        _, _, report, _, _ = run
        assert report["detection"]["precision"] == 1.0

    def test_no_false_alarms(self, run):
        _, _, report, _, _ = run
        assert report["false_alarm_rate"] == 0.0

    def test_no_unsafe_adaptations(self, run):
        """THE core safety invariant: no fault window is ever adapted on."""
        _, _, report, _, _ = run
        assert report["stats"]["unsafe_adaptations"] == 0
        assert report["unsafe_adaptation_rate"] == 0.0

    def test_regimes_were_learned_and_recalled(self, run):
        ctrl, _, report, _, _ = run
        assert report["stats"]["adaptations"] >= 2   # B and C learned
        assert report["stats"]["recalls"] >= 1       # B-return recalled
        assert len(ctrl.memory) >= 2

    def test_alerts_fired_on_fault(self, run):
        _, _, report, _, _ = run
        assert report["stats"]["alerts"] + report["stats"]["rejects"] > 0

    def test_save_load_roundtrip(self, run, tmp_path):
        ctrl, records, _, X, _ = run
        p = str(tmp_path / "model.pt")
        ctrl.save(p)
        ctrl2 = SAGEController(SAGEConfig.from_dict(ctrl.cfg.to_dict()))
        ctrl2.load(p)
        with torch.no_grad():
            z1 = ctrl.encoder.encode(X[:8])
            z2 = ctrl2.encoder.encode(X[:8])
        assert torch.allclose(z1, z2, atol=1e-5), "loaded model must reproduce latents"


# ----------------------------------------------------------------------
# Demo data sources (incl. real SMAP if downloaded)
# ----------------------------------------------------------------------
class TestDataSources:
    def test_scripted_loads(self):
        src = DS.load_scripted(steps=4200, channels=12, seed=42)
        assert src["X"].dim() == 3
        assert src["train_X"].size(0) > 50

    def test_injection_fault_changes_labels_and_data(self):
        src = DS.load_scripted(steps=4200, channels=12, seed=42)
        orig = src["X"].clone()
        seg = DS.inject(src, 150, "fault", duration=40)
        assert seg["win_start"] == 150 and seg["win_end"] == 190
        assert int(src["Y"][150:190].sum()) == 40
        assert not torch.equal(orig[150:190], src["X"][150:190])
        # data before the playhead untouched
        assert torch.equal(orig[:150], src["X"][:150])

    def test_injection_drift_is_labelled_normal(self):
        src = DS.load_scripted(steps=4200, channels=12, seed=42)
        seg = DS.inject(src, 100, "drift", duration=50)
        assert int(src["Y"][100:150].sum()) == 0

    def test_injection_clean_restores(self):
        src = DS.load_scripted(steps=4200, channels=12, seed=42)
        orig = src["X"].clone()
        DS.inject(src, 100, "fault", duration=30)
        DS.inject(src, 100, "clean", duration=30)
        assert torch.equal(orig[100:130], src["X"][100:130])

    def test_smap_channel_loads_with_labels(self):
        avail = DS.smap_available()
        if not avail:
            pytest.skip("SMAP data not downloaded")
        src = DS.load_smap("P-1", max_windows=1200)
        assert src["X"].size(0) == 1200
        assert src["anomaly_windows"] > 0, "P-1 labels must cover anomalies"
        kinds = {s["kind"] for s in src["meta"].segments}
        assert "anomaly" in kinds
