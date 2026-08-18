"""Evaluation metrics: detection, temporal, continual-learning and safety.

Safety metrics are the research centrepiece:
    UnsafeAdaptationRate = adaptation events containing anomalies / all adaptations
    AdaptationEfficiency = performance gain after adaptation / # parameter updates
"""

from typing import Dict, List, Optional

import numpy as np


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def point_adjust_f1(y_true, y_pred):
    """Point-adjust scoring (common in the SMAP/MSL literature): if any point in
    an anomaly interval is flagged, the whole interval counts as detected."""
    y_true = np.asarray(y_true).copy()
    y_pred = np.asarray(y_pred).copy()
    i = 0
    n = len(y_true)
    while i < n:
        if y_true[i]:
            j = i
            while j < n and y_true[j]:
                j += 1
            if y_pred[i:j].any():
                y_pred[i:j] = 1
            i = j
        else:
            i += 1
    return detection_metrics(y_true, y_pred)


def detection_delay(y_true: np.ndarray, y_pred: np.ndarray) -> List[int]:
    """Delay (in steps) between each anomaly-interval start and first detection."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    delays = []
    i = 0
    n = len(y_true)
    while i < n:
        if y_true[i]:
            j = i
            while j < n and y_true[j]:
                j += 1
            hits = np.where(y_pred[i:j])[0]
            delays.append(int(hits[0]) if len(hits) else -1)  # -1 = missed
            i = j
        else:
            i += 1
    return delays


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    normal = ~y_true
    if normal.sum() == 0:
        return 0.0
    return float((y_pred & normal).sum() / normal.sum())


# ----------------------------------------------------------------------
# Continual-learning metrics
# ----------------------------------------------------------------------
def segment_performance(records, segments, score_key="anomaly"):
    """Mean anomaly score per stream segment (used to see drift recovery)."""
    scores = np.array([r[score_key] for r in records])
    out = {}
    for s in segments:
        seg = scores[s["win_start"]:s["win_end"]]
        out[s["name"]] = float(seg.mean()) if len(seg) else float("nan")
    return out


def forgetting(performance_after: Dict[str, float],
               performance_before: Dict[str, float]) -> float:
    """Average performance drop on earlier regimes measured at the end."""
    keys = [k for k in performance_before if k in performance_after]
    if not keys:
        return 0.0
    return float(np.mean([performance_after[k] - performance_before[k] for k in keys]))


def unsafe_adaptation_rate(stats: Dict[str, int]) -> float:
    if stats["adaptations"] == 0:
        return 0.0
    return stats["unsafe_adaptations"] / stats["adaptations"]


def summarise(controller, records, segments, labels) -> Dict:
    """Full metric report for one run.

    The framework's fault output is the safety gate deciding REJECT or ALERT
    ("this looks like a fault — do not learn it"). That decision, not the raw
    internal anomaly score, is the detector output we score against the labels.
    """
    y_pred = np.array([
        1 if r["action"] in ("REJECT", "ALERT") else 0
        for r in records])
    win_labels = np.array([labels[min(i, len(labels) - 1)]
                           for i in range(len(records))]) if labels is not None \
        else np.zeros(len(records))
    det = detection_metrics(win_labels, y_pred)
    det_pa = point_adjust_f1(win_labels, y_pred)
    delays = detection_delay(win_labels, y_pred)
    seg_perf = segment_performance(records, segments)
    stats = controller.stats
    return {
        "detection": det,
        "detection_point_adjust": det_pa,
        "delays": delays,
        "missed": int(sum(1 for d in delays if d == -1)),
        "false_alarm_rate": false_alarm_rate(win_labels, y_pred),
        "segment_anomaly_score": seg_perf,
        "stats": dict(stats),
        "unsafe_adaptation_rate": unsafe_adaptation_rate(stats),
        "n_regimes": len(controller.memory),
    }
