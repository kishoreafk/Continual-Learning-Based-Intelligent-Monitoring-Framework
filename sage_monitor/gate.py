"""Safety gate: decides whether the model is permitted to learn.

The model is NOT allowed to learn from every incoming observation. The gate
combines confidence, normality, drift evidence and teacher-student agreement
into a single decision score:

    G_t = w1*C_t + w2*(1 - A_t) + w3*D_t + w4*S_t

and issues one of the autonomous actions:
    NO_ACTION / ADAPT / RECALL / REJECT / ALERT
"""

from enum import Enum
from typing import Optional

import torch

from .config import SAGEConfig


class Action(str, Enum):
    NO_ACTION = "NO_ACTION"       # normal regime, keep monitoring
    ADAPT = "ADAPT"               # trusted drift -> learn the new normal
    RECALL = "RECALL"             # known regime returned -> reuse old model
    REJECT = "REJECT"             # suspected fault / unsafe -> do not learn
    ALERT = "ALERT"               # emergency / OOD -> raise anomaly


class SafetyGate:
    def __init__(self, cfg: SAGEConfig):
        self.cfg = cfg
        self.history = []

    def decide(
        self,
        confidence: float,
        anomaly: float,
        drift: float,
        agreement: float,
        ood: float,
        regime_sim: float,
        warm: bool,
        exceed: int = 0,
        exceed_frac: float = 0.0,
        max_zscore: float = 0.0,
        coherent_streak: bool = False,
        clean_streak: bool = False,
        fault_confirmed: bool = False,
        verified_return: Optional[bool] = None,
    ):
        """Return (action, gate_score, reason).

        Decision logic (drift/anomaly disentanglement):
          - RESIDUAL SHAPE, not raw exceedance count, separates conditions.
            ``max_zscore`` identifies concentrated extreme residual energy,
            while ``exceed_frac`` identifies a widespread novel condition.
            A moderate max z-score is allowed to reach verify-before-adapt;
            fixed channel counts cannot make that decision because a stale
            decoder can make every channel exceed on a learnable shift.
          - persistence (fault_confirmed) separates real faults from
            boundary transients.
          - a NEW regime (coherent shift) is certified by a sustained run of
            distributed low-exceedance windows (coherent_streak), never by the
            raw anomaly level: while calibrated to the old regime the anomaly
            score saturates on any unseen condition, so requiring anomaly < 0.5
            would deadlock adaptation. Fault residue inside the window breaks
            the run (exceedance >= 2) and is never certified.
          - RECALL requires verification (verified_return): cosine similarity
            in latent space is scale-invariant, so an affine variant of a known
            regime (e.g. B vs A) matches it while still scoring far outside its
            error band. Only matches that the candidate's adapter + calibration
            actually explain are recalled; the rest fall through to ADAPT.
          - OOD is reported as a monitoring signal; it saturates on every
            unseen condition and is not used to trigger actions.
        """
        cfg = self.cfg
        # A component fault has a concentrated extreme channel residual. A
        # novel/OOD condition has a less extreme residual spread across most
        # channels. The ordering makes the two bands mutually exclusive.
        fault_band = (exceed >= cfg.fault_exceed_count
                      and max_zscore >= cfg.fault_min_max_zscore)
        novel_band = (not fault_band
                      and max_zscore >= cfg.novel_min_max_zscore
                      and exceed_frac >= cfg.ood_exceed_frac)

        # genuinely novel condition: MOST channels unreconstructable. Monitor
        # only; unknown != fault and unknown != new normal. Checked first so a
        # widespread-novel reading can never escalate into a fault ALERT.
        if anomaly > 0.7 and novel_band:
            self._log(Action.NO_ACTION, 0.0, "novel condition: monitoring only")
            return Action.NO_ACTION, 0.0, "novel condition: monitoring only"

        # emergency: persistent, confirmed component fault (a FEW channels
        # deviating, sustained across windows)
        if fault_confirmed and anomaly > cfg.alert_anomaly and fault_band:
            self._log(Action.ALERT, 0.0, "emergency: persistent component fault")
            return Action.ALERT, 0.0, "emergency: persistent component fault"

        # persistent component fault -> never learned as a new normal. This
        # blocks the catastrophic "fault -> learned as normal" failure mode.
        if fault_confirmed and anomaly > 0.5 and fault_band:
            self._log(Action.REJECT, 0.0, "persistent component fault: blocked")
            return Action.REJECT, 0.0, "persistent component fault: blocked"

        # transient fault-band spikes: gather evidence, do not learn
        if anomaly > 0.5 and fault_band:
            self._log(Action.NO_ACTION, 0.0, "fault-band spike: gathering evidence")
            return Action.NO_ACTION, 0.0, "fault-band spike: gathering evidence"

        if not warm:
            self._log(Action.NO_ACTION, 0.0, "warm-up: statistics not ready")
            return Action.NO_ACTION, 0.0, "warm-up: statistics not ready"

        # stable operating condition
        if drift < 0.3 and anomaly < 0.5:
            g = self._gate_score(confidence, anomaly, drift, agreement)
            self._log(Action.NO_ACTION, g, "distribution stable")
            return Action.NO_ACTION, g, "distribution stable"

        g = self._gate_score(confidence, anomaly, drift, agreement)

        # high anomaly without a fault signature. Whether this is a learnable
        # coherent regime shift or an unlearnable condition is decided by
        # verify-before-adapt (the probe), not by exceedance counting: with a
        # stale decoder every unseen condition exceeds, so only learnability
        # separates "new normal" from "unlearnable".
        if anomaly >= 0.5:
            if regime_sim >= cfg.regime_recall_similarity and verified_return is True:
                self._log(Action.RECALL, g, "known regime returned (verified)")
                return Action.RECALL, g, "known regime returned (verified)"
            if coherent_streak:
                self._log(Action.ADAPT, g, "shift candidate: probing learnability")
                return Action.ADAPT, g, "shift candidate: probing learnability"
            self._log(Action.NO_ACTION, g, "shift candidate: gathering evidence")
            return Action.NO_ACTION, g, "shift candidate: gathering evidence"

        # normal-looking data under drift
        if drift >= 0.3:
            if regime_sim >= cfg.regime_recall_similarity and verified_return is True:
                self._log(Action.RECALL, g, "known regime returned (verified)")
                return Action.RECALL, g, "known regime returned (verified)"
            if g >= cfg.gate_threshold:
                self._log(Action.ADAPT, g, "safe to learn new normal")
                return Action.ADAPT, g, "safe to learn new normal"

        self._log(Action.NO_ACTION, g, "ambiguous: no action taken")
        return Action.NO_ACTION, g, "ambiguous: no action taken"

    def _gate_score(self, confidence, anomaly, drift, agreement):
        return (
            self.cfg.gate_w_confidence * confidence
            + self.cfg.gate_w_normality * (1.0 - anomaly)
            + self.cfg.gate_w_drift * drift
            + self.cfg.gate_w_agreement * agreement
        )

    def _log(self, action, score, reason):
        self.history.append((action, score, reason))


@torch.no_grad()
def teacher_student_agreement(teacher_out, student_out):
    """Agreement in [0,1] between frozen teacher and adapting student.

    Measured in latent space (not reconstruction space) because the adaptive
    output heads legitimately diverge while the student fits a new regime.
    """
    diff = (teacher_out["z"] - student_out["z"]).pow(2).mean(dim=-1)
    return float(torch.exp(-diff / 0.5).mean().item())
