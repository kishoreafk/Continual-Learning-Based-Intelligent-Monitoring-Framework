"""SAGEController: the full continual-monitoring loop.

    Observe -> Encode -> [Drift | Anomaly | OOD] -> Safety Gate
        -> {NO_ACTION, ADAPT, RECALL, REJECT, ALERT}
        -> adapter update / regime recall / replay
"""

import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SAGEConfig
from .encoder import TemporalEncoder
from .heads import AnomalyHead, DriftDetector, OODHead
from .memory import RegimeMemory
from .gate import Action, SafetyGate, teacher_student_agreement
from . import losses as L


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


class SAGEController:
    def __init__(self, cfg: SAGEConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

        self.encoder = TemporalEncoder(cfg).to(self.device)
        self.teacher = copy.deepcopy(self.encoder).to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.anomaly_head = AnomalyHead(cfg).to(self.device)
        self.drift = DriftDetector(cfg).to(self.device)
        self.ood = OODHead(cfg).to(self.device)
        self.memory = RegimeMemory(cfg)
        self.gate = SafetyGate(cfg)

        self.n_trusted = 0
        self._recent_drift: List[float] = []
        self._recent_windows: List[torch.Tensor] = []  # trailing windows (cpu)
        self._recent_exceed: List[int] = []            # their exceedance counts
        self._recent_max_z: List[float] = []            # strongest channel residual
        self._coherent_streak = 0
        self._clean_streak = 0
        self._localized_streak = 0
        self._trial_countdown = 0
        self._trial_cooldown = 0
        self._trial_saved_adapter = None  # adapter to restore on failed trial
        self._trial_saved_calib = ([], None, None)
        self._adapt_cooldown = 0  # windows before re-probing a failed ADAPT
        self._refine_streak = 0   # consecutive verify failures on a young regime
        self.stats = {
            "adaptations": 0,
            "unsafe_adaptations": 0,
            "recalls": 0,
            "trial_recalls": 0,
            "rejects": 0,
            "alerts": 0,
            "regimes_created": 0,
        }
        self.score_log: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Phase 1 — supervised/base training on labeled normal telemetry
    # ------------------------------------------------------------------
    def train_base(self, windows: torch.Tensor, epochs: Optional[int] = None,
                   verbose: bool = True):
        """Train the frozen base encoder + anomaly head on normal windows."""
        epochs = epochs or self.cfg.train_epochs
        windows = windows.float().to(self.device)
        n = windows.size(0)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(windows),
            batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        params = list(self.encoder.parameters()) + list(self.anomaly_head.parameters())
        opt = torch.optim.Adam(params, lr=self.cfg.train_lr,
                               weight_decay=self.cfg.weight_decay)
        self.encoder.train()
        self.anomaly_head.train()
        H = self.cfg.horizon

        for ep in range(epochs):
            tot = 0.0
            for (x,) in loader:
                out = self.encoder(x)
                future = x[:, -H:, :].roll(-1, dims=1)  # cheap forecast target
                l_recon = L.reconstruction_loss(out["recon"], x)
                l_fc = L.forecast_loss(out["forecast"], future)
                loss = self.cfg.lambda_recon * l_recon + self.cfg.lambda_forecast * l_fc
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tot += loss.item() * x.size(0)
            if verbose:
                print(f"  [base epoch {ep + 1}/{epochs}] loss={tot / n:.4f}")

        # freeze base, keep adapter trainable
        for p in self.encoder.base_parameters():
            p.requires_grad_(False)
        self._sync_teacher()
        self.encoder.eval()
        self._seed_statistics(windows)

    @torch.no_grad()
    def _seed_statistics(self, windows: torch.Tensor):
        """Initialise drift reference, OOD head, anomaly stats, first regime."""
        windows = windows.float().to(self.device)
        z = self.encoder.encode(windows)
        self.drift.update_reference(z)
        self.ood.fit(z, z.mean(0, keepdim=True))
        out = self.encoder(windows)
        regime, _, created = self.memory.assign_or_create(
            z, self.encoder.get_adapter_state(), x=windows[:self.cfg.memory_per_regime])
        self.anomaly_head.set_from(windows, out["recon"], regime.regime_id)
        self._store_band(regime)
        regime.threshold = self.anomaly_head.dynamic_threshold()
        self.n_trusted = windows.size(0)
        if created:
            self.stats["regimes_created"] += 1

    def _sync_teacher(self):
        self.teacher.load_state_dict(self.encoder.state_dict())

    @torch.no_grad()
    def _store_band(self, regime, windows=None, recon=None):
        """Snapshot the calibrated error band onto the regime.

        When (windows, recon) are given, the band is computed from ONLY that
        regime's windows — regime-pure. Otherwise the current global band is
        used (seed time, before regimes diverge).
        """
        if windows is not None and recon is not None:
            band = self.anomaly_head.band_from(windows, recon)
        else:
            band = self.anomaly_head.band()
        if band is not None:
            regime.err_mu, regime.err_sd = band

    @torch.no_grad()
    def _ema_teacher(self):
        d = self.cfg.ema_decay
        for pt, ps in zip(self.teacher.parameters(), self.encoder.parameters()):
            pt.data.mul_(d).add_(ps.data, alpha=1 - d)

    # ------------------------------------------------------------------
    # Streaming step — one observation window at a time
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score_window(self, x: torch.Tensor):
        out_s = self.encoder(x)
        out_t = self.teacher(x)
        z = out_s["z"]
        regime, sim = self.memory.nearest(z)
        # REGIME-AWARE, BAND-COMPATIBLE scoring: a window is scored against
        # the error band of the most similar regime whose band actually fits
        # the window's reconstruction error. Cosine-nearest alone is unsafe —
        # affine variants share latent direction (A vs B cosine ~0.93), so the
        # nearest prototype flip-flops window by window and the score would
        # oscillate between "normal" and saturated depending on which regime's
        # band happens to be picked. A band that does not fit the error is
        # simply not used, and the global band applies instead.
        band = None
        if self.memory.regimes:
            err = float(self.anomaly_head.window_error(
                x, out_s["recon"]).mean().item())
            best_sim = -1.0
            for r in self.memory.regimes:
                if r.err_mu is None or r.err_sd is None:
                    continue
                sim_r = self.memory._similarity(z, r.prototype)
                if sim_r < self.cfg.regime_recall_similarity:
                    continue
                sd_eff = max(r.err_sd, 0.15 * r.err_mu)
                if abs(err - r.err_mu) <= 2.0 * sd_eff and sim_r > best_sim:
                    band = (r.err_mu, r.err_sd)
                    best_sim = sim_r
        anomaly = float(self.anomaly_head(
            z, x, out_s["recon"], band=band).mean().item())
        drift = float(min(0.6 * self.drift.score(z) + 0.4 * self.drift.ph_score(), 1.0))
        ood = self.ood.score(z)
        agreement = teacher_student_agreement(out_t, out_s)
        confidence = max(0.0, min(1.0, agreement))
        return {
            "z": z, "out_s": out_s,
            "anomaly": anomaly, "drift": drift, "ood": ood,
            "agreement": agreement, "confidence": confidence,
            "regime": regime, "sim": sim,
        }

    @torch.no_grad()
    def _verify_return_best(self, z: torch.Tensor, x: torch.Tensor):
        """Find the regime that VERIFIEDLY explains this window, or None.

        Cosine-nearest alone is not a safe recall selector: affine variants
        (A vs B) share latent direction, so nearest() flip-flops between them
        window by window. We try every regime with sufficient similarity, most
        similar first, and return the first whose adapter + error band truly
        reproduce the window. Side-effect free: the adapter is restored.
        """
        cands = []
        for r in self.memory.regimes:
            sim_r = self.memory._similarity(z, r.prototype)
            if sim_r >= self.cfg.regime_recall_similarity:
                cands.append((sim_r, r))
        if not cands:
            return None
        cands.sort(key=lambda t: -t[0])
        saved = self.encoder.get_adapter_state()
        try:
            for sim_r, cand in cands:
                if not cand.adapter_state or cand.err_mu is None:
                    continue
                self.encoder.set_adapter_state(cand.adapter_state)
                out = self.encoder(x)
                err = self.anomaly_head.window_error(x, out["recon"])
                score = self.anomaly_head.score_errors(
                    err, cand.err_mu, cand.err_sd)
                if bool((score < 0.5).all().item()):
                    return cand
            return None
        finally:
            self.encoder.set_adapter_state(saved)

    def step(self, x: torch.Tensor, true_fault: bool = False) -> Dict:
        """Process one window through the monitoring loop.

        `true_fault` is only used for metric bookkeeping (unsafe-adaptation
        accounting) — the gate itself never sees it.
        """
        x = x.float().to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        self._recent_windows.append(x[0].detach().cpu())
        if len(self._recent_windows) > 64:
            self._recent_windows.pop(0)
        s = self._score_window(x)
        warm = self.n_trusted >= self.cfg.gate_min_obs
        threshold = self.anomaly_head.dynamic_threshold()

        # advance the change detector once per observation with the raw latent
        # distance, then combine it into the drift score
        dist = self.drift.distance(s["z"])
        self.drift.feed(dist)
        s["drift"] = float(min(0.6 * self.drift.score(s["z"])
                               + 0.4 * self.drift.ph_score(), 1.0))

        # Residual shape is geometry-robust: the maximum channel z-score
        # catches concentrated fault energy, while the fraction of exceeding
        # channels identifies a widespread novel condition. A raw count cannot
        # distinguish either from a learnable shift when channel geometry
        # changes.
        z = self.anomaly_head.channel_zscores(x, s["out_s"]["recon"])
        if z is None:
            exceed_count, max_zscore = 0, 0.0
        else:
            exceed_count = int((z > self.cfg.exceed_z_thr).sum().item())
            max_zscore = float(z.max().item())
        exceed = exceed_count / max(1, self.cfg.n_channels)
        fault_band = (exceed_count >= self.cfg.fault_exceed_count
                      and max_zscore >= self.cfg.fault_min_max_zscore)
        novel_band = (not fault_band
                      and max_zscore >= self.cfg.novel_min_max_zscore
                      and exceed >= self.cfg.ood_exceed_frac)
        self._recent_exceed.append(exceed_count)
        self._recent_max_z.append(max_zscore)
        if len(self._recent_exceed) > 64:
            self._recent_exceed.pop(0)
            self._recent_max_z.pop(0)

        # A high-similarity match is only a true RECALL if a candidate
        # regime's adapter + calibration actually explain the data. Affine
        # variants of a known regime (e.g. B vs A) share direction with it in
        # latent space (cosine similarity is scale-invariant) but score far
        # outside its error band; they must be ADAPTed to, not recalled.
        # Verification runs for ANY high-similarity match — the drift branch
        # (anomaly < 0.5, drift high) previously recalled without checking and
        # flip-flopped the active adapter between aligned regimes window by
        # window, which is what made B's score oscillate 0.02 <-> 0.99.
        verified_regime = None
        if s["sim"] >= self.cfg.regime_recall_similarity:
            verified_regime = self._verify_return_best(s["z"], x)
        verified_return = verified_regime is not None

        # YOUNG-REGIME REFINEMENT: a regime committed mid-transition carries a
        # band measured on transitional windows; the settled stream then fails
        # its verification window after window (verify best returns None while
        # cosine similarity stays high), which would oscillate the anomaly
        # score forever between recalled and unrecalled values. Instead of
        # spawning duplicate slots, re-probe on the newest windows and refine
        # the regime in place. Never refine while a fault or novel band is
        # active — fault residue / novel data must not enter regime memory.
        if (verified_regime is None
                and s["anomaly"] >= 0.5
                and not fault_band and not novel_band
                and s["regime"] is not None
                and s["regime"].n_seen <= self.cfg.regime_refine_max_seen
                and s["sim"] >= self.cfg.regime_recall_similarity
                and self._trial_countdown == 0
                and self._trial_cooldown == 0):
            self._refine_streak += 1
            if self._refine_streak >= self.cfg.refine_persist:
                self._refine_streak = 0
                self._refine_young_regime(s["regime"], x)
                verified_regime = self._verify_return_best(s["z"], x)
                verified_return = verified_regime is not None
                s2 = self._score_window(x)
                s["anomaly"] = s2["anomaly"]
        else:
            self._refine_streak = 0

        # persistent component-fault evidence: anomalous windows with a few
        # deviating sensors. Persistence separates real faults from transients.
        fault_ev = s["anomaly"] >= 0.5 and fault_band
        self._localized_streak = self._localized_streak + 1 if fault_ev else 0
        fault_confirmed = self._localized_streak >= self.cfg.fault_confirm_streak

        # a clean run certifies a coherent new regime. The streak counts ANY
        # sustained anomalous run — fault-band windows are handled by the
        # gate's own fault branches and by the probe's absolute-error cap, so
        # excluding them here would let stale-decoder noise reset the streak
        # forever and starve legitimate regime shifts of their ADAPT probe.
        distributed = s["anomaly"] >= 0.5
        self._coherent_streak = self._coherent_streak + 1 if distributed else 0
        coherent_streak = self._coherent_streak >= self.cfg.coherent_shift_persist

        clean = s["anomaly"] < 0.5
        self._clean_streak = self._clean_streak + 1 if clean else 0
        clean_streak = self._clean_streak >= self.cfg.coherent_shift_persist

        action, g, reason = self.gate.decide(
            confidence=s["confidence"], anomaly=s["anomaly"], drift=s["drift"],
            agreement=s["agreement"], ood=s["ood"], regime_sim=s["sim"], warm=warm,
            exceed=exceed_count, exceed_frac=exceed, max_zscore=max_zscore,
            coherent_streak=coherent_streak,
            clean_streak=clean_streak, fault_confirmed=fault_confirmed,
            verified_return=verified_return)

        # verify-before-alarm: a sustained fault-band reading can also be a
        # RETURNING REGIME whose recalled adapter/calibration no longer fit.
        # Before escalating, give the nearest known regime a trial: restore its
        # adapter + calibration and watch the exceedance. A returning regime
        # resolves (drops below the fault band); a real fault persists and the
        # ALERT fires after the trial fails.
        if self._trial_countdown > 0:
            self._trial_countdown -= 1
            if not fault_band:
                # trial succeeded: a known regime explains the data. Commit it
                # for real now (drift reference + calibration re-anchor), not
                # during the fault — state changes only on proven success.
                self._recall(s, regime=verified_regime)
                self.stats["recalls"] += 1
                self._trial_saved_adapter = None
                self._trial_countdown = 0
        elif self._trial_cooldown > 0:
            self._trial_cooldown -= 1

        if (action == Action.ALERT
                and self._trial_countdown == 0
                and self._trial_cooldown == 0
                and exceed_count < self.cfg.trial_max_exceed
                and s["regime"] is not None and s["regime"].windows):
            # first sustained fault evidence: try the nearest known regime's
            # ADAPTER + CALIBRATION only — no drift/memory side effects. Those
            # are committed only if the trial window resolves the exceedance.
            cand = s["regime"]
            self._trial_saved_adapter = self.encoder.get_adapter_state()
            saved_errors = list(self.anomaly_head._errors)
            saved_ch_sd = (self.anomaly_head._ch_sd.clone()
                           if self.anomaly_head._ch_sd is not None else None)
            saved_lock = self.anomaly_head._lock
            self._trial_saved_calib = (saved_errors, saved_ch_sd, saved_lock)
            if cand.adapter_state:
                self.encoder.set_adapter_state(cand.adapter_state)
            mem = torch.stack(cand.windows).to(self.device)
            out_mem = self.encoder(mem)
            self.anomaly_head.set_from(mem, out_mem["recon"], cand.regime_id)
            self.stats["trial_recalls"] += 1
            self._localized_streak = 0
            self._trial_countdown = self.cfg.trial_windows
            action, g, reason = Action.RECALL, 0.0, "trial recall: verifying known regime"
        elif (action == Action.ALERT and self._trial_countdown > 0):
            # fault evidence re-confirmed during an active trial: trial failed.
            # Restore the adapter + calibration exactly (never let a fault
            # contaminate the model or its calibration).
            if self._trial_saved_adapter is not None:
                self.encoder.set_adapter_state(self._trial_saved_adapter)
                self._trial_saved_adapter = None
            saved_errors, saved_ch_sd, saved_lock = getattr(
                self, "_trial_saved_calib", ([], None, None))
            if saved_errors:
                self.anomaly_head._errors = saved_errors
            self.anomaly_head._ch_sd = saved_ch_sd
            self.anomaly_head._lock = saved_lock
            self._trial_countdown = 0
            self._trial_cooldown = self.cfg.trial_cooldown

        # cooldown after a failed learnability probe: don't re-probe every window
        if self._adapt_cooldown > 0:
            self._adapt_cooldown -= 1
            if action == Action.ADAPT:
                action, g, reason = (Action.NO_ACTION, g,
                                     "probe cooldown: not re-testing yet")

        # never learn while a verify-before-alarm trial is in flight
        if (action == Action.ADAPT
                and (self._trial_countdown > 0 or self._trial_cooldown > 0)):
            action, g, reason = Action.NO_ACTION, g, "trial active: no learning while verifying"

        # a trial recall must not run the normal RECALL path again (its state
        # changes were already applied when the trial started / succeeded)
        if action == Action.RECALL and self._trial_countdown > 0:
            record = {"action": action.value, "gate": g, "reason": reason,
                      "anomaly": s["anomaly"], "drift": s["drift"], "ood": s["ood"],
                      "regime_sim": s["sim"], "threshold": threshold,
                      "exceedance": exceed, "exceed_count": exceed_count,
                      "max_zscore": max_zscore}
            self.score_log.append(record)
            return record

        record = {"action": action.value, "gate": g, "reason": reason,
                  "anomaly": s["anomaly"], "drift": s["drift"], "ood": s["ood"],
                  "regime_sim": s["sim"], "threshold": threshold,
                  "exceedance": exceed, "exceed_count": exceed_count,
                  "max_zscore": max_zscore}

        if action == Action.ADAPT:
            committed = self._try_adapt(x, s)
            if committed:
                self.stats["adaptations"] += 1
                self._coherent_streak = 0
                if true_fault:
                    self.stats["unsafe_adaptations"] += 1
            else:
                # the condition is not learnable by this model: monitor only
                self._coherent_streak = 0
                action, g, reason = Action.NO_ACTION, g, "unlearnable condition: monitoring only"
                record["action"] = action.value
                record["reason"] = reason
        elif action == Action.RECALL:
            self._recall(s, regime=verified_regime)
            self.stats["recalls"] += 1
        elif action == Action.REJECT:
            # rejected observations are identified but never stored: they must
            # not contaminate regime memory, the drift reference, or the
            # anomaly calibration.
            self.stats["rejects"] += 1
        elif action == Action.ALERT:
            self.stats["alerts"] += 1
        else:
            # NO_ACTION on a verified-normal window: its error is trusted
            # evidence about the CURRENT regime's band. Both gates matter:
            # the score gate keeps the band tight enough that real faults
            # still clear the ALERT threshold, and the exceedance gate keeps
            # any window with a fault signature out of the calibration.
            if (warm and s["anomaly"] < 0.5 and self._trial_countdown == 0
                    and exceed_count < self.cfg.fault_exceed_count):
                with torch.no_grad():
                    err = self.anomaly_head.window_error(
                        x, s["out_s"]["recon"]).view(-1)
                    self.anomaly_head._errors.extend(err.tolist())
                    cap = self.anomaly_head.buf_size
                    self.anomaly_head._errors = self.anomaly_head._errors[-cap:]

        self.score_log.append(record)
        return record

    # ------------------------------------------------------------------
    # Adaptation — parameter-efficient, replay-protected
    # ------------------------------------------------------------------
    def _try_adapt(self, x: torch.Tensor, s: Dict) -> bool:
        """Verify-before-adapt: a coherent-shift candidate is only committed
        as a new regime if the model can actually absorb it.

        A few adaptation steps run on the FIT split of the certified trailing
        windows, then the HELD-OUT split is re-scored against the CURRENT error
        band. Learnable conditions (e.g. B: every channel proportional to one
        latent) drop back inside the band; unlearnable ones (novel conditions
        whose per-channel structure no pooled latent can carry, or fault
        residue jitter) stay far outside it and are never absorbed into
        memory. Scoring a held-out split prevents a novel condition from
        overfitting itself into the band during the probe. On failure the
        adapter and calibration are restored exactly.
        """
        recent = torch.stack(
            self._recent_windows[-self.cfg.coherent_shift_persist:]).to(self.device)
        if recent.size(0) < self.cfg.coherent_shift_persist:
            return False
        # ABSOLUTE safety net: the certification streak can be fooled at
        # segment boundaries, where fault residue rides inside the certified
        # trailing windows. Strong evidence now requires both a multi-channel
        # exceedance and the concentrated extreme max-z signature of a fault.
        # A stale decoder on a legitimate shift can exceed on many channels,
        # but its max z-score remains below the fault boundary.
        recent_exc = self._recent_exceed[-self.cfg.coherent_shift_persist:]
        recent_max_z = self._recent_max_z[-self.cfg.coherent_shift_persist:]
        strong = sum(
            e >= self.cfg.fault_exceed_count and z >= self.cfg.fault_min_max_zscore
            for e, z in zip(recent_exc, recent_max_z))
        if (len(recent_exc) >= self.cfg.coherent_shift_persist
                and 2 * strong >= len(recent_exc)):
            self._adapt_cooldown = self.cfg.adapt_fail_cooldown
            return False
        band = self.anomaly_head.band()
        if band is None:
            return False
        band_mu, band_sd = band
        saved_adapter = self.encoder.get_adapter_state()
        saved_errors = list(self.anomaly_head._errors)
        saved_ch_sd = (self.anomaly_head._ch_sd.clone()
                       if self.anomaly_head._ch_sd is not None else None)
        saved_lock = self.anomaly_head._lock

        # fit / held-out split of the certified trailing windows
        n_fit = max(1, int(recent.size(0) * 0.6))
        fit_x, hold_x = recent[:n_fit], recent[n_fit:]
        hold = hold_x if hold_x.numel() else fit_x

        with torch.no_grad():
            out0 = self.encoder(hold)
            err_before = float(self.anomaly_head.window_error(
                hold, out0["recon"]).mean().item())

        self._probe_adapt(fit_x)

        with torch.no_grad():
            out = self.encoder(hold)
            err = self.anomaly_head.window_error(hold, out["recon"])
            score = self.anomaly_head.score_errors(err, band_mu, band_sd)
            err_after = float(err.mean().item())
            in_band = bool((score < 0.5).float().mean().item() >= 0.9)
            # SCALE-INVARIANT learnability: a larger-amplitude coherent regime
            # can never fit a tighter regime's absolute band, yet it is fully
            # learnable. Commit if the probe substantially reduces the error
            # AND the result stays in a sane multiple of the current band
            # (genuinely novel conditions stay far above even after probing,
            # and the gate's exceedance certification keeps faults out here).
            learned = (err_before > 1e-6
                       and err_after <= (1.0 - self.cfg.adapt_min_improvement) * err_before
                       and err_after <= self.cfg.adapt_max_abs_ratio * band_mu)
            ok = in_band or learned

        if not ok:
            self.encoder.set_adapter_state(saved_adapter)
            self.anomaly_head._errors = saved_errors
            self.anomaly_head._ch_sd = saved_ch_sd
            self.anomaly_head._lock = saved_lock
            # don't re-probe every window while the condition stays unlearnable
            self._adapt_cooldown = self.cfg.adapt_fail_cooldown
            return False

        self._commit_adapt(x, s, recent)
        return True

    def _probe_adapt(self, recent: torch.Tensor):
        """Throwaway trial adaptation on the certified windows.

        Runs with the encoder in EVAL mode: with GroupNorm (mode-independent)
        and dropout disabled, the probe is deterministic, so the probe's
        re-scoring matches exactly what streaming scoring will see afterwards.
        (Running in train() mode would mutate normalization statistics and
        make probe results disagree with eval-mode scoring.)

        Consistency anchors the student to the trusted teacher on REPLAY
        (trusted) data only — the new regime's latent must be free to move,
        otherwise the adapter can never absorb the shift.
        """
        was_training = self.encoder.training
        self.encoder.eval()
        opt = torch.optim.Adam(self.encoder.adaptive_parameters(),
                               lr=self.cfg.adapt_lr)
        H = self.cfg.horizon
        replay_x = self.memory.replay_windows(self.cfg.adapt_batch, self.device)
        for _ in range(self.cfg.adapt_epochs):
            out = self.encoder(recent)
            future = recent[:, -H:, :].roll(-1, dims=1)
            loss = (self.cfg.lambda_recon * L.reconstruction_loss(out["recon"], recent)
                    + self.cfg.lambda_forecast * L.forecast_loss(out["forecast"], future))
            if replay_x is not None:
                r_out = self.encoder(replay_x)
                with torch.no_grad():
                    t_out = self.teacher(replay_x)
                loss += self.cfg.lambda_replay * L.replay_loss(r_out["recon"], replay_x)
                loss += self.cfg.lambda_consistency * L.consistency_loss(r_out["z"], t_out["z"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if was_training:
            self.encoder.train()

    def _commit_adapt(self, x: torch.Tensor, s: Dict, recent: torch.Tensor):
        """Bookkeeping once a condition is certified learnable: memory
        absorption, teacher EMA, drift re-anchor, and the calibration locked
        to the CERTIFIED trailing windows — the regime's operating condition."""
        settle = s["drift"] < 0.35  # stream stable -> finalize any pending regime
        # Absorb ALL certified trailing windows (not just the current one) so
        # the new regime owns a real replay/calibration base. A 1-window
        # regime wrecked the global calibration on its first RECALL.
        # Score them with the POST-PROBE encoder so memory can decide whether
        # they BELONG to an existing regime or must open a new one
        # (band-compatible merging — cosine similarity alone collapses affine
        # variants like B into A and contaminates both bands).
        with torch.no_grad():
            probe_out = self.encoder(recent)
            probe_err = self.anomaly_head.window_error(recent, probe_out["recon"])
            z_recent = probe_out["z"]
        regime, sim, created = self.memory.assign_or_create(
            z_recent, self.encoder.get_adapter_state(), x=recent, settle=settle,
            err=probe_err)
        if created:
            self.stats["regimes_created"] += 1

        self._ema_teacher()
        self.memory.update_adapter(regime.regime_id, self.encoder.get_adapter_state())
        self.n_trusted += x.size(0)
        with torch.no_grad():
            out_mem = self.encoder(recent)
            # calibrate on the HELD-OUT portion of the certified windows —
            # the fit split was literally trained on, so its errors are
            # optimistically low and would bias the new band tight
            n_fit = max(1, int(recent.size(0) * 0.6))
            calib = recent[n_fit:] if recent.size(0) - n_fit >= 8 else recent
            recon_calib = out_mem["recon"][-calib.size(0):]
            self.anomaly_head.set_from(calib, recon_calib, regime.regime_id)
            # regime-PURE band: from this regime's windows only, so a shared
            # global buffer never contaminates per-regime scoring later
            self._store_band(regime, windows=calib, recon=recon_calib)
            if not self.anomaly_head._errors:
                # set_from refused (too few windows): fall back to the current
                # global buffer so the score never deadlocks at 0.5
                self._store_band(regime)
            regime.threshold = self.anomaly_head.dynamic_threshold()
            self._refresh_ood(x)
            # re-anchor the drift reference to the newly accepted regime
            z_ref = self.encoder.encode(recent[-64:])
            self.drift.reset(new_reference=z_ref)

    @torch.no_grad()
    def _refresh_ood(self, x: torch.Tensor):
        ref = self.memory.replay_windows(256, self.device)
        if ref is None:
            ref = x
        z_ref = self.encoder.encode(ref)
        protos = self.memory.prototypes()
        if protos is None:
            protos = z_ref.mean(0, keepdim=True)
        self.ood.fit(z_ref, protos.to(z_ref.device))

    @torch.no_grad()
    def _recall(self, s: Dict, regime=None):
        """A known regime returned: restore its adapter instead of relearning.

        `regime` is the VERIFY-confirmed candidate when given; otherwise fall
        back to the cosine-nearest match. Recalling the unverified nearest
        regime is exactly what flip-flopped adapters between aligned regimes.
        """
        if regime is None:
            regime = s["regime"]
        if regime is None:
            return
        if regime.adapter_state:
            self.encoder.set_adapter_state(regime.adapter_state)
            self._ema_teacher()
        # re-anchor drift reference and lock anomaly calibration to the
        # recalled regime's memory — but ONLY if that memory is big enough.
        # Recalibrating from a 1-3 window regime empties the error buffer
        # below the 8-sample minimum, which makes every subsequent score
        # fall back to the uninformative 0.5 default (the "deadlock" mode).
        if regime.windows and len(regime.windows) >= 8:
            mem = torch.stack(regime.windows).to(self.device)
            out_mem = self.encoder(mem)
            self.anomaly_head.set_from(mem, out_mem["recon"], regime.regime_id)
            self._store_band(regime, windows=mem, recon=out_mem["recon"])
            z_mem = self.encoder.encode(mem)
            self.drift.reset(new_reference=z_mem)
        self.n_trusted += 1

    def _refine_young_regime(self, regime, x: torch.Tensor):
        """Re-probe a recently committed regime on the newest windows.

        A regime committed while the stream was still moving (mid drift-ramp)
        stores a band measured on transitional windows. Once the stream
        settles, every window fails that band's verification, so the regime
        would oscillate the score between recalled and unrecalled values
        indefinitely. We re-run the adaptation probe on the newest trailing
        windows and, if the result is strictly better, refresh the regime's
        adapter, windows and band in place. On failure the adapter and
        calibration are restored exactly.
        """
        recent = torch.stack(
            self._recent_windows[-self.cfg.coherent_shift_persist:]).to(self.device)
        if recent.size(0) < self.cfg.coherent_shift_persist:
            return
        # same absolute safety net as verify-before-adapt: a majority of
        # concentrated extreme fault evidence must never enter regime memory.
        recent_exc = self._recent_exceed[-self.cfg.coherent_shift_persist:]
        recent_max_z = self._recent_max_z[-self.cfg.coherent_shift_persist:]
        strong = sum(
            e >= self.cfg.fault_exceed_count and z >= self.cfg.fault_min_max_zscore
            for e, z in zip(recent_exc, recent_max_z))
        if (len(recent_exc) >= self.cfg.coherent_shift_persist
                and 2 * strong >= len(recent_exc)):
            return
        band = self.anomaly_head.band()
        if band is None:
            return
        band_mu, band_sd = band
        saved_adapter = self.encoder.get_adapter_state()
        saved_errors = list(self.anomaly_head._errors)
        saved_ch_sd = (self.anomaly_head._ch_sd.clone()
                       if self.anomaly_head._ch_sd is not None else None)
        saved_lock = self.anomaly_head._lock

        # fit / held-out split — the same discipline as verify-before-adapt:
        # a novel condition can always reduce its own training error, so the
        # improvement must be measured on windows the probe has not seen.
        n_fit = max(1, int(recent.size(0) * 0.6))
        fit_x, hold_x = recent[:n_fit], recent[n_fit:]
        hold = hold_x if hold_x.numel() else fit_x

        with torch.no_grad():
            out0 = self.encoder(hold)
            err_before = float(self.anomaly_head.window_error(
                hold, out0["recon"]).mean().item())

        self._probe_adapt(fit_x)

        with torch.no_grad():
            out_hold = self.encoder(hold)
            err = self.anomaly_head.window_error(hold, out_hold["recon"])
            err_after = float(err.mean().item())
            learned = (err_before > 1e-6
                       and err_after <= (1.0 - self.cfg.adapt_min_improvement) * err_before
                       and err_after <= self.cfg.adapt_max_abs_ratio * band_mu)

        if not learned:
            self.encoder.set_adapter_state(saved_adapter)
            self.anomaly_head._errors = saved_errors
            self.anomaly_head._ch_sd = saved_ch_sd
            self.anomaly_head._lock = saved_lock
            return

        # refresh the regime in place: adapter, windows, band, threshold
        self.memory.update_adapter(regime.regime_id, self.encoder.get_adapter_state())
        self._ema_teacher()
        with torch.no_grad():
            out_all = self.encoder(recent)
            self.anomaly_head.set_from(recent, out_all["recon"], regime.regime_id)
            self._store_band(regime, windows=recent, recon=out_all["recon"])
            regime.threshold = self.anomaly_head.dynamic_threshold()
            z_ref = self.encoder.encode(recent[-64:])
            self.drift.reset(new_reference=z_ref)
        # push the settled windows into the regime's replay memory
        with torch.no_grad():
            self.memory._absorb(regime, out_all["z"], recent)

    # ------------------------------------------------------------------
    # Convenience: run a whole stream
    # ------------------------------------------------------------------
    def run_stream(self, windows: torch.Tensor, labels: Optional[torch.Tensor] = None,
                   log_every: int = 200):
        records = []
        for i in range(windows.size(0)):
            fault = bool(labels[i].item()) if labels is not None else False
            rec = self.step(windows[i], true_fault=fault)
            rec["t"] = i
            records.append(rec)
            if log_every and (i + 1) % log_every == 0:
                print(f"  t={i + 1}/{windows.size(0)} action={rec['action']} "
                      f"drift={rec['drift']:.2f} anomaly={rec['anomaly']:.2f}")
        return records

    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "teacher": self.teacher.state_dict(),
            "anomaly_head": self.anomaly_head.state_dict(),
            "anomaly_errors": self.anomaly_head._errors[-2048:],
            "config": self.cfg.to_dict(),
        }, path)

    def load(self, path: str):
        # Safe loading: refuse to execute arbitrary code from untrusted files.
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        self.teacher.load_state_dict(state["teacher"])
        self.anomaly_head.load_state_dict(state["anomaly_head"])
        self.anomaly_head._errors = list(state.get("anomaly_errors", []))
        self._sync_teacher()
