"""Dual intelligence heads: drift detection, anomaly scoring, OOD detection.

The core design principle (DDADE-style): distribution change != system fault.
The drift detector estimates whether the *normal operating distribution* has
changed; the anomaly detector estimates whether the system currently looks
faulty; the OOD head separates "unknown condition" from both.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SAGEConfig


class DriftDetector:
    """Distribution drift score in latent space.

    Maintains running statistics of "trusted normal" embeddings and scores
    incoming embeddings against them. Combines a Mahalanobis-style distance
    with a simple Page-Hinkley change detector on the distance stream.
    """

    def __init__(self, cfg: SAGEConfig):
        self.cfg = cfg
        self.mean: Optional[torch.Tensor] = None
        self.var: Optional[torch.Tensor] = None
        self.n_seen = 0
        # Page-Hinkley state
        self.ph_sum = 0.0
        self.ph_min = 0.0
        self.ph_cum_mean = 0.0
        self.ph_n = 0
        self.ph_delta = 0.005
        self.ph_lambda = 8.0
        self.ema_dist = 0.0

    def to(self, device):
        if self.mean is not None:
            self.mean = self.mean.to(device)
            self.var = self.var.to(device)
        return self

    @torch.no_grad()
    def update_reference(self, z: torch.Tensor):
        """Incorporate trusted-normal embeddings into the reference model."""
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        batch_mean = z.mean(0)
        batch_var = z.var(0, unbiased=False)
        if self.mean is None:
            self.mean = batch_mean
            self.var = batch_var + 1e-6
            self.n_seen = z.size(0)
        else:
            n = self.n_seen
            m = z.size(0)
            new_n = n + m
            delta = batch_mean - self.mean
            self.mean = (n * self.mean + m * batch_mean) / new_n
            m_a = self.var * n
            m_b = batch_var * m
            self.var = (m_a + m_b + delta.pow(2) * n * m / new_n) / new_n + 1e-6
            self.n_seen = new_n

    @torch.no_grad()
    def score(self, z: torch.Tensor) -> float:
        """Drift score in roughly [0, 1]; higher = stronger distribution change.

        This is a PURE observation: it does not mutate detector state. Call
        `feed()` to advance the change-detector statistics.
        """
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if self.mean is None:
            return 0.0
        dist = ((z - self.mean).pow(2) / (self.var + 1e-6)).mean().sqrt().item()
        d_score = 1.0 - torch.exp(torch.tensor(-dist / 3.0)).item()
        return float(min(d_score, 1.0))

    @torch.no_grad()
    def feed(self, dist_value: float):
        """Advance the Page-Hinkley change detector with one distance sample."""
        a = self.cfg.drift_ema_alpha
        self.ema_dist = (1 - a) * self.ema_dist + a * dist_value
        self.ph_n += 1
        self.ph_cum_mean += (self.ema_dist - self.ph_cum_mean) / self.ph_n
        self.ph_sum += self.ema_dist - self.ph_cum_mean - self.ph_delta
        self.ph_min = min(self.ph_min, self.ph_sum)

    def ph_score(self) -> float:
        """Normalized Page-Hinkley statistic in [0, 1]."""
        if self.ph_n == 0:
            return 0.0
        return float(min((self.ph_sum - self.ph_min) / self.ph_lambda, 1.0))

    @torch.no_grad()
    def distance(self, z: torch.Tensor) -> float:
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if self.mean is None:
            return 0.0
        return float(((z - self.mean).pow(2) / (self.var + 1e-6))
                     .mean().sqrt().item())

    def reset(self, new_reference: Optional[torch.Tensor] = None):
        """Clear change-detector state after a successful adaptation/recall.

        If `new_reference` is given, the trusted-normal reference is re-fit to
        the newly accepted regime instead of keeping the old mixture.
        """
        self.ph_sum = 0.0
        self.ph_min = 0.0
        self.ph_cum_mean = 0.0
        self.ph_n = 0
        self.ema_dist = 0.0
        if new_reference is not None:
            self.mean = None
            self.var = None
            self.n_seen = 0
            self.update_reference(new_reference)


class AnomalyHead(nn.Module):
    """Calibrated reconstruction-error anomaly score.

    The score is the reconstruction error expressed in units of the
    trusted-normal error distribution: A_t = sigmoid((e_t - mu) / 2 sigma).
    Statistics follow a rolling buffer of trusted errors, so they re-anchor
    quickly when a new regime is accepted and stay stable during faults.
    """

    def __init__(self, cfg: SAGEConfig, buf_size: int = 4096):
        super().__init__()
        self.cfg = cfg
        self.buf_size = buf_size
        self._errors: list = []
        self._lock: Optional[int] = None  # regime whose error band is held
        # per-channel normal residual spread, for the exceedance statistic
        self._ch_sd: Optional[torch.Tensor] = None

    @staticmethod
    def _affine(x, recon):
        """Robust per-window affine alignment of recon onto x.

        The scale is fitted PER CHANNEL and reduced by the MEDIAN across
        channels, not by global least squares. Least squares lets a fault on
        a few same-sign channels drag the global scale up, which inflates the
        aligned residuals on perfectly clean channels and makes a component
        fault look like a widespread novel condition. The median scale is
        immune to a minority of deviant channels while still following a
        coherent regime shift (all channels move together).

        Returns (a, b) with shape (B, 1) such that a*recon + b is the
        coherent (scale + bias) alignment of the reconstruction onto the
        window — the part of x a legitimate regime shift explains.
        """
        B = x.size(0)
        xm = x.mean(dim=(1, 2), keepdim=True)          # (B, 1, 1)
        rm = recon.mean(dim=(1, 2), keepdim=True)
        xc = x - xm
        rc = recon - rm
        num = (xc * rc).sum(dim=1)                     # (B, C)
        den = (rc * rc).sum(dim=1) + 1e-6              # (B, C)
        a_c = num / den                                # per-channel scale
        a = a_c.median(dim=1, keepdim=True).values     # (B, 1) robust scale
        a3 = a.unsqueeze(-1)                           # (B, 1, 1)
        b = (xm - a3 * rm).reshape(B, 1)               # (B, 1)
        return a, b

    @staticmethod
    def _aligned_residual(x, recon):
        """Per-channel residual after removing the global affine component."""
        B = x.size(0)
        xf = x.reshape(B, -1)
        rf = recon.reshape(B, -1)
        a, b = AnomalyHead._affine(x, recon)
        resid = (xf - (a * rf + b)).reshape(B, x.size(1), x.size(2))
        return resid  # (B, T, C)

    @torch.no_grad()
    def channel_zscores(self, x, recon):
        """Return aligned residual z-scores for each channel in each window.

        The per-channel RMS residual is normalized by the trusted-normal
        residual spread. Keeping the full z-score vector lets the controller
        distinguish a moderate stale-decoder shift from a concentrated fault
        without relying on a fixed channel count.
        """
        resid = self._aligned_residual(x, recon)
        rms = resid.pow(2).mean(1).sqrt()          # (B, C)
        if self._ch_sd is None:
            return None
        sd = self._ch_sd.to(rms.device).unsqueeze(0)
        return rms / (sd + 1e-6)

    @torch.no_grad()
    def channel_exceedance(self, x, recon, z_thr: float = 2.5) -> int:
        """Number of channels whose aligned residual exceeds ``z_thr``."""
        z = self.channel_zscores(x, recon)
        if z is None:
            return 0
        return int((z > z_thr).sum().item())

    @torch.no_grad()
    def set_from(self, x, recon, regime_id: int):
        """Make one confirmed regime's errors THE calibration band.

        With per-regime scoring the band must describe exactly one operating
        condition. Replacing (rather than appending) keeps the band regime-
        pure; a full regime memory yields well over the 8-sample minimum, so
        the score can never fall back to the 0.5 default and deadlock.
        """
        err = self.window_error(x, recon).detach().float().view(-1)
        self._errors = err.tolist()[-self.buf_size:]
        self._lock = regime_id
        # calibrate per-channel normal residual spread for the exceedance stat
        resid = self._aligned_residual(x, recon)
        rms = resid.pow(2).mean(1).sqrt()          # (B, C)
        self._ch_sd = rms.mean(0) + 1e-6

    @torch.no_grad()
    def band_from(self, x, recon):
        """Error band (mu, sd) computed from ONLY these windows' errors —
        used to store a regime-specific band that is never contaminated by
        other regimes sharing a global buffer."""
        err = self.window_error(x, recon).detach().float().view(-1)
        if err.numel() < 2:
            return None
        return float(err.mean().item()), float(err.std(unbiased=False).item()) + 1e-6

    def window_error(self, x, recon):
        """Global-affine-invariant reconstruction error.

        A coherent operating-condition shift changes the overall level/scale of
        ALL sensors together. We remove only that GLOBAL affine component by
        least-squares fitting recon onto x per window, then score the leftover
        per-channel residuals. Legitimate drift therefore yields small error,
        while a fault (a few sensors deviating from the global fit) stays large.
        This is the drift/anomaly disentanglement signal.
        """
        resid = self._aligned_residual(x, recon)
        err_ch = resid.pow(2).mean(dim=1)     # (B, C)
        # top channels dominate: faults hit specific sensors hard
        top, _ = err_ch.topk(max(1, err_ch.size(1) // 2), dim=1)
        return top.mean(dim=1)

    def _stats(self):
        if len(self._errors) < 8:
            return None
        e = torch.tensor(self._errors, dtype=torch.float32)
        return float(e.mean().item()), float(e.std(unbiased=False).item()) + 1e-6

    def band(self):
        """Current calibration band (mu, sd) of trusted-normal errors."""
        return self._stats()

    @staticmethod
    def score_errors(err, mu, sd):
        """Normalized anomaly score for raw window errors.

        The score uses an effective spread with a relative floor
        (sd_eff = max(sd, 0.15 * mu)). A pure affine shift of a coherent
        regime keeps the same *relative* reconstruction quality, so a purely
        relative band is the honest comparison; an absolute band measured on
        one regime's very consistent errors (sd << mu) would otherwise saturate
        at 1.0 for every unseen condition and erase the drift/fault
        disentanglement the aligned error was designed to provide.
        """
        sd_eff = max(sd, 0.15 * mu)
        # 0.5 sits at the trusted-normal upper bound (mean + 2 sigma_eff), so
        # legitimate drift stays mid-range while faults saturate near 1.
        return torch.sigmoid((err - (mu + 2.0 * sd_eff)) / sd_eff)

    def forward(self, z, x, recon, band=None):
        """Anomaly score. If `band` (mu, sd) is given, the window is scored
        against that regime's own error band instead of the global one."""
        err = self.window_error(x, recon)
        if band is not None:
            mu, sd = band
        else:
            stats = self._stats()
            if stats is None:
                return torch.full((err.size(0),), 0.5, device=err.device)
            mu, sd = stats
        return self.score_errors(err, mu, sd)

    def score_window(self, x, recon, band=None):
        """Score windows without the (unused) latent, against an explicit band."""
        err = self.window_error(x, recon)
        if band is None:
            band = self._stats()
        if band is None:
            return torch.full((err.size(0),), 0.5, device=err.device)
        return self.score_errors(err, band[0], band[1])

    @torch.no_grad()
    def raw_error(self, x, recon):
        return self.window_error(x, recon).mean().item()

    def dynamic_threshold(self, k_sigma: float = 3.0) -> float:
        """Adaptive threshold: mean + k * std of trusted normal errors."""
        stats = self._stats()
        if stats is None:
            return 0.5
        mu, sd = stats
        return float(mu + k_sigma * sd)


class OODHead:
    """Prototype-distance + density based OOD score.

    P(unknown) rises when the embedding is far from every regime prototype
    and sits in a low-density region of the memory. The raw combined distance
    is normalised against the distribution of distances observed on trusted
    normal data, so the score is calibrated (near 0 for known regimes, near 1
    for genuinely unseen conditions) rather than saturating.
    """

    def __init__(self, cfg: SAGEConfig):
        self.cfg = cfg
        self.prototypes: Optional[torch.Tensor] = None  # (R, d)
        self.cov_inv: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self._ref_dist_mu = None   # mean raw distance on trusted normals
        self._ref_dist_sd = None   # std  raw distance on trusted normals

    def to(self, device):
        if self.prototypes is not None:
            self.prototypes = self.prototypes.to(device)
        if self.cov_inv is not None:
            self.cov_inv = self.cov_inv.to(device)
            self.mean = self.mean.to(device)
        return self

    def _raw(self, z: torch.Tensor) -> torch.Tensor:
        """Combined (prototype + Mahalanobis) distance, per sample."""
        d_proto = torch.cdist(z, self.prototypes).min(dim=-1).values
        diff = z - self.mean
        d_maha = (diff @ self.cov_inv * diff).sum(-1).clamp(min=0).sqrt()
        return d_proto + 0.5 * d_maha

    @torch.no_grad()
    def fit(self, z_all: torch.Tensor, prototypes: torch.Tensor):
        z_all = z_all.detach().float()
        self.prototypes = prototypes.detach().float()
        self.mean = z_all.mean(0)
        centered = z_all - self.mean
        cov = centered.T @ centered / max(z_all.size(0) - 1, 1)
        cov += torch.eye(cov.size(0), device=cov.device) * self.cfg.ood_mahalanobis_reg
        self.cov_inv = torch.linalg.pinv(cov)
        # calibrate the distance scale on the trusted reference cloud
        raw = self._raw(z_all)
        self._ref_dist_mu = float(raw.mean().item())
        self._ref_dist_sd = float(raw.std(unbiased=False).item()) + 1e-6

    @torch.no_grad()
    def score(self, z: torch.Tensor) -> float:
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if self.prototypes is None or self.cov_inv is None:
            return 0.5  # uninformative before fit
        raw = self._raw(z).mean()
        if self._ref_dist_mu is None:
            return 0.5
        # distance in units of the trusted-normal spread
        units = (raw - self._ref_dist_mu) / (2.0 * self._ref_dist_sd)
        score = torch.sigmoid(units)
        return float(score.item())
