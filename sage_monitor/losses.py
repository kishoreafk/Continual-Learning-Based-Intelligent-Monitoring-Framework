"""Multi-objective training losses.

    L = l1*L_recon + l2*L_forecast + l3*L_contrastive
        + l4*L_consistency + l5*L_replay
"""

import torch
import torch.nn.functional as F


def reconstruction_loss(recon, x):
    return F.mse_loss(recon, x)


def forecast_loss(forecast, target):
    """target: (B, H, C) future telemetry."""
    return F.mse_loss(forecast, target)


def contrastive_loss(z, regime_ids):
    """SupCon-style loss: same-regime latents pull together, others push apart.

    Falls back to 0 when a batch has fewer than 2 regimes.
    """
    labels = torch.tensor(regime_ids, device=z.device)
    uniq = labels.unique()
    if uniq.numel() < 2:
        return z.new_zeros(())
    z = F.normalize(z, dim=1)
    sim = z @ z.T / 0.1
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos.fill_diagonal_(0)
    mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim.masked_select(mask).view(sim.size(0), -1)
    pos = pos.masked_select(mask).view(sim.size(0), -1)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = pos.sum(1).clamp(min=1)
    return -(pos * log_prob).sum(1).div(denom).mean()


def consistency_loss(student_z, teacher_z):
    """Latent-space consistency: keep the adapting student anchored to the
    teacher's representation (heads may diverge to fit the new regime)."""
    return F.mse_loss(student_z, teacher_z)


def replay_loss(recon, x):
    return F.mse_loss(recon, x)
