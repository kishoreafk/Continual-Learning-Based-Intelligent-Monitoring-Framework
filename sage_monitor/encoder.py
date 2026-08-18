"""Temporal encoder: TCN backbone + lightweight attention.

The encoder maps a multivariate telemetry window to a latent health state.
Parameter-efficient adaptation is realised with low-rank (LoRA-style)
adapters that are the ONLY parameters updated during continual adaptation;
the frozen base keeps previously learned structure.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.c1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.c2 = CausalConv1d(channels, channels, kernel_size, dilation)
        # GroupNorm, not BatchNorm: BN running statistics mutate in train()
        # mode and differ from eval() scoring, which silently corrupts the
        # frozen backbone during continual-adaptation probes. GroupNorm is
        # mode-independent by construction.
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.drop(torch.relu(self.norm1(self.c1(x))))
        out = self.drop(torch.relu(self.norm2(self.c2(out))))
        return F.relu(x + out)


class LowRankAdapter(nn.Module):
    """LoRA-style adapter: frozen during base use, enabled per regime."""

    def __init__(self, dim, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


class TemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.proj = nn.Linear(cfg.n_channels, d)
        self.tcn = nn.Sequential(
            *[TCNBlock(d, cfg.kernel_size, 2 ** i, cfg.dropout)
              for i in range(cfg.n_encoder_layers)]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=2 * d,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(
            layer, num_layers=cfg.n_attention_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        # parameter-efficient continual adaptation module
        self.adapter = LowRankAdapter(d, cfg.adapter_rank)
        self.adapter_enabled = True

        # decoder: reconstructs per-timestep FROM THE ENCODER'S TEMPORAL
        # FEATURES, not from the pooled latent. Two structural problems are
        # fixed by this:
        #   1. A constant-in-time reconstruction can only fit each window's
        #      mean level, so the "reconstruction error" is the irreducible
        #      temporal variance — amplitude-dependent, which makes a
        #      legitimate regime shift look like a fault forever.
        #   2. A decoder fed only the pooled latent (even with a TCN) receives
        #      a constant-in-time input and can only emit constant output.
        # Decoding from the per-step features makes this a genuine
        # autoencoder, and because the decoder is an ADAPTIVE module, a newly
        # accepted regime can be reconstructed well after a few probe steps —
        # its error falls back into band and it reads as normal afterwards.
        self.dec_tcn = nn.Sequential(
            TCNBlock(d, cfg.kernel_size, 1, cfg.dropout),
            TCNBlock(d, cfg.kernel_size, 2, cfg.dropout),
        )
        self.dec_norm = nn.LayerNorm(d)
        self.recon_head = nn.Linear(d, cfg.n_channels)
        self.forecast_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, cfg.horizon * cfg.n_channels),
        )

    # parameter-efficient continual adaptation: low-rank adapter + decoder +
    # output heads. The temporal backbone stays frozen, preserving structure.
    ADAPTIVE_MODULES = ("adapter", "dec_tcn", "dec_norm",
                        "recon_head", "forecast_head")

    def temporal_features(self, x):
        """x: (B, T, C) -> h: (B, T, d) per-step features."""
        h = self.proj(x)
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h = self.attn(h)
        return h

    def encode(self, x):
        """x: (B, T, C) -> z: (B, d)."""
        h = self.temporal_features(x)
        z = h.mean(dim=1)                     # global pooling over time
        z = self.norm(z)
        if self.adapter_enabled:
            z = self.adapter(z)
        return z

    def decode(self, h):
        """h: (B, T, d) per-step features -> recon: (B, T, C)."""
        h = self.dec_tcn(h.transpose(1, 2)).transpose(1, 2)
        h = self.dec_norm(h)
        return self.recon_head(h)

    def forecast(self, z):
        fc = self.forecast_head(z)
        return fc.view(-1, self.cfg.horizon, self.cfg.n_channels)

    def forward(self, x):
        h = self.temporal_features(x)
        z = self.norm(h.mean(dim=1))
        if self.adapter_enabled:
            z = self.adapter(z)
        return {
            "z": z,
            "recon": self.decode(h),
            "forecast": self.forecast(z),
        }

    # ---- parameter-efficient adaptation API -----------------------------
    def base_parameters(self):
        adaptive_ids = {id(p) for p in self.adaptive_parameters()}
        for p in self.parameters():
            if id(p) not in adaptive_ids:
                yield p

    def adaptive_parameters(self):
        for name in self.ADAPTIVE_MODULES:
            yield from getattr(self, name).parameters()

    # backwards-compatible alias
    adapter_parameters = adaptive_parameters

    def set_adapter_state(self, state_dict):
        """Load a saved regime's adaptive parameters (adapter + heads)."""
        for name in self.ADAPTIVE_MODULES:
            module = getattr(self, name)
            prefix = name + "."
            sub = {k[len(prefix):]: v for k, v in state_dict.items()
                   if k.startswith(prefix)}
            if sub:
                module.load_state_dict(sub)

    def get_adapter_state(self):
        state = {}
        for name in self.ADAPTIVE_MODULES:
            for k, v in getattr(self, name).state_dict().items():
                state[f"{name}.{k}"] = v.detach().clone()
        return state
