"""Configuration for the SAGE-Monitor framework."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict


@dataclass
class SAGEConfig:
    """All hyper-parameters of the framework in one place."""

    # --- data geometry ---
    n_channels: int = 25          # number of telemetry channels after selection
    window: int = 60              # samples per observation window
    stride: int = 1               # stride between consecutive windows
    horizon: int = 10             # forecast horizon for the forecast loss

    # --- encoder ---
    d_model: int = 64             # latent width
    n_encoder_layers: int = 3     # TCN blocks
    kernel_size: int = 7
    n_heads: int = 4              # attention heads
    n_attention_layers: int = 1
    dropout: float = 0.1

    # --- losses (multi-objective) ---
    lambda_recon: float = 1.0
    lambda_forecast: float = 0.5
    lambda_contrastive: float = 0.1
    lambda_consistency: float = 0.5
    lambda_replay: float = 0.5

    # --- drift detector ---
    drift_ref_window: int = 500   # reference-window length for distribution stats
    drift_ema_alpha: float = 0.05
    drift_unlock_persist: int = 20    # consecutive drifted windows to unlock ADAPT
    drift_unlock_error_ratio: float = 1.5  # unlock only if raw error stays near band

    # --- OOD head ---
    ood_mahalanobis_reg: float = 1e-4   # covariance ridge
    ood_energy_temp: float = 1.0

    # --- safety gate ---
    gate_threshold: float = 0.5   # tau_G
    gate_w_confidence: float = 0.30
    gate_w_normality: float = 0.30
    gate_w_drift: float = 0.20
    gate_w_agreement: float = 0.20
    gate_min_obs: int = 64        # warm-up before gating decisions
    # --- residual-shape fault discrimination (channel-count robust) ---
    exceed_z_thr: float = 2.5     # z-score at which a channel counts as deviant
    fault_exceed_count: int = 3   # retain a multi-channel guard for faults
    novel_min_max_zscore: float = 8.0   # B stays below this; novel/OOD rises above
    fault_min_max_zscore: float = 10.5  # component faults have concentrated extremes
    ood_exceed_frac: float = 0.80  # novel/OOD must affect most channels, not merely
                                   # exceed a fixed count that changes with geometry
    alert_anomaly: float = 0.85   # anomaly level that escalates a fault to ALERT
    ood_unknown: float = 0.90     # OOD level reported as "unknown condition"
    # persistence of low-exceedance (coherent) evidence certifying a new regime.
    # Must exceed the fixed residue tail left by any fault inside the window
    # (window//stride samples), so fault residue can never certify an ADAPT.
    coherent_shift_persist: int = 16
    fault_confirm_streak: int = 3    # consecutive fault-band windows confirming
                                     # a fault (kills boundary transients)
    # --- verify-before-alarm: try the nearest known regime before ALERTing ---
    trial_windows: int = 8       # windows a trial recall gets to prove itself
    trial_cooldown: int = 25     # windows after a failed trial before retrying
    trial_max_exceed: int = 3    # exceedance >= this -> clear multi-channel
                                 # fault, skip the trial (a returning regime is
                                 # affine-explainable, so it never exceeds here)

    # --- adaptation ---
    adapter_rank: int = 8
    adapt_lr: float = 3e-3
    adapt_batch: int = 32
    # probe depth. 3 gradient steps leave a just-committed regime under-fit:
    # its band is measured on half-fitted reconstructions, so the settled
    # stream keeps failing verification afterwards and the score oscillates.
    # ~8 steps fit the new regime properly while the hold-out scoring and the
    # replay/consistency anchors still prevent overfitting and forgetting.
    adapt_epochs: int = 8
    # verify-before-adapt also commits when the probe substantially REDUCES
    # the held-out error, even if it is still above the previous regime's
    # (stale) absolute band — a larger-amplitude coherent regime can never
    # fit a tighter regime's absolute band, yet is fully learnable.
    adapt_min_improvement: float = 0.15
    adapt_max_abs_ratio: float = 4.0  # err_after must stay below this x band mean
    adapt_fail_cooldown: int = 12     # windows before re-probing a failed ADAPT
    ema_decay: float = 0.995      # teacher update rate
    consistency_eps: float = 0.25 # teacher-student agreement margin

    # --- regime memory ---
    n_regimes: int = 12           # max regime slots
    memory_per_regime: int = 64   # replay windows kept per regime
    regime_assign_threshold: float = 0.30  # similarity to open/absorb a regime
    regime_recall_similarity: float = 0.62 # similarity required to RECALL a regime
    # young-regime refinement: a regime committed mid-transition carries a
    # band measured on transitional windows and keeps failing verification on
    # the settled stream. While n_seen <= refine_max_seen, a sustained run of
    # verify failures re-probes on the newest windows and refines the regime
    # in place instead of spawning duplicate slots.
    regime_refine_max_seen: int = 40
    refine_persist: int = 4       # consecutive verify failures -> refine probe

    # --- training ---
    train_lr: float = 1e-3
    train_epochs: int = 10
    batch_size: int = 64
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "auto"          # "auto" | "cpu" | "cuda"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SAGEConfig":
        valid = {f for f in SAGEConfig.__dataclass_fields__}  # type: ignore[attr-defined]
        return SAGEConfig(**{k: v for k, v in d.items() if k in valid})
