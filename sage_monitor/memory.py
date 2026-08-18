"""Regime-aware memory: prototype embeddings, adapter states, replay buffers.

Instead of one giant replay buffer, history is partitioned into regimes.
When a previously seen operating condition returns, its prototype is matched
and its adapter + statistics are recalled instead of relearning from scratch.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class Regime:
    regime_id: int
    prototype: torch.Tensor                 # (d,)
    adapter_state: Dict[str, torch.Tensor] = field(default_factory=dict)
    embeddings: List[torch.Tensor] = field(default_factory=list)  # latent replay
    windows: List[torch.Tensor] = field(default_factory=list)     # raw replay
    n_seen: int = 0
    threshold: float = 0.5
    err_mu: Optional[float] = None          # calibrated error band of this regime
    err_sd: Optional[float] = None          # (used by verify-before-recall)


class RegimeMemory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.regimes: List[Regime] = []
        self._next_id = 0
        self._pending_id: Optional[int] = None  # regime absorbing drift transition

    def __len__(self):
        return len(self.regimes)

    def _get(self, rid):
        for r in self.regimes:
            if r.regime_id == rid:
                return r
        return None

    def finalize_pending(self):
        """Close the pending transition regime once the stream settles."""
        self._pending_id = None

    def prototypes(self) -> Optional[torch.Tensor]:
        if not self.regimes:
            return None
        return torch.stack([r.prototype for r in self.regimes])

    def nearest(self, z: torch.Tensor):
        """Return (regime, similarity) of the closest regime, or (None, 0)."""
        if not self.regimes:
            return None, 0.0
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        protos = self.prototypes().to(z.device)
        # cosine similarity
        zs = F.normalize(z, dim=1)
        ps = F.normalize(protos, dim=1)
        sims = zs @ ps.T                      # (B, R)
        best = sims.mean(0).argmax().item()
        return self.regimes[best], float(sims.mean(0).max().item())

    def _similarity(self, z: torch.Tensor, proto: torch.Tensor) -> float:
        """Cosine similarity between (a batch of) latents and one prototype."""
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        zs = F.normalize(z, dim=1)                              # (B, d)
        p = F.normalize(proto.detach().float().unsqueeze(0), dim=1)  # (1, d)
        return float((zs @ p.T).mean().item())                  # mean over batch

    def _band_compatible(self, regime: Regime, err) -> bool:
        """True if the new observation's reconstruction error sits NEAR this
        regime's stored calibration band — not merely below its top.

        Cosine similarity in latent space is scale-invariant, so an affine
        variant of a known regime (regime B = A at higher amplitude) scores
        ~0.93 against A yet reconstructs at a very different error level.
        Merging them on similarity alone contaminates both bands and leaves the
        larger regime saturated at anomaly ~ 1.0 forever. Requiring the error
        to lie near the band centre keeps genuinely distinct operating
        conditions in separate slots even when their latents align.
        """
        if regime.err_mu is None or regime.err_sd is None or err is None:
            return True  # no band calibrated yet -> cannot disambiguate
        e = err.detach().float().view(-1)
        if e.numel() == 0:
            return True
        sd_eff = max(regime.err_sd, 0.15 * regime.err_mu)
        e_mean = float(e.mean().item())
        return abs(e_mean - regime.err_mu) <= 2.0 * sd_eff

    def assign_or_create(self, z: torch.Tensor, adapter_state: Dict[str, torch.Tensor],
                         x: Optional[torch.Tensor] = None, settle: bool = False,
                         err: Optional[torch.Tensor] = None):
        """Attach z to the nearest BAND-COMPATIBLE regime, else open a new one.

        If a pending (transition) regime is open, new observations are absorbed
        there rather than opening yet another slot. `settle=True` closes it.
        `err` (the window's post-adaptation reconstruction error) gates merging
        so distinct regimes whose latents happen to align are never collapsed
        into one.
        """
        if settle:
            self.finalize_pending()
        pending = self._get(self._pending_id) if self._pending_id is not None else None

        regime, sim = self.nearest(z)
        if pending is not None and (regime is None or sim < self.cfg.regime_assign_threshold + 0.15):
            self._absorb(pending, z, x)
            return pending, 1.0, False

        # Pick the most similar regime whose calibration band actually explains
        # this observation. This is what stops regime B from merging into A.
        best, best_sim = None, -1.0
        for r in self.regimes:
            sim_r = self._similarity(z, r.prototype)
            if (sim_r >= self.cfg.regime_assign_threshold
                    and self._band_compatible(r, err)):
                if sim_r > best_sim:
                    best, best_sim = r, sim_r
        if best is not None:
            self._absorb(best, z, x)
            return best, best_sim, False

        # create a new regime (evict least-used if full)
        if len(self.regimes) >= self.cfg.n_regimes:
            self.regimes.sort(key=lambda r: r.n_seen)
            self.regimes.pop(0)
        new_regime = Regime(regime_id=self._next_id,
                            prototype=z.detach().float().mean(0).clone())
        self._next_id += 1
        self.regimes.append(new_regime)
        self._absorb(new_regime, z, x)
        new_regime.adapter_state = {k: v.detach().clone()
                                    for k, v in adapter_state.items()}
        self._pending_id = new_regime.regime_id
        return new_regime, 1.0, True

    def _absorb(self, regime: Regime, z: torch.Tensor, x: Optional[torch.Tensor]):
        z = z.detach().float()
        if z.dim() == 1:
            z = z.unsqueeze(0)
        regime.n_seen += z.size(0)
        # running prototype
        k = z.size(0)
        n = regime.n_seen
        regime.prototype = (regime.prototype * (n - k) + z.mean(0) * k) / n
        # bounded replay memory
        regime.embeddings.extend(z[i].clone() for i in range(k))
        cap = self.cfg.memory_per_regime
        if len(regime.embeddings) > cap:
            regime.embeddings = regime.embeddings[-cap:]
        if x is not None:
            x = x.detach().float()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            regime.windows.extend(x[i].clone() for i in range(x.size(0)))
            if len(regime.windows) > cap:
                regime.windows = regime.windows[-cap:]

    def replay_batch(self, batch: int, device) -> Optional[torch.Tensor]:
        """Sample embeddings across regimes for the replay loss."""
        all_emb = []
        for r in self.regimes:
            all_emb.extend(r.embeddings)
        if not all_emb:
            return None
        all_emb = torch.stack(all_emb).to(device)
        idx = torch.randperm(all_emb.size(0))[:batch]
        return all_emb[idx]

    def replay_windows(self, batch: int, device) -> Optional[torch.Tensor]:
        """Sample raw windows across regimes (for the replay reconstruction loss)."""
        all_win = []
        for r in self.regimes:
            all_win.extend(r.windows)
        if not all_win:
            return None
        all_win = torch.stack(all_win).to(device)
        idx = torch.randperm(all_win.size(0))[:batch]
        return all_win[idx]

    def update_adapter(self, regime_id: int, adapter_state: Dict[str, torch.Tensor]):
        for r in self.regimes:
            if r.regime_id == regime_id:
                r.adapter_state = {k: v.detach().clone()
                                   for k, v in adapter_state.items()}
