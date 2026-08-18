"""SAGE-Monitor: Safety-gated Adaptive Continual Monitor.

A streaming autonomous health-monitoring framework that separates
environmental/operational drift from genuine faults, gates learning behind a
safety check, performs parameter-efficient continual adaptation on trusted
normal data, and uses regime-aware memory to recall previously learned
operating conditions.
"""

__version__ = "0.1.0"

from .config import SAGEConfig
from .encoder import TemporalEncoder
from .controller import SAGEController

__all__ = ["SAGEConfig", "TemporalEncoder", "SAGEController"]
