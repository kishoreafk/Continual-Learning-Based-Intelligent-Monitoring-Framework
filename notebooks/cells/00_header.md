# SAGE-Monitor — Safety-Gated Continual Monitoring (Colab training)

**Development of a Continual-Learning-Based Intelligent Monitoring Framework for Systems Operating Under Dynamically Changing Conditions**

This notebook trains and evaluates the framework in the cloud:

1. **Setup** — clones the public `sage_monitor` package from GitHub and imports it.
2. **Data** — downloads NASA **SMAP/MSL** telemetry from Kaggle (real spacecraft/rover data).
3. **Stream** — builds a chronological continual-learning stream: normal regime → drift → new regime → **fault** → regime return → novel/OOD.
4. **Train** — base model on the clean regime, then the safety-gated streaming loop.
5. **Evaluate** — detection, temporal, continual-learning and **safety** metrics (including `UnsafeAdaptationRate`).

> Runtime → change runtime type → **GPU** for best speed (CPU also works). The setup cell needs internet access to clone the GitHub repository.
