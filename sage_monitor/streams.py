"""Chronological continual-learning stream construction.

Do NOT shuffle: the whole point is temporal order. The stream is a scripted
sequence of regimes that exercises every mechanism:

    A (learn) -> gradual drift -> B (adapt) -> sudden drift -> C (adapt)
    -> FAULT (must not learn) -> B again (recall) -> novel condition (OOD)

Returns (windows, labels, meta) where labels mark the injected fault segment
and meta records the ground-truth segment boundaries for evaluation.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch

from .datasets import make_windows


@dataclass
class StreamMeta:
    segments: List[dict] = field(default_factory=list)


def _segment(values, labels, name, kind, start, length, meta):
    meta.segments.append({
        "name": name, "kind": kind,
        "start": start, "end": start + length,
    })


def build_stream(
    base: np.ndarray,
    window: int = 60,
    stride: int = 4,
    seg_len: int = 600,
    fault_len: int = 300,
    seed: int = 0,
):
    """Build the demonstration stream from a clean multivariate series."""
    rng = np.random.default_rng(seed)
    n, c = base.shape
    meta = StreamMeta()

    # ---- segment definitions on the raw timeline -----------------------
    pieces = []

    # A: pristine regime
    a = base[:seg_len]
    pieces.append(("A_normal", "normal", a, np.zeros(len(a))))

    # gradual drift A -> B (slow scale + bias ramp)
    g_len = seg_len // 2
    ramp = np.linspace(0, 1, g_len)[:, None]
    grad = base[seg_len:seg_len + g_len] * (1 + 0.5 * ramp) + 0.3 * ramp
    pieces.append(("gradual_drift", "drift", grad, np.zeros(g_len)))

    # B: new stable regime (scaled/biased version of the generator)
    b_src = base[seg_len + g_len:2 * seg_len + g_len]
    b = b_src * 1.5 + 0.3
    pieces.append(("B_normal", "normal", b, np.zeros(len(b))))

    # sudden drift into C (abrupt mean/variance change)
    c_src = base[2 * seg_len + g_len:3 * seg_len + g_len]
    cseg = c_src * 0.6 - 0.8
    pieces.append(("sudden_drift_C", "drift", cseg[:seg_len // 2],
                   np.zeros(seg_len // 2)))

    # C settles
    pieces.append(("C_normal", "normal", cseg[seg_len // 2:],
                   np.zeros(len(cseg) - seg_len // 2)))

    # FAULT: a realistic, sustained sensor fault on a few channels. It combines
    # (a) a step offset (stuck/drifting sensor) with (b) an unreconstructable
    # jitter burst. The jitter breaks temporal predictability so the encoder
    # cannot absorb it with a smooth reconstruction, and the offset keeps the
    # fault sustained rather than transient. This is what the model must NOT
    # learn as a new normal regime.
    f_src = cseg.copy()[:fault_len]
    if len(f_src) < fault_len:
        f_src = np.tile(f_src, (2, 1))[:fault_len]
    ch = rng.choice(c, size=min(3, c), replace=False)
    f_labels = np.zeros(fault_len)
    cut = fault_len // 4
    f_src = f_src.copy()
    f_src[cut:, ch] += 2.5 * rng.choice([-1.0, 1.0], size=len(ch))       # sustained step
    f_src[cut:, ch] += 1.2 * rng.standard_normal((fault_len - cut, len(ch)))  # jitter burst
    f_labels[cut:] = 1
    pieces.append(("FAULT", "fault", f_src, f_labels))

    # B returns (should trigger RECALL, not relearning)
    b2 = b_src[:seg_len] * 1.5 + 0.3 + 0.02 * rng.standard_normal((seg_len, c))
    pieces.append(("B_return", "recall", b2, np.zeros(len(b2))))

    # novel / OOD condition (unseen frequency structure)
    t = np.arange(seg_len)
    novel = np.stack([
        np.sin(2 * np.pi * (0.07 + 0.01 * i) * t) * (1.2 + 0.3 * i % 2)
        for i in range(c)
    ], axis=1).astype(np.float32)
    pieces.append(("novel_OOD", "ood", novel, np.zeros(seg_len)))

    # ---- assemble the raw timeline -------------------------------------
    values = np.concatenate([p[2] for p in pieces], axis=0).astype(np.float32)
    labels = np.concatenate([p[3] for p in pieces], axis=0).astype(np.int64)

    # standardize on the leading normal regime only
    mu = values[:seg_len].mean(0, keepdims=True)
    sd = values[:seg_len].std(0, keepdims=True) + 1e-6
    values = (values - mu) / sd

    # record boundaries
    pos = 0
    for name, kind, seg, _ in pieces:
        _segment(values, labels, name, kind, pos, len(seg), meta)
        pos += len(seg)

    # windowize (window labels = any anomaly inside the window)
    X, Y = make_windows(values, labels, window=window, stride=stride)

    # map segment boundaries into window indices
    for s in meta.segments:
        s["win_start"] = max(0, (s["start"] - window) // stride)
        s["win_end"] = max(s["win_start"], (s["end"] - window) // stride)

    return X, Y, meta
