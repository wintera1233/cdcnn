"""The paper's `Normal` block.

Fig. 2 places a block named "Normal" before data augmentation and never defines
it anywhere in the paper or the supplement, so the choice is ours. Two readings
are implemented:

`standard_scaler`
    Per-feature standardisation with the mean and standard deviation **fitted on
    Batch 1 only**, as `CLAUDE.md` section 2 requires. Parameters travel with the
    checkpoint and are applied unchanged to every target batch.

`per_sample`
    Each measurement is standardised against its own 128 values. Parameter-free,
    so nothing is fitted on anything and a drifted target sample is normalised by
    its own statistics rather than by Batch 1's. The previous project measured
    this as worth +0.109 target mean.
"""

from __future__ import annotations

import numpy as np

from src.protocol import ProtocolError

EPSILON = 1e-8
NORMALIZERS = ("standard_scaler", "per_sample")


def fit(kind: str, x: np.ndarray) -> dict:
    """Fit on the source batch. `per_sample` has nothing to fit."""
    if kind not in NORMALIZERS:
        raise ProtocolError(f"unknown normalizer {kind!r}")
    if kind == "per_sample":
        return {"kind": kind}
    if x.ndim != 2:
        raise ProtocolError(f"expected a 2-D array, got shape {x.shape}")
    mean = x.mean(axis=0)
    # ddof=0 matches sklearn's StandardScaler, which the previous project used.
    std = x.std(axis=0)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ProtocolError("non-finite statistics while fitting the Normal block")
    return {"kind": kind, "mean": mean.tolist(), "std": std.tolist()}


def apply(params: dict, x: np.ndarray) -> np.ndarray:
    kind = params["kind"]
    if kind == "per_sample":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        return (x - mean) / np.maximum(std, EPSILON)
    if kind == "standard_scaler":
        mean = np.asarray(params["mean"], dtype=np.float64)
        std = np.asarray(params["std"], dtype=np.float64)
        if mean.shape[0] != x.shape[1]:
            raise ProtocolError(
                f"normalizer fitted on {mean.shape[0]} features, got {x.shape[1]}")
        return (x - mean) / np.maximum(std, EPSILON)
    raise ProtocolError(f"unknown normalizer {kind!r}")
