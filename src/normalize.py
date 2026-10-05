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

from src import project as projection
from src.protocol import ProtocolError

EPSILON = 1e-8

# The 128 features are sensor-major: positions 8*(s-1)+k hold statistic k of
# sensor s, for 16 sensors and 8 statistics. Two of the eight are steady-state
# resistances of order 1e4; the other six are transients of order 1.
N_SENSORS = 16
N_STATISTICS = 8

NORMALIZERS = ("standard_scaler", "per_sample", "standard_then_per_sample",
               "signed_log_then_per_sample", "per_statistic_group",
               "logps_proj_offset_axis", "logps_proj_sub3", "logps_proj_eth")

# v12: signed-log -> per-sample, then the drift subspace of Batch 1 projected
# out. The basis is fitted on the rows handed to `fit` and travels with the
# checkpoint, so `src.evaluate` applies the same projection to every target
# sample. See `src/project.py`.
PROJECTED = {"logps_proj_offset_axis": "offset_axis",
             "logps_proj_sub3": "sub3",
             "logps_proj_eth": "eth"}


def _per_sample(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.maximum(std, EPSILON)


def _signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _per_statistic_group(x: np.ndarray) -> np.ndarray:
    """Standardise each statistic across the 16 sensors, within each sample.

    `per_sample` pools all 128 values, so the two order-1e4 steady-state features
    dominate the mean and standard deviation and compress the six order-1
    transients. Grouping by statistic keeps each on its own scale. The layout is
    documented, not inferred, and is never a 2-D image; see CLAUDE.md section 5.
    """
    grouped = x.reshape(-1, N_SENSORS, N_STATISTICS)
    mean = grouped.mean(axis=1, keepdims=True)
    std = grouped.std(axis=1, keepdims=True)
    return ((grouped - mean) / np.maximum(std, EPSILON)).reshape(x.shape)


def fit(kind: str, x: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """Fit the Normal block on `x`, which must be Batch 1 rows and nothing else.

    `rows` gives each sample's position in `batch1.dat`; the projected kinds
    need it to find the acquisition sessions. It defaults to `arange(len(x))`,
    which is right when `x` is the whole source in file order.
    """
    if kind in PROJECTED:
        if rows is None:
            rows = np.arange(len(x))
        z = _per_sample(_signed_log(x))
        basis = projection.drift_basis(z, rows, PROJECTED[kind])
        return {"kind": kind, "basis": basis.tolist()}
    """Fit on the source batch. Only the `standard_scaler` stage has parameters."""
    if kind not in NORMALIZERS:
        raise ProtocolError(f"unknown normalizer {kind!r}")
    if kind in ("per_sample", "signed_log_then_per_sample", "per_statistic_group"):
        return {"kind": kind}
    if kind == "standard_then_per_sample":
        inner = fit("standard_scaler", x)
        return {"kind": kind, "mean": inner["mean"], "std": inner["std"]}
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
        return _per_sample(x)
    if kind == "signed_log_then_per_sample":
        return _per_sample(_signed_log(x))
    if kind in PROJECTED:
        return projection.project(_per_sample(_signed_log(x)), params["basis"])
    if kind == "per_statistic_group":
        return _per_statistic_group(x)
    if kind == "standard_then_per_sample":
        return _per_sample(apply({**params, "kind": "standard_scaler"}, x))
    if kind == "standard_scaler":
        mean = np.asarray(params["mean"], dtype=np.float64)
        std = np.asarray(params["std"], dtype=np.float64)
        if mean.shape[0] != x.shape[1]:
            raise ProtocolError(
                f"normalizer fitted on {mean.shape[0]} features, got {x.shape[1]}")
        return (x - mean) / np.maximum(std, EPSILON)
    raise ProtocolError(f"unknown normalizer {kind!r}")
