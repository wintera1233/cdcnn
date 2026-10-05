"""v12: the drift subspace of Batch 1, and the projection that removes it.

Three of the six gases were acquired twice in Batch 1 (`src.augment.SOURCE_BLOCKS`).
Each gas's second-session centroid minus its first-session centroid is one
offset vector in the normalised input space. The three candidates differ only
in how the three vectors are used:

    offset_axis   the single line the three unit offsets share: first right
                  singular vector of the 3 x 128 stack, uncentred. Source-only,
                  no gas chosen, indifferent to the sign of each offset.
    sub3          the span of all three offsets. Source-only, no choice at all.
    eth           Ethanol's offset alone. That it aligns best with the real
                  drift was learned against target data, so a run using it is
                  target-informed and must say so.

The projection `x' = x - U U^T x` is a fixed linear map with no learnable part,
applied to Batch 1 in training and to every later sample at inference through
the normaliser stored in the checkpoint. It is not a reshape; the input stays a
flat 128-vector (CLAUDE.md section 5).

`rows` are the positions in `batch1.dat` of the rows handed in, so the basis
can be fitted on a subset - a cross-validation fold - using only that subset's
members of each acquisition session. See `docs/v12-step0-separability.md`.
"""

from __future__ import annotations

import numpy as np

from src.augment import SOURCE_BLOCKS
from src.protocol import ProtocolError

SUBSPACES = ("offset_axis", "sub3", "eth")
MIN_SESSION_ROWS = 2


def session_offsets(z: np.ndarray, rows: np.ndarray) -> dict[int, np.ndarray]:
    """Second-session centroid minus first-session centroid, per class with two.

    `z[i]` is row `rows[i]` of Batch 1 after the Normal block. Only rows present
    in `z` take part, so a fold's basis never sees its held-out rows.
    """
    rows = np.asarray(rows)
    if rows.shape != (len(z),):
        raise ProtocolError("rows must give one Batch 1 position per sample")
    offsets = {}
    for label, ((a, b), (c, d)) in SOURCE_BLOCKS.items():
        first = (rows >= a) & (rows < b)
        second = (rows >= c) & (rows < d)
        if first.sum() < MIN_SESSION_ROWS or second.sum() < MIN_SESSION_ROWS:
            raise ProtocolError(
                f"label {label}: too few rows of a session present to fit an offset")
        offsets[label] = z[second].mean(axis=0) - z[first].mean(axis=0)
    return offsets


def drift_basis(z: np.ndarray, rows: np.ndarray, subspace: str) -> np.ndarray:
    """An orthonormal basis `[128, k]` of the chosen drift subspace."""
    if subspace not in SUBSPACES:
        raise ProtocolError(f"unknown drift subspace {subspace!r}")
    offsets = session_offsets(z, rows)
    unit = {label: v / np.linalg.norm(v) for label, v in offsets.items()}
    if subspace == "eth":
        return unit[1][:, None]
    stacked = np.stack([unit[label] for label in sorted(unit)])
    if subspace == "offset_axis":
        _, _, vh = np.linalg.svd(stacked, full_matrices=False)
        return vh[0][:, None]
    basis, _ = np.linalg.qr(stacked.T)
    return basis


def project(z: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """`z - U U^T z`, row by row."""
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != z.shape[1]:
        raise ProtocolError(
            f"basis is {basis.shape}, expected ({z.shape[1]}, k)")
    return z - (z @ basis) @ basis.T
