"""Validation for persisted source-only cross-validation folds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_folds(
    path: str | Path,
    features: np.ndarray,
    lines: np.ndarray,
    labels: np.ndarray,
    source: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load folds and verify sample identity, coverage, and duplicate isolation."""
    frame = pd.read_csv(path)
    ids = [f"batch{source}:line{value}" for value in lines]
    if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(ids):
        raise ValueError("Saved fold identities do not exactly match source samples")
    frame = frame.set_index("sample_id").loc[ids].reset_index()
    if not np.array_equal(frame.line_number, lines) or not np.array_equal(
        frame.gas_label, labels
    ):
        raise ValueError("Saved fold provenance does not match source data")
    fold_ids = frame.cv_fold.to_numpy(int)
    if not np.array_equal(np.unique(fold_ids), np.arange(1, 6)):
        raise ValueError("Expected saved folds 1-5")
    _, duplicate_groups = np.unique(features, axis=0, return_inverse=True)
    crossing_groups = [
        int(group)
        for group in np.unique(duplicate_groups)
        if np.unique(fold_ids[duplicate_groups == group]).size > 1
    ]
    if crossing_groups:
        raise ValueError(f"Exact-duplicate groups cross folds: {crossing_groups}")
    return frame, fold_ids
