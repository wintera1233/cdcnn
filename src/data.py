"""Loading the UCI Gas Sensor Array Drift dataset under the source-only protocol.

The raw files are immutable (`CLAUDE.md` section 1). Nothing here writes to
`Dataset/`, clips values, removes outliers, or imputes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.protocol import ProtocolError, TargetAccessLog

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"

N_FEATURES = 128
N_CLASSES = 6
SOURCE_BATCH = 1
TARGET_BATCHES = (2, 3, 4, 5, 6, 7, 8, 9, 10)

# Verified by `wc -l Dataset/batch*.dat` on 2026-09-23.
BATCH_ROWS = {1: 445, 2: 1244, 3: 1586, 4: 161, 5: 197,
              6: 2300, 7: 3613, 8: 294, 9: 470, 10: 3600}

# The dataset's own documentation states the encoding directly: "1: Ethanol;
# 2: Ethylene; 3: Ammonia; 4: Acetaldehyde; 5: Acetone; 6: Toluene". Labels 2 and
# 3 are exchanged relative to that statement on the evidence of Batch 1's
# principal-component structure against the paper's Fig. 4(a); see
# docs/label-mapping.md, which also records why the per-batch count table cannot
# be used as the criterion.
GAS_LABELS = {1: "Ethanol", 2: "Ammonia", 3: "Ethylene",
              4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}


def batch_path(index: int) -> Path:
    if index not in BATCH_ROWS:
        raise ProtocolError(f"batch {index} is not part of this dataset")
    return DATASET_DIR / f"batch{index}.dat"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse the dense LIBSVM layout: `label 1:v ... 128:v`, one row per line."""
    labels: list[int] = []
    rows: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != N_FEATURES + 1:
                raise ProtocolError(
                    f"{path.name} line {number}: expected {N_FEATURES + 1} fields, "
                    f"found {len(fields)}")
            labels.append(int(fields[0]))
            values = np.empty(N_FEATURES, dtype=np.float64)
            for position, field in enumerate(fields[1:], start=1):
                index, _, value = field.partition(":")
                if int(index) != position:
                    raise ProtocolError(
                        f"{path.name} line {number}: feature index {index} out of order "
                        f"at position {position}")
                values[position - 1] = float(value)
            rows.append(values)
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if set(np.unique(y)) - set(GAS_LABELS):
        raise ProtocolError(f"{path.name}: labels outside 1..{N_CLASSES}")
    return x, y


def _load(index: int) -> tuple[np.ndarray, np.ndarray]:
    path = batch_path(index)
    x, y = _parse(path)
    if len(y) != BATCH_ROWS[index]:
        raise ProtocolError(
            f"{path.name}: expected {BATCH_ROWS[index]} rows, found {len(y)}. "
            "The dataset is immutable; this file has changed.")
    return x, y


def load_source() -> tuple[np.ndarray, np.ndarray]:
    """Batch 1. The only data any training, fitting, or tuning step may see."""
    return _load(SOURCE_BATCH)


def load_target(index: int, access_log: TargetAccessLog) -> tuple[np.ndarray, np.ndarray]:
    """Open one target batch, recording the access.

    Requires an access log so that no code path can read a target batch without
    leaving evidence for `src.audit.leakage_audit`.
    """
    if index not in TARGET_BATCHES:
        raise ProtocolError(f"batch {index} is not a target batch")
    if not isinstance(access_log, TargetAccessLog):
        raise ProtocolError("reading a target batch requires a TargetAccessLog")
    access_log.record_access(batch_path(index))
    return _load(index)


def class_counts(y: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def dataset_hashes() -> dict[str, str]:
    """Hash every batch file without parsing it; hashing is not reading features."""
    return {path.name: sha256_file(path)
            for path in sorted(DATASET_DIR.glob("batch*.dat"))}
