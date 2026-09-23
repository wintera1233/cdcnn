"""How much of the real drift does a trained backbone remove?

Post-hoc analysis of a completed run. Every checkpoint named here is already
frozen and its target batches already opened and audited, so this opens nothing
the run did not; it must never be used to choose anything.

The statistic is a ratio of ratios. For each target batch and each class present
in both it and Batch 1, the class centroid's displacement from Batch 1's own
centroid is divided by Batch 1's within-class radius, giving a drift in units of
the source spread. Doing that at the input and again at `z_f` and comparing the
two says what fraction of the drift the five blocks removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from src import normalize
from src.data import TARGET_BATCHES, load_source, load_target
from src.evaluate import load_checkpoint
from src.model import to_input
from src.protocol import TargetAccessLog


def drift_ratio(source: np.ndarray, source_y: np.ndarray,
                target: np.ndarray, target_y: np.ndarray) -> float:
    ratios = []
    for label in np.unique(source_y):
        rows = source[source_y == label]
        if not (target_y == label).any():
            continue
        centre = rows.mean(axis=0)
        radius = np.linalg.norm(rows - centre, axis=1).mean()
        moved = np.linalg.norm(target[target_y == label].mean(axis=0) - centre)
        ratios.append(moved / radius)
    return float(np.mean(ratios))


@torch.no_grad()
def features(model, normalizer, x, device, batch=512):
    inputs = to_input(normalize.apply(normalizer, x))
    out = [model.features(inputs[i:i + batch].to(device)).flatten(1).cpu().numpy()
           for i in range(0, len(inputs), batch)]
    return np.concatenate(out)


def main(paths: list[str], device: str = "cpu") -> None:
    log = TargetAccessLog()
    source_x, source_y = load_source()
    targets = {i: load_target(i, log) for i in TARGET_BATCHES}

    print(f"{'checkpoint':<26}{'input':>9}{'z_f':>9}{'removed':>10}")
    for path in paths:
        model, payload = load_checkpoint(Path(path), device)
        normalizer = payload["normalizer"]
        source_z = normalize.apply(normalizer, source_x)
        source_f = features(model, normalizer, source_x, device)

        raw, deep = [], []
        for index, (x, y) in targets.items():
            raw.append(drift_ratio(source_z, source_y,
                                   normalize.apply(normalizer, x), y))
            deep.append(drift_ratio(source_f, source_y,
                                    features(model, normalizer, x, device), y))
        a, b = float(np.mean(raw)), float(np.mean(deep))
        name = f"{payload['variant']} s{payload['seed']}"
        print(f"{name:<26}{a:>9.3f}{b:>9.3f}{1 - b / a:>9.1%}")


if __name__ == "__main__":
    main(sys.argv[1:])
