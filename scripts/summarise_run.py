#!/usr/bin/env python
"""Summarise a finished run: target mean, SD, separability, per-batch, per-class.

Post-hoc reporting only; reads the run's own `target_summary.json` and
`target_results.json`, which the run wrote after every checkpoint was frozen.

    python scripts/summarise_run.py runs/<run> --reference R-gen

Separability follows the configs' `rung_difference_standard_errors`: two
cells differ when |mean_a - mean_b| exceeds 2 * sqrt((sd_a^2 + sd_b^2) / n).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GASES = ["Ethanol", "Ammonia", "Ethylene", "Acetaldehyde", "Acetone", "Toluene"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--reference", default=None,
                        help="variant name every other cell is compared against")
    parser.add_argument("--standard-errors", type=float, default=2.0)
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "target_summary.json").read_text())
    results = json.loads((run / "target_results.json").read_text())

    cells = {k.split("@")[0]: v for k, v in summary.items()}
    n = len(next(iter(cells.values()))["seeds"])
    ref = args.reference or max(cells, key=lambda k: cells[k]["target_mean"])
    ref_mean, ref_sd = cells[ref]["target_mean"], cells[ref]["target_mean_sd"]

    print(f"run {run.name}: {len(cells)} cells x {n} seeds, reference {ref}\n")
    print(f"{'variant':<14}{'Batch 1':>9}{'target':>9}{'SD':>8}{'vs ref':>9}"
          f"{'threshold':>11}{'separable':>11}")
    for name, cell in sorted(cells.items(), key=lambda kv: -kv[1]["target_mean"]):
        diff = cell["target_mean"] - ref_mean
        threshold = args.standard_errors * np.sqrt(
            (cell["target_mean_sd"] ** 2 + ref_sd ** 2) / n)
        sep = "-" if name == ref else ("yes" if abs(diff) > threshold else "no")
        print(f"{name:<14}{cell['source_accuracy']:>9.4f}{cell['target_mean']:>9.4f}"
              f"{cell['target_mean_sd']:>8.4f}{diff:>+9.4f}{threshold:>11.4f}{sep:>11}")

    print("\nper batch")
    batches = sorted(next(iter(cells.values()))["per_batch_accuracy"], key=int)
    print(f"{'variant':<14}" + "".join(f"{'B' + b:>7}" for b in batches) + f"{'pooled':>8}")
    for name, cell in sorted(cells.items(), key=lambda kv: -kv[1]["target_mean"]):
        row = "".join(f"{cell['per_batch_accuracy'][b]:>7.3f}" for b in batches)
        print(f"{name:<14}{row}{cell['pooled_accuracy']:>8.4f}")

    # Per-class recall, pooled over batches and seeds, from the confusion matrices.
    print("\nper-class recall (rows = true class, pooled over Batches 2-10 and seeds)")
    print(f"{'variant':<14}" + "".join(f"{g:>13}" for g in GASES))
    recalls = {}
    for entry in results:
        name = entry["cell"].split("@")[0]
        total = np.zeros((6, 6))
        for matrix in entry["confusion"].values():
            total += np.asarray(matrix, dtype=float)
        recalls.setdefault(name, []).append(total)
    for name in sorted(recalls, key=lambda k: -cells[k]["target_mean"]):
        total = sum(recalls[name])
        recall = np.diag(total) / np.maximum(total.sum(axis=1), 1)
        print(f"{name:<14}" + "".join(f"{r:>13.3f}" for r in recall))
    if ref in recalls:
        print(f"\nrecall change vs {ref}")
        base = sum(recalls[ref]); base = np.diag(base) / np.maximum(base.sum(axis=1), 1)
        for name in sorted(recalls, key=lambda k: -cells[k]["target_mean"]):
            if name == ref:
                continue
            total = sum(recalls[name])
            recall = np.diag(total) / np.maximum(total.sum(axis=1), 1)
            print(f"{name:<14}" + "".join(f"{d:>+13.3f}" for d in recall - base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
