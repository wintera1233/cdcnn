#!/usr/bin/env python
"""v12 step 0: does Batch 1 stay separable once the drift subspace is removed?

Batch 1 only. No target file is opened, no checkpoint is used, no GPU.

For each candidate subspace U (proposal-v12.md, section "設計"):

    eth   k=1  Ethanol's acquisition-block offset
    pc1   k=1  first right singular vector of the three unit offsets, uncentred
    sub3  k=3  the span of all three offsets

the projection P = I - U U^T is applied to Batch 1 in the signed-log ->
per-sample space, and three things are measured against the unprojected data:

  1. the fifteen class-pair separations, centroid distance over the mean of the
     two within-class radii;
  2. how much of Batch 1's total variance and between-class variance the
     removed subspace carried;
  3. five-fold stratified cross-validation accuracy on Batch 1, with the same
     fixed folds `src.cv` uses, under two cheap classifiers: nearest centroid
     and multinomial logistic regression. These stand in for the network until
     the normaliser is implemented; the network CV is the gate in step 1.

Writes `reports/projection_separability.json`.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import normalize  # noqa: E402
from src.augment import block_offsets  # noqa: E402
from src.cv import stratified_folds  # noqa: E402
from src.data import GAS_LABELS, load_source  # noqa: E402


def candidate_subspaces(z: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    offsets = block_offsets(z, y)
    unit = {label: v / np.linalg.norm(v) for label, v in offsets.items()}
    stacked = np.stack([unit[label] for label in sorted(unit)])
    _, _, vh = np.linalg.svd(stacked, full_matrices=False)
    basis3, _ = np.linalg.qr(stacked.T)
    return {"eth": unit[1][:, None], "pc1": vh[0][:, None], "sub3": basis3}


def project(z: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return z - (z @ basis) @ basis.T


def separations(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    centroids = {c: z[y == c].mean(axis=0) for c in sorted(GAS_LABELS)}
    radii = {c: float(np.linalg.norm(z[y == c] - centroids[c], axis=1).mean())
             for c in centroids}
    out = {}
    for a, b in combinations(sorted(GAS_LABELS), 2):
        d = float(np.linalg.norm(centroids[a] - centroids[b]))
        out[f"{GAS_LABELS[a]}-{GAS_LABELS[b]}"] = d / (0.5 * (radii[a] + radii[b]))
    return out


def variance_shares(z: np.ndarray, y: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    centred = z - z.mean(axis=0)
    total = float((centred ** 2).sum())
    removed_total = float(((centred @ basis) ** 2).sum())
    grand = z.mean(axis=0)
    between = np.stack([np.sqrt((y == c).sum()) * (z[y == c].mean(axis=0) - grand)
                        for c in sorted(GAS_LABELS)])
    between_total = float((between ** 2).sum())
    removed_between = float(((between @ basis) ** 2).sum())
    return {"total_variance_removed": removed_total / total,
            "between_class_variance_removed": removed_between / between_total}


def cross_validate(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    folds = stratified_folds(y)
    scores = {"nearest_centroid": [], "logistic": []}
    for held in folds:
        train = np.setdiff1d(np.arange(len(y)), held)
        centroids = np.stack([z[train][y[train] == c].mean(axis=0)
                              for c in sorted(GAS_LABELS)])
        nearest = np.argmin(((z[held][:, None, :] - centroids[None]) ** 2).sum(-1), axis=1) + 1
        scores["nearest_centroid"].append(float((nearest == y[held]).mean()))
        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(z[train], y[train])
        scores["logistic"].append(float((clf.predict(z[held]) == y[held]).mean()))
    return {k: float(np.mean(v)) for k, v in scores.items()}


def main() -> int:
    x, y = load_source()
    z = normalize.apply(normalize.fit("signed_log_then_per_sample", x), x)
    subspaces = candidate_subspaces(z, y)

    report = {"none": {"separations": separations(z, y),
                       "cv": cross_validate(z, y)}}
    for name, basis in subspaces.items():
        zp = project(z, basis)
        report[name] = {"k": int(basis.shape[1]),
                        "separations": separations(zp, y),
                        "variance": variance_shares(z, y, basis),
                        "cv": cross_validate(zp, y)}
    out = ROOT / "reports" / "projection_separability.json"
    out.write_text(json.dumps(report, indent=2))
    print(out, "\n")

    names = ["none"] + list(subspaces)
    print(f"{'5-fold CV on Batch 1':<24}" + "".join(f"{n:>10}" for n in names))
    for clf in ("nearest_centroid", "logistic"):
        print(f"{clf:<24}" + "".join(f"{report[n]['cv'][clf]:>10.4f}" for n in names))
    print(f"\n{'variance removed':<24}" + "".join(f"{n:>10}" for n in subspaces))
    for key in ("total_variance_removed", "between_class_variance_removed"):
        print(f"{key:<24}" + "".join(f"{report[n]['variance'][key]:>10.3f}" for n in subspaces))
    print(f"\n{'pair separation':<26}" + "".join(f"{n:>9}" for n in names))
    pairs = sorted(report["none"]["separations"], key=lambda p: report["none"]["separations"][p])
    for pair in pairs:
        print(f"{pair:<26}" + "".join(f"{report[n]['separations'][pair]:>9.2f}" for n in names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
