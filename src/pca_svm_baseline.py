"""Leakage-safe PCA/RBF-SVM baseline using the PCA project's data loader."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.pca_analysis import load_batch, save_json, sha256, versions


def _pipeline(retention, c_value, gamma, pca_cfg):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=retention, svd_solver=pca_cfg["svd_solver"],
                    whiten=pca_cfg["whiten"])),
        ("svm", SVC(kernel="rbf", C=c_value, gamma=gamma)),
    ])


def _exact_duplicate_groups(x):
    """Give identical feature rows the same stable integer group."""
    _, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    return inverse, counts


def main(config_path="configs/pca_svm.json"):
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    pca_config_file = (root / cfg["pca_config"]).resolve()
    data_cfg = json.loads(pca_config_file.read_text(encoding="utf-8"))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_pca_svm"
    out = root / data_cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)

    arrays, labels, line_numbers, manifests = {}, {}, {}, []
    ds = data_cfg["dataset"]
    for batch in ds["expected_batch_ids"]:
        path = root / ds["path"] / ds["batch_file_pattern"].format(batch_id=batch)
        x, y, lines = load_batch(path, batch, ds["expected_feature_count"])
        arrays[batch], labels[batch], line_numbers[batch] = x, y, lines
        manifests.append({"batch": batch, "path": str(path.relative_to(root)),
                          "bytes": path.stat().st_size, "sha256": sha256(path),
                          "records": int(len(x))})

    expected = (13910, 10, 6, 128)
    observed = (sum(map(len, arrays.values())), len(arrays),
                len(np.unique(np.concatenate(list(labels.values())))),
                next(iter(arrays.values())).shape[1])
    if observed != expected:
        raise ValueError(f"Dataset validation failed: observed {observed}, expected {expected}")
    all_x = np.vstack(list(arrays.values()))
    if not np.isfinite(all_x).all():
        raise ValueError("Dataset contains non-finite feature values")

    source = cfg["source_batch"]
    x_source, y_source = arrays[source], labels[source]
    groups, group_counts = _exact_duplicate_groups(x_source)
    split = StratifiedGroupKFold(n_splits=cfg["cv"]["folds"], shuffle=True,
                                random_state=cfg["cv"]["seed"])
    folds = list(split.split(x_source, y_source, groups))
    fold_assignment = np.empty(len(x_source), dtype=int)
    for fold, (train_idx, valid_idx) in enumerate(folds, 1):
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise RuntimeError("An exact-duplicate group crosses a CV fold boundary")
        fold_assignment[valid_idx] = fold
    pd.DataFrame({"sample_id": [f"batch{source}:line{v}" for v in line_numbers[source]],
                  "line_number": line_numbers[source], "gas_label": y_source,
                  "duplicate_group": groups, "cv_fold": fold_assignment}).to_csv(
                      out / "cv_fold_assignments.csv", index=False)

    fold_rows = []
    candidate = 0
    pca_cfg = cfg["pipeline"]["pca"]
    for retention in cfg["search"]["pca_variance_retention"]:
        for c_value in cfg["search"]["C"]:
            for gamma in cfg["search"]["gamma"]:
                candidate += 1
                for fold, (train_idx, valid_idx) in enumerate(folds, 1):
                    model = _pipeline(retention, c_value, gamma, pca_cfg)
                    model.fit(x_source[train_idx], y_source[train_idx])
                    predicted = model.predict(x_source[valid_idx])
                    fold_rows.append({"candidate_id": candidate, "fold": fold,
                                      "pca_variance_retention": retention, "C": c_value,
                                      "gamma": gamma, "train_samples": len(train_idx),
                                      "validation_samples": len(valid_idx),
                                      "pca_components": model.named_steps["pca"].n_components_,
                                      "accuracy": accuracy_score(y_source[valid_idx], predicted)})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out / "cv_fold_results.csv", index=False)
    result_df = (fold_df.groupby(["candidate_id", "pca_variance_retention", "C", "gamma"],
                                 sort=False, dropna=False)
                 .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
                      min_pca_components=("pca_components", "min"),
                      max_pca_components=("pca_components", "max"))
                 .reset_index())
    # Stable candidate order is the documented tie breaker.
    result_df["rank"] = result_df["mean_accuracy"].rank(method="min", ascending=False).astype(int)
    result_df.to_csv(out / "cv_results.csv", index=False)
    best = result_df.loc[result_df["mean_accuracy"].idxmax()]

    best_gamma = best.gamma
    try:
        best_gamma = float(best_gamma)
    except (TypeError, ValueError):
        pass
    final_model = _pipeline(float(best.pca_variance_retention), float(best.C), best_gamma, pca_cfg)
    final_model.fit(x_source, y_source)
    joblib.dump(final_model, out / "model.joblib")

    prediction_frames, metric_rows = [], []
    for batch in ds["expected_batch_ids"]:
        if batch == source:
            continue
        pred = final_model.predict(arrays[batch])
        correct = pred == labels[batch]
        metric_rows.append({"batch": batch, "samples": len(pred), "accuracy": correct.mean()})
        prediction_frames.append(pd.DataFrame({
            "sample_id": [f"batch{batch}:line{v}" for v in line_numbers[batch]],
            "batch": batch, "line_number": line_numbers[batch],
            "true_gas_label": labels[batch], "predicted_gas_label": pred,
            "correct": correct}))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "batch_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(out / "predictions.csv.gz", index=False,
                                                           compression="gzip")

    resolved_cfg = json.loads(json.dumps(cfg))
    resolved_cfg["resolved_dataset_configuration"] = data_cfg
    resolved_cfg["resolved_dataset_path"] = str((root / ds["path"]).resolve())
    save_json(out / "configuration.json", resolved_cfg)
    save_json(out / "input_manifest.json", manifests)
    save_json(out / "software_versions.json", versions())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = None
    summary = {
        "best_parameters": {"pca_variance_retention": float(best.pca_variance_retention),
                            "C": float(best.C), "gamma": best_gamma},
        "best_cv_accuracy": float(best.mean_accuracy),
        "final_pca_components": int(final_model.named_steps["pca"].n_components_),
        "target_unweighted_mean_accuracy": float(metrics.accuracy.mean()),
    }
    save_json(out / "summary.json", summary)
    manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"{sys.executable} scripts/run_pca_svm.py --config {config_path}",
        "git_revision": revision, "input_config_sha256": sha256(config_file),
        "pca_project_config_sha256": sha256(pca_config_file),
        "preprocessing_fit_scope": "Each CV pipeline fit only on its Batch 1 training fold; final pipeline fit on all Batch 1; Batches 2-10 transform/predict only.",
        "selection_scope": "5-fold Batch 1 CV accuracy only; target batches never used for tuning.",
        "duplicate_handling": {"method": "StratifiedGroupKFold over exact feature-row groups",
                               "exact_duplicate_groups_in_batch_1": int((group_counts > 1).sum()),
                               "records_in_duplicate_groups_in_batch_1": int(group_counts[group_counts > 1].sum()),
                               "duplicates_removed": 0},
        "exclusions": [], "summary": summary,
    }
    save_json(out / "run_manifest.json", manifest)
    _write_report(out, summary, metrics)
    save_json(out / "output_inventory.json", sorted(p.name for p in out.iterdir()
                                                     if p.name != "output_inventory.json"))
    return out


def _write_report(out, summary, metrics):
    rows = "\n".join(f"| {int(r.batch)} | {int(r.samples):,} | {r.accuracy:.6f} |"
                     for r in metrics.itertuples(index=False))
    p = summary["best_parameters"]
    text = f"""# PCA–SVM baseline report

This run is a project baseline, not a reproduction of verified paper settings. The selected pipeline is `StandardScaler → PCA → RBF-SVM`, with PCA retaining {p['pca_variance_retention']:.0%} variance, C={p['C']:g}, and gamma={p['gamma']}. It achieved {summary['best_cv_accuracy']:.6f} mean accuracy in five-fold stratified, duplicate-group-aware Batch 1 CV (seed 42). The refit on all Batch 1 retained {summary['final_pca_components']} principal components.

| Target batch | Samples | Accuracy |
|---:|---:|---:|
{rows}

The unweighted mean accuracy across Batches 2–10 is **{summary['target_unweighted_mean_accuracy']:.6f}**. Each batch contributes equally to this mean, irrespective of sample count.

All scaler, PCA, and SVM fitting during selection occurred inside each Batch 1 fold. The final complete pipeline was fit only on all Batch 1; saved PCA scores were not reused. Batches 2–10 were used only once for final evaluation and did not influence tuning. Exact duplicates were retained, with each identical-feature group constrained to one validation fold. No records were excluded, deduplicated, clipped, or imputed.
"""
    (out / "report.md").write_text(text, encoding="utf-8")
