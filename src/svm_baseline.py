"""Leakage-safe no-PCA RBF-SVM baseline using saved Batch 1 folds."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.pca_analysis import load_batch, save_json, sha256, versions


def _pipeline(c_value, gamma):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=c_value, gamma=gamma)),
    ])


def _load_saved_folds(path, line_numbers, labels, source_batch):
    saved = pd.read_csv(path)
    expected_columns = {"sample_id", "line_number", "gas_label", "cv_fold"}
    missing = expected_columns - set(saved.columns)
    if missing:
        raise ValueError(f"Saved CV fold file is missing columns: {sorted(missing)}")
    expected_ids = [f"batch{source_batch}:line{line}" for line in line_numbers]
    if len(saved) != len(expected_ids) or saved.sample_id.duplicated().any():
        raise ValueError("Saved CV folds do not contain exactly one row per Batch 1 sample")
    indexed = saved.set_index("sample_id")
    if set(indexed.index) != set(expected_ids):
        raise ValueError("Saved CV sample identities do not match loaded Batch 1")
    ordered = indexed.loc[expected_ids].reset_index()
    if not np.array_equal(ordered.line_number.to_numpy(), line_numbers):
        raise ValueError("Saved CV line numbers do not match loaded Batch 1")
    if not np.array_equal(ordered.gas_label.to_numpy(), labels):
        raise ValueError("Saved CV labels do not match loaded Batch 1")
    fold_ids = ordered.cv_fold.to_numpy(dtype=int)
    unique = np.unique(fold_ids)
    if not np.array_equal(unique, np.arange(1, len(unique) + 1)) or len(unique) < 2:
        raise ValueError(f"Saved CV fold IDs are invalid: {unique.tolist()}")
    folds = [(np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold))
             for fold in unique]
    return ordered, folds


def _read_pca_comparison(run_dir):
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "batch_metrics.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    if set(metrics.batch) != set(range(2, 11)):
        raise ValueError("PCA-SVM comparison run does not contain Batches 2-10")
    return summary, metrics, summary_path, metrics_path


def main(config_path="configs/svm.json"):
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    data_config_file = (root / cfg["pca_config"]).resolve()
    data_cfg = json.loads(data_config_file.read_text(encoding="utf-8"))
    fold_file = (root / cfg["saved_cv_folds"]).resolve()
    pca_run = (root / cfg["pca_svm_run"]).resolve()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_svm_no_pca"
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

    observed = (sum(map(len, arrays.values())), len(arrays),
                len(np.unique(np.concatenate(list(labels.values())))),
                next(iter(arrays.values())).shape[1])
    expected = (13910, 10, 6, cfg["pipeline"]["feature_count"])
    if observed != expected:
        raise ValueError(f"Dataset validation failed: observed {observed}, expected {expected}")
    if not np.isfinite(np.vstack(list(arrays.values()))).all():
        raise ValueError("Dataset contains non-finite feature values")

    source = cfg["source_batch"]
    x_source, y_source = arrays[source], labels[source]
    saved_folds, folds = _load_saved_folds(
        fold_file, line_numbers[source], y_source, source)
    shutil.copyfile(fold_file, out / "cv_fold_assignments.csv")

    fold_rows = []
    candidate = 0
    for c_value in cfg["search"]["C"]:
        for gamma in cfg["search"]["gamma"]:
            candidate += 1
            for fold, (train_idx, valid_idx) in enumerate(folds, 1):
                model = _pipeline(c_value, gamma)
                model.fit(x_source[train_idx], y_source[train_idx])
                predicted = model.predict(x_source[valid_idx])
                fold_rows.append({"candidate_id": candidate, "fold": fold, "C": c_value,
                                  "gamma": gamma, "train_samples": len(train_idx),
                                  "validation_samples": len(valid_idx),
                                  "accuracy": accuracy_score(y_source[valid_idx], predicted)})
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out / "cv_fold_results.csv", index=False)
    result_df = (fold_df.groupby(["candidate_id", "C", "gamma"], sort=False, dropna=False)
                 .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"))
                 .reset_index())
    result_df["rank"] = result_df.mean_accuracy.rank(method="min", ascending=False).astype(int)
    result_df.to_csv(out / "cv_results.csv", index=False)
    best = result_df.loc[result_df.mean_accuracy.idxmax()]
    best_gamma = best.gamma
    try:
        best_gamma = float(best_gamma)
    except (TypeError, ValueError):
        pass

    final_model = _pipeline(float(best.C), best_gamma)
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
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        out / "predictions.csv.gz", index=False, compression="gzip")

    pca_summary, pca_metrics, pca_summary_path, pca_metrics_path = _read_pca_comparison(pca_run)
    comparison = metrics[["batch", "samples", "accuracy"]].rename(
        columns={"accuracy": "svm_no_pca_accuracy"}).merge(
            pca_metrics[["batch", "accuracy"]].rename(
                columns={"accuracy": "pca_svm_accuracy"}), on="batch", validate="one_to_one")
    comparison["svm_no_pca_minus_pca_svm"] = (
        comparison.svm_no_pca_accuracy - comparison.pca_svm_accuracy)
    comparison.to_csv(out / "comparison.csv", index=False)

    summary = {
        "best_parameters": {"C": float(best.C), "gamma": best_gamma},
        "best_cv_accuracy": float(best.mean_accuracy),
        "feature_count": int(x_source.shape[1]),
        "target_unweighted_mean_accuracy": float(metrics.accuracy.mean()),
        "pca_svm_comparison": {
            "run_id": pca_run.name,
            "best_parameters": pca_summary["best_parameters"],
            "best_cv_accuracy": pca_summary["best_cv_accuracy"],
            "target_unweighted_mean_accuracy": pca_summary["target_unweighted_mean_accuracy"],
        },
    }
    resolved_cfg = json.loads(json.dumps(cfg))
    resolved_cfg["resolved_dataset_configuration"] = data_cfg
    resolved_cfg["resolved_dataset_path"] = str((root / ds["path"]).resolve())
    resolved_cfg["resolved_saved_cv_folds"] = str(fold_file)
    resolved_cfg["resolved_pca_svm_run"] = str(pca_run)
    save_json(out / "configuration.json", resolved_cfg)
    input_artifacts = manifests + [
        {"role": "saved_batch_1_cv_folds", "path": str(fold_file.relative_to(root)),
         "bytes": fold_file.stat().st_size, "sha256": sha256(fold_file),
         "records": int(len(saved_folds))},
        {"role": "pca_svm_summary", "path": str(pca_summary_path.relative_to(root)),
         "bytes": pca_summary_path.stat().st_size, "sha256": sha256(pca_summary_path)},
        {"role": "pca_svm_batch_metrics", "path": str(pca_metrics_path.relative_to(root)),
         "bytes": pca_metrics_path.stat().st_size, "sha256": sha256(pca_metrics_path)},
    ]
    save_json(out / "input_manifest.json", input_artifacts)
    save_json(out / "software_versions.json", versions())
    save_json(out / "summary.json", summary)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = None
    manifest = {
        "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": f"{sys.executable} scripts/run_svm.py --config {config_path}",
        "git_revision": revision, "input_config_sha256": sha256(config_file),
        "data_config_sha256": sha256(data_config_file),
        "saved_cv_folds_sha256": sha256(fold_file),
        "preprocessing_fit_scope": "StandardScaler fit inside each saved Batch 1 CV training fold; final scaler and SVM fit on all Batch 1; Batches 2-10 transform/predict only.",
        "selection_scope": "Saved Batch 1 CV folds and mean accuracy only; target batches never used for tuning.",
        "feature_scope": "All 128 original extracted features; no PCA or feature selection.",
        "sample_policy": "All valid samples retained; no deduplication, clipping, imputation, or resampling.",
        "exclusions": [], "summary": summary,
    }
    save_json(out / "run_manifest.json", manifest)
    _write_report(out, summary, comparison)
    save_json(out / "output_inventory.json", sorted(
        p.name for p in out.iterdir() if p.name != "output_inventory.json"))
    return out


def _write_report(out, summary, comparison):
    p = summary["best_parameters"]
    old = summary["pca_svm_comparison"]
    old_p = old["best_parameters"]
    rows = "\n".join(
        f"| {int(r.batch)} | {int(r.samples):,} | {r.svm_no_pca_accuracy:.6f} | "
        f"{r.pca_svm_accuracy:.6f} | {r.svm_no_pca_minus_pca_svm:+.6f} |"
        for r in comparison.itertuples(index=False))
    text = f"""# No-PCA SVM baseline report

The selected pipeline is `StandardScaler → SVC(kernel=rbf)` using all 128 original extracted features. The Batch 1 search selected C={p['C']:g} and gamma={p['gamma']}, with mean saved-fold CV accuracy {summary['best_cv_accuracy']:.6f}. The model was then refit on all Batch 1 samples.

## Comparison with PCA-SVM

The earlier PCA-SVM run `{old['run_id']}` selected PCA retention={old_p['pca_variance_retention']:.2f}, C={old_p['C']:g}, and gamma={old_p['gamma']}, with Batch 1 CV accuracy {old['best_cv_accuracy']:.6f}.

| Target batch | Samples | No-PCA SVM accuracy | PCA-SVM accuracy | Difference |
|---:|---:|---:|---:|---:|
{rows}

The unweighted mean accuracy across Batches 2–10 is **{summary['target_unweighted_mean_accuracy']:.6f}** for no-PCA SVM and **{old['target_unweighted_mean_accuracy']:.6f}** for PCA-SVM. Each target batch contributes equally.

The saved Batch 1 folds were reused exactly and verified against loader-derived sample identities, line numbers, and gas labels. During CV, each scaler was fit only on that fold's training samples. Hyperparameters were selected only by Batch 1 CV accuracy; target results were computed after selection and never used for tuning. No records were excluded, deduplicated, clipped, imputed, or resampled.
"""
    (out / "report.md").write_text(text, encoding="utf-8")
