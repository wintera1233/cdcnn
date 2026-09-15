#!/usr/bin/env python3
"""Phase-separated launcher for the complete canonical CDCNN v6.3 experiment.

The source worker processes can read only Batch 1.  A single controller verifies
all 20 frozen checkpoints before it starts the one target-loading phase.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdcnn_ablation import (
    CANONICAL_STAGES,
    CDCNNModel,
    DataAccessGuard,
    ProtocolError,
    _confusion_rows,
    _folds,
    _implementation_identity,
    _provenance_input_manifest,
    _software_versions,
    _training_arrays,
    _write_output_inventory,
    capture_nvidia_smi_evidence,
    predict,
    resolve_training_device,
    train_model,
    utc_now,
    validate_config,
)
from src.pca_analysis import save_json, sha256


SEEDS = (1042, 2024, 3407, 42, 123)
STAGES = ("B0", "A1", "A2-semantic", "A3")


def stage_slug(stage: str) -> str:
    return stage.lower().replace("-", "_")


def read_config(root: Path, config_path: str) -> tuple[Path, dict]:
    path = (root / config_path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    validate_config(cfg)
    if tuple(cfg["stages"]) != STAGES or tuple(cfg["seeds"]) != SEEDS:
        raise ProtocolError("The full launcher requires the canonical four stages and five seeds")
    return path, cfg


def enforce_cuda_worker_strategy(cfg: dict, max_workers: int) -> None:
    """Keep one fresh training subprocess at a time on the single configured GPU."""
    device = torch.device(cfg["training"]["device"])
    if device.type != "cuda":
        raise ProtocolError(f"The full v6 launcher requires CUDA, received {device}")
    if max_workers != 1:
        raise ProtocolError(
            "CUDA source training must use --max-workers 1; concurrent worker "
            "subprocesses would contend for one GPU")


def verify_gpu_smoke_gate(root: Path, config_file: Path, smoke_run: str) -> Path:
    """Require a passing, configuration-matched B0 CUDA smoke artifact."""
    config = json.loads(config_file.read_text(encoding="utf-8"))
    run = Path(smoke_run)
    if not run.is_absolute():
        run = root / run
    result_path = run / "gpu_smoke_result.json"
    if not result_path.is_file():
        raise ProtocolError(f"Missing GPU smoke result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    required = {
        "status": "passed", "stage": "B0", "source_batch": 1,
        "target_batches_loaded": False,
        "implementation_version": config["implementation_version"],
        "config_sha256": sha256(config_file),
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ProtocolError(
                f"GPU smoke gate mismatch for {key}: {result.get(key)!r} != {expected!r}")
    if not result.get("gpu_pid_visible_through_nvidia_smi"):
        raise ProtocolError("GPU smoke did not prove a process PID was visible through nvidia-smi")
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        raise ProtocolError(f"Missing GPU smoke manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_implementation = {
        entry["path"]: entry["sha256"] for entry in manifest.get("implementation_files", [])
    }
    for relative in ("src/cdcnn_ablation.py", "scripts/run_cdcnn_v6_full.py"):
        current = sha256(root / relative)
        if recorded_implementation.get(relative) != current:
            raise ProtocolError(
                f"GPU smoke implementation mismatch for {relative}; rerun gpu-smoke")
    return run.resolve()


def save_failure(path: Path, *, phase: str, exc: BaseException, extra: dict | None = None) -> dict:
    record = {
        "utc": utc_now(),
        "phase": phase,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    if extra:
        record.update(extra)
    save_json(path, record)
    return record


def source_worker(stage: str, seed: int, config_path: str, output: str) -> Path:
    """Train one stage/seed through CV and final fitting using Batch 1 only."""
    root = Path.cwd().resolve()
    config_file, cfg = read_config(root, config_path)
    if stage not in STAGES or seed not in SEEDS:
        raise ProtocolError(f"Noncanonical source task: stage={stage!r}, seed={seed!r}")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    access = DataAccessGuard(root, cfg["dataset"])
    device_audit: list[dict] = []
    started = utc_now()
    save_json(out / "config.json", {**cfg, "resolved_stage": stage, "resolved_seed": seed})
    save_json(out / "software_versions.json", _software_versions())
    try:
        source = access.load(1, "source-only validation, cross-validation, and final fitting")
        fold_frame, fold_ids = _folds(root, cfg, source)
        fold_frame.to_csv(out / "cv_fold_assignments.csv", index=False)
        history_rows: list[dict] = []
        cv_rows: list[dict] = []
        cv_predictions: list[pd.DataFrame] = []
        provenance_frames: list[pd.DataFrame] = []
        scaler_scopes: list[dict] = []

        for fold in range(1, 6):
            train_idx = np.flatnonzero(fold_ids != fold)
            valid_idx = np.flatnonzero(fold_ids == fold)
            scaler = StandardScaler().fit(source.x[train_idx])
            scaler_scopes.append({
                "seed": seed, "context": f"cv_fold_{fold}", "fit_batch": 1,
                "fit_original_samples": len(train_idx), "fit_augmented_samples": 0,
                "fit_target_samples": 0,
            })
            train_x, train_y, provenance = _training_arrays(
                stage, scaler.transform(source.x[train_idx]), source.y[train_idx], train_idx,
                source.line_numbers, seed, f"seed_{seed}_cv_fold_{fold}")
            if len(provenance):
                provenance.insert(0, "stage", stage)
                provenance.insert(1, "model_seed", seed)
                provenance_frames.append(provenance)
            model, history = train_model(
                stage, train_x, train_y, cfg, seed,
                validation=(scaler.transform(source.x[valid_idx]), source.y[valid_idx]),
                device_audit=device_audit, training_context=f"cv_fold_{fold}")
            history_rows.extend({"phase": "cv", "seed": seed, "fold": fold, **row}
                                for row in history)
            logits = predict(model, scaler.transform(source.x[valid_idx]), cfg["training"]["batch_size"])
            pred = logits.argmax(1) + 1
            cv_rows.append({
                "seed": seed, "fold": fold, "checkpoint_epoch": 100,
                "samples": len(valid_idx), "accuracy": float(accuracy_score(source.y[valid_idx], pred)),
            })
            cv_predictions.append(pd.DataFrame({
                "seed": seed, "fold": fold,
                "sample_id": [f"batch1:line{int(source.line_numbers[i])}" for i in valid_idx],
                "true_gas_label": source.y[valid_idx], "predicted_gas_label": pred,
                "correct": pred == source.y[valid_idx],
            }))
            print(f"source milestone stage={stage} seed={seed} fold={fold}/5 complete", flush=True)
            del model

        final_scaler = StandardScaler().fit(source.x)
        scaler_scopes.append({
            "seed": seed, "context": "final_fit", "fit_batch": 1,
            "fit_original_samples": len(source.x), "fit_augmented_samples": 0,
            "fit_target_samples": 0,
        })
        final_x, final_y, provenance = _training_arrays(
            stage, final_scaler.transform(source.x), source.y, np.arange(len(source.x)),
            source.line_numbers, seed, f"seed_{seed}_final_fit")
        if len(provenance):
            provenance.insert(0, "stage", stage)
            provenance.insert(1, "model_seed", seed)
            provenance_frames.append(provenance)
        model, history = train_model(
            stage, final_x, final_y, cfg, seed, device_audit=device_audit,
            training_context="final_fit")
        history_rows.extend({"phase": "final_fit", "seed": seed, "fold": pd.NA, **row}
                            for row in history)

        checkpoint_path = out / "model.pt"
        torch.save({
            "stage": stage, "seed": seed, "epoch": 100,
            "implementation_version": cfg["implementation_version"],
            "model_state_dict": model.state_dict(),
            "scaler_mean": final_scaler.mean_, "scaler_scale": final_scaler.scale_,
            "training": cfg["training"], "augmentation": cfg["augmentation"],
            "feature_generation": cfg["feature_generation"], "loss": cfg["loss"],
            "a3_numerical_stability": cfg["a3_numerical_stability"],
            "random_seeds": {"model_dataloader_and_a1": seed,
                             "feature_generation": seed + 2_000_000},
            "source_batch": 1, "target_batches_loaded": [],
        }, checkpoint_path)
        a3_has_batchnorm_running_state = stage == "A3" and any(
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats for module in model.modules())
        del model

        history_frame = pd.DataFrame(history_rows)
        cv_frame = pd.DataFrame(cv_rows)
        cv_prediction_frame = pd.concat(cv_predictions, ignore_index=True)
        history_frame.to_csv(out / "training_history.csv.gz", index=False, compression="gzip")
        cv_frame.to_csv(out / "cv_fold_metrics.csv", index=False)
        cv_prediction_frame.to_csv(out / "cv_predictions.csv.gz", index=False, compression="gzip")
        pd.DataFrame(scaler_scopes).to_csv(out / "scaler_fit_scopes.csv", index=False)
        save_json(out / "device_placement_audit.json", device_audit)
        style_rows = history_frame[history_frame.restyled_component.notna()][[
            "phase", "seed", "fold", "epoch", "restyled_component", "sampled_sigma_min",
            "sampled_sigma_max", "sampled_mean_min", "sampled_mean_max",
            "generation_nonfinite_tensor_count",
        ]]
        style_rows.to_csv(out / "style_sampling_statistics.csv.gz", index=False, compression="gzip")
        stability_rows = history_frame[[
            "phase", "seed", "fold", "epoch", "gradient_max_norm",
            "gradient_norm_before_clip_max", "gradient_norm_after_clip_max",
            "gradient_clipped_batches",
        ]]
        stability_rows.to_csv(
            out / "numerical_stability_statistics.csv.gz", index=False, compression="gzip")
        if provenance_frames:
            pd.concat(provenance_frames, ignore_index=True).to_csv(
                out / "augmentation_provenance.csv.gz", index=False, compression="gzip")

        save_json(out / "data_access_log.json", access.events)
        raw_loads = [event for event in access.events if event["event"] == "raw_file_load"]
        save_json(out / "input_manifest.json",
                  _provenance_input_manifest(root, config_file, cfg) + raw_loads)
        save_json(out / "feature_generation_geometry.json", {
            "stage": stage, "insertion": "between residual blocks 3 and 4",
            "block3_shape": ["B", 128, 128], "pool": {"kernel_size": 2, "stride": 2},
            "upsample": {"mode": "nearest", "length": 128},
            "statistics_shape": ["B", 128, 1], "statistics_axis": "length",
            "variance": "population over length (unbiased=False)",
            "batch_spread": "population standard deviation across active source mini-batch",
            "epsilon": cfg["feature_generation"]["epsilon"],
            "semantic_restyled_component": "pooled/upsampled low-frequency-like branch",
            "a3_numerical_stability": cfg["a3_numerical_stability"],
        })
        freeze = {
            "frozen_utc": utc_now(), "stage": stage, "seed": seed, "checkpoint_epoch": 100,
            "implementation_version": cfg["implementation_version"],
            "checkpoint_rule": "epoch 100 fixed before training; no early stopping",
            "selection_scope": "Batch 1 only", "target_batches_loaded_at_freeze": [],
            "checkpoint": {"path": "model.pt", "sha256": sha256(checkpoint_path)},
        }
        save_json(out / "model_freeze_manifest.json", freeze)

        history_ok = (
            len(history_frame) == 600
            and history_frame.groupby(["phase", "fold"], dropna=False).size().eq(100).all()
            and history_frame.epoch.max() == 100
        )
        checks = {
            "batch1_only_raw_load": [row["batch"] for row in raw_loads] == [1],
            "target_batches_never_loaded": not any(row.get("batch") in range(2, 11) for row in access.events),
            "exactly_100_epochs_for_each_of_five_folds_and_final_fit": bool(history_ok),
            "cv_prediction_coverage": len(cv_prediction_frame) == 445
            and cv_prediction_frame.sample_id.nunique() == 445,
            "all_scalers_fit_batch1_originals_only": all(
                row["fit_batch"] == 1 and row["fit_augmented_samples"] == 0
                and row["fit_target_samples"] == 0 for row in scaler_scopes),
            "checkpoint_epoch_100": freeze["checkpoint_epoch"] == 100,
            "generated_sigmas_positive_and_finite": bool(
                style_rows.empty or (style_rows.sampled_sigma_min.gt(0).all()
                                     and style_rows.generation_nonfinite_tensor_count.eq(0).all())),
            "a3_has_no_batchnorm_running_state": not a3_has_batchnorm_running_state,
            "a3_sigma_within_hard_bounds": bool(
                stage != "A3" or (
                    style_rows.sampled_sigma_min.ge(
                        cfg["a3_numerical_stability"]["sigma_min"]).all()
                    and style_rows.sampled_sigma_max.le(
                        cfg["a3_numerical_stability"]["sigma_max"]).all()
                )),
            "a3_gradients_clipped_to_max_norm": bool(
                stage != "A3" or stability_rows.gradient_norm_after_clip_max.le(
                    cfg["a3_numerical_stability"]["gradient_max_norm"] + 1e-5).all()),
            "six_training_contexts_verified_on_cuda": len(device_audit) == 6 and all(
                row["selected_device"].startswith("cuda")
                and row["model_parameter_device"] == row["selected_device"]
                and row["input_tensor_device"] == row["selected_device"]
                and row["label_tensor_device"] == row["selected_device"]
                and row["gpu_pid_visible_through_nvidia_smi"]
                for row in device_audit),
            "no_exclusions": True, "raw_data_unchanged": True,
        }
        audit = {
            "status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "preprocessing_fit_scope": "fold-local Batch 1 originals; final all-Batch-1 originals",
            "feature_statistics_scope": "active Batch 1 training mini-batch",
            "target_metrics_used_for_selection": False,
        }
        save_json(out / "source_only_audit.json", audit)
        save_json(out / "summary.json", {
            "stage": stage, "seed": seed, "status": "source_frozen",
            "source_cv_fold_accuracies": cv_frame.set_index("fold").accuracy.to_dict(),
            "source_cv_mean_accuracy": float(cv_frame.accuracy.mean()),
            "target_evaluation_performed": False,
        })
        save_json(out / "run_manifest.json", {
            "run_id": out.name, "created_utc": started, "completed_utc": utc_now(),
            "implementation_version": cfg["implementation_version"],
            "status": "source_frozen" if audit["status"] == "passed" else "failed",
            "stage": stage, "seed": seed, "implementation_files": _implementation_identity(root)
            + [{"path": "scripts/run_cdcnn_v6_full.py", "sha256": sha256(Path(__file__)),
                "bytes": Path(__file__).stat().st_size}],
            "preprocessing_fit_scope": audit["preprocessing_fit_scope"],
            "selection_scope": "Batch 1 only; fixed epoch 100", "exclusions": [],
            "raw_data_modified": False,
        })
        if audit["status"] != "passed":
            raise ProtocolError(f"Source-only audit failed for {stage} seed {seed}")
        print(f"source frozen stage={stage} seed={seed} checkpoint={checkpoint_path}", flush=True)
        return out
    except BaseException as exc:
        if device_audit and not (out / "device_placement_audit.json").exists():
            save_json(out / "device_placement_audit.json", device_audit)
        if not (out / "data_access_log.json").exists():
            save_json(out / "data_access_log.json", access.events)
        if not (out / "failure.json").exists():
            save_failure(out / "failure.json", phase="source_only_training", exc=exc,
                         extra={"stage": stage, "seed": seed})
        if not (out / "run_manifest.json").exists():
            save_json(out / "run_manifest.json", {
                "run_id": out.name, "created_utc": started, "failed_utc": utc_now(),
                "status": "failed", "stage": stage, "seed": seed,
                "exclusions": [], "raw_data_modified": False,
            })
        raise


def verify_source_runs(
    root: Path, cfg: dict, selections: list[dict], seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    expected_count = len(STAGES) * len(seeds)
    if len(selections) != expected_count:
        raise ProtocolError(
            f"Expected {expected_count} successful source runs, found {len(selections)}")
    expected = {(stage, seed) for stage in STAGES for seed in seeds}
    actual = {(row["stage"], int(row["seed"])) for row in selections}
    if actual != expected:
        raise ProtocolError(f"Source task coverage mismatch: missing={sorted(expected - actual)}")
    checkpoints = []
    for row in selections:
        run = Path(row["path"])
        if not run.is_absolute():
            run = root / run
        audit = json.loads((run / "source_only_audit.json").read_text())
        manifest = json.loads((run / "model_freeze_manifest.json").read_text())
        access = json.loads((run / "data_access_log.json").read_text())
        checkpoint = run / manifest["checkpoint"]["path"]
        if audit["status"] != "passed":
            raise ProtocolError(f"Source audit did not pass: {run}")
        if any(event.get("batch") in range(2, 11) for event in access):
            raise ProtocolError(f"Target access found in source run: {run}")
        if manifest["checkpoint_epoch"] != 100 or sha256(checkpoint) != manifest["checkpoint"]["sha256"]:
            raise ProtocolError(f"Frozen checkpoint verification failed: {run}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata_matches = (
            payload.get("stage") == row["stage"]
            and payload.get("seed") == int(row["seed"])
            and payload.get("epoch") == 100
            and payload.get("implementation_version") == cfg["implementation_version"]
            and payload.get("a3_numerical_stability") == cfg["a3_numerical_stability"]
        )
        del payload
        if not metadata_matches:
            raise ProtocolError(f"Frozen checkpoint v6.3 metadata mismatch: {run}")
        checkpoints.append({
            "stage": row["stage"], "seed": int(row["seed"]),
            "source_run": str(run.relative_to(root)), "checkpoint": str(checkpoint.relative_to(root)),
            "sha256": sha256(checkpoint), "frozen_utc": manifest["frozen_utc"],
        })
    return sorted(checkpoints, key=lambda r: (STAGES.index(r["stage"]), seeds.index(r["seed"])))


def markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if i else "---" for i in range(len(columns))) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def write_report(out: Path, metrics: pd.DataFrame, predictions: pd.DataFrame,
                 cv_metrics: pd.DataFrame, confusions: pd.DataFrame,
                 failures: list[dict], cfg: dict) -> None:
    seed_summary = metrics.groupby(["stage", "seed"], sort=False).agg(
        target_mean_accuracy=("accuracy", "mean"))
    pooled = predictions.groupby(["stage", "seed"], sort=False).correct.mean().rename("target_pooled_accuracy")
    cv = cv_metrics.groupby(["stage", "seed"], sort=False).accuracy.mean().rename("source_cv_accuracy")
    seed_summary = seed_summary.join(pooled).join(cv).reset_index()
    stage_summary = seed_summary.groupby("stage", sort=False).agg(
        source_cv_mean=("source_cv_accuracy", "mean"), source_cv_std=("source_cv_accuracy", "std"),
        target_mean=("target_mean_accuracy", "mean"), target_mean_std=("target_mean_accuracy", "std"),
        target_pooled_mean=("target_pooled_accuracy", "mean"),
        target_pooled_std=("target_pooled_accuracy", "std"),
    ).reset_index()
    batch_summary = metrics.groupby(["stage", "batch"], sort=False).accuracy.agg(["mean", "std"]).reset_index()

    stage_rows = [[r.stage, *(f"{value:.6f}" for value in (
        r.source_cv_mean, r.source_cv_std, r.target_mean, r.target_mean_std,
        r.target_pooled_mean, r.target_pooled_std))] for r in stage_summary.itertuples(index=False)]
    seed_rows = [[r.stage, str(r.seed), f"{r.source_cv_accuracy:.6f}",
                  f"{r.target_mean_accuracy:.6f}", f"{r.target_pooled_accuracy:.6f}"]
                 for r in seed_summary.itertuples(index=False)]
    per_batch_rows = [[r.stage, str(r.seed), str(r.batch), str(r.samples), str(r.correct),
                       f"{r.accuracy:.6f}"] for r in metrics.itertuples(index=False)]
    batch_rows = [[r.stage, str(r.batch), f"{r.mean:.6f}", f"{r.std:.6f}"]
                  for r in batch_summary.itertuples(index=False)]

    aggregate_conf = confusions.groupby(
        ["stage", "batch", "true_gas_label", "predicted_gas_label"], as_index=False
    )["count"].sum()
    confusion_sections = []
    for stage in STAGES:
        confusion_sections.append(f"### {stage}\n")
        for batch in range(2, 11):
            subset = aggregate_conf[(aggregate_conf.stage == stage) & (aggregate_conf.batch == batch)]
            matrix = subset.pivot(index="true_gas_label", columns="predicted_gas_label", values="count")
            matrix = matrix.reindex(index=range(1, 7), columns=range(1, 7), fill_value=0)
            rows = [[str(idx), *(str(int(value)) for value in matrix.loc[idx])] for idx in range(1, 7)]
            confusion_sections.append(
                f"Batch {batch} (counts pooled across five seeds)\n\n"
                + markdown_table(["true \\ pred", "1", "2", "3", "4", "5", "6"], rows) + "\n")

    if failures:
        failure_text = "\n".join(
            f"- `{row.get('phase', 'unknown')}`: {row.get('stage', '')} seed "
            f"{row.get('seed', '')}, attempt {row.get('attempt', '')}: "
            f"{row.get('exception_type', 'process failure')} — {row.get('message', '')}"
            for row in failures)
    else:
        failure_text = "No failures were recorded."

    report = f"""# CDCNN v6.3 full experiment report

## Outcome

All canonical experiments—B0, A1, A2-semantic, and A3—used seeds 1042, 2024,
3407, 42, and 123 with exactly 100 epochs for every Batch-1 CV fold and every
all-Batch-1 final fit. All 20 final checkpoints were frozen and hash-verified
before the single target-loading phase began. Batches 2–10 were each loaded once
into memory and then evaluated by every frozen checkpoint. No target result was
used for training, validation, checkpoint selection, or any other decision.

Standard deviations below are sample standard deviations across the five seeds
(`ddof=1`). Target mean is the unweighted mean of the nine batch accuracies;
target pooled accuracy weights target samples equally.

## Stage-level mean and standard deviation across seeds

{markdown_table(['Stage', 'CV mean', 'CV SD', 'Target mean', 'Target-mean SD', 'Pooled mean', 'Pooled SD'], stage_rows)}

## Per-seed summaries

{markdown_table(['Stage', 'Seed', 'Batch-1 CV', 'Target mean', 'Target pooled'], seed_rows)}

## Per-seed, per-batch results

{markdown_table(['Stage', 'Seed', 'Batch', 'Samples', 'Correct', 'Accuracy'], per_batch_rows)}

## Per-batch mean and standard deviation across seeds

{markdown_table(['Stage', 'Batch', 'Mean accuracy', 'SD'], batch_rows)}

## Confusion matrices

The matrices below pool counts across the five seeds for each stage and batch.
Exact per-seed, per-batch 6×6 counts are preserved in `confusion_matrices.csv`.
Rows are true gas labels and columns are predicted gas labels.

{chr(10).join(confusion_sections)}

## Failure records

{failure_text}

The machine-readable record is `failure_records.json`; failed attempts, if any,
remain in their original attempt directories and were not deleted or overwritten.

## Protocol and interpretation

Each CV scaler was fit only on the corresponding Batch-1 training fold, and each
final scaler was fit only on all original Batch-1 rows. No records were excluded,
deduplicated, clipped, or imputed. The feature positions are the 128 extracted
features (16 sensors × 8 features), not raw time samples. Cross-batch accuracy
changes are measured distribution shifts and do not, by themselves, establish a
specific sensor-drift mechanism.

The shared five-block 1D ResNet, feature-generation location, augmentation
equations, and probability MSE have the paper/project traceability described in
`{cfg['specification']}`. Momentum, exact learning-rate schedule, same-class
pairing, physical-semantic branch naming, projection dimension, temperature,
loss weights, A3 anchor-mean contrastive reduction, hard bounds, LayerNorm, and
gradient clipping are project-controlled.

## Artifact index

- `global_freeze_manifest.json`: all 20 checkpoint identities and freeze evidence
- `target_batch_metrics.csv`: exact stage/seed/batch metrics
- `target_predictions.csv.gz`: sample-level predictions
- `confusion_matrices.csv`: exact per-seed 6×6 confusion counts
- `source_cv_metrics.csv` and `source_cv_predictions.csv.gz`: Batch-1 CV evidence
- `dataset_validation.json`, `data_access_log.json`, and
  `leakage_target_access_audit.json`: structure and access-boundary evidence
- `failure_records.json`: empty or populated execution-failure ledger
"""
    (out / "experiment_report_v6_3.md").write_text(report, encoding="utf-8")
    seed_summary.to_csv(out / "per_seed_summary.csv", index=False)
    batch_summary.to_csv(out / "per_batch_summary.csv", index=False)
    stage_summary.to_csv(out / "stage_summary.csv", index=False)


def write_pilot_report(
    out: Path, metrics: pd.DataFrame, predictions: pd.DataFrame,
    cv_metrics: pd.DataFrame, confusions: pd.DataFrame,
    failures: list[dict], cfg: dict, seed: int,
) -> None:
    """Write a result report that cannot be mistaken for the five-seed experiment."""
    summary = metrics.groupby("stage", sort=False).agg(
        target_mean_accuracy=("accuracy", "mean"))
    pooled = predictions.groupby("stage", sort=False).correct.mean().rename(
        "target_pooled_accuracy")
    cv = cv_metrics.groupby("stage", sort=False).accuracy.mean().rename(
        "source_cv_accuracy")
    summary = summary.join(pooled).join(cv).reset_index()
    summary_rows = [[
        row.stage, str(seed), f"{row.source_cv_accuracy:.6f}",
        f"{row.target_mean_accuracy:.6f}", f"{row.target_pooled_accuracy:.6f}",
    ] for row in summary.itertuples(index=False)]
    per_batch_rows = [[
        row.stage, str(row.seed), str(row.batch), str(row.samples), str(row.correct),
        f"{row.accuracy:.6f}",
    ] for row in metrics.itertuples(index=False)]

    confusion_sections = []
    for stage in STAGES:
        confusion_sections.append(f"### {stage}\n")
        for batch in range(2, 11):
            subset = confusions[(confusions.stage == stage) & (confusions.batch == batch)]
            matrix = subset.pivot(
                index="true_gas_label", columns="predicted_gas_label", values="count")
            matrix = matrix.reindex(index=range(1, 7), columns=range(1, 7), fill_value=0)
            rows = [[str(idx), *(str(int(value)) for value in matrix.loc[idx])]
                    for idx in range(1, 7)]
            confusion_sections.append(
                f"Batch {batch} (seed {seed})\n\n"
                + markdown_table(["true \\ pred", "1", "2", "3", "4", "5", "6"], rows)
                + "\n")

    failure_text = "No failures were recorded."
    if failures:
        failure_text = "\n".join(
            f"- `{row.get('phase', 'unknown')}`: {row.get('stage', '')} seed "
            f"{row.get('seed', '')}, attempt {row.get('attempt', '')}: "
            f"{row.get('exception_type', 'process failure')} — {row.get('message', '')}"
            for row in failures)

    report = f"""# CDCNN v6.3 one-seed pilot report

## Outcome

This is a **one-seed pilot**, not the canonical five-seed comparison. B0, A1,
A2-semantic, and A3 used seed {seed}, with exactly 100 epochs for every Batch-1
CV fold and every all-Batch-1 final fit. A2-paper-literal was not included. All
four final checkpoints were frozen and hash-verified before the single target
loading phase began. Batches 2–10 were each loaded once and evaluated without
using target results for training, validation, checkpoint selection, or tuning.

No across-seed standard deviations are reported because this pilot has one seed.
Target mean is the unweighted mean of the nine batch accuracies; target pooled
accuracy weights target samples equally.

## Stage summary

{markdown_table(['Stage', 'Seed', 'Batch-1 CV', 'Target mean', 'Target pooled'], summary_rows)}

## Per-batch results

{markdown_table(['Stage', 'Seed', 'Batch', 'Samples', 'Correct', 'Accuracy'], per_batch_rows)}

## Confusion matrices

Exact seed-{seed}, per-batch 6×6 counts are preserved in
`confusion_matrices.csv`. Rows are true gas labels and columns are predicted gas
labels.

{chr(10).join(confusion_sections)}

## Failure records

{failure_text}

The machine-readable failure record is `failure_records.json`; failed attempts,
if any, remain in their unique attempt directories.

## Protocol and isolation

Each CV scaler was fit only on its Batch-1 training fold, and each final scaler
was fit only on all original Batch-1 rows. No records were excluded,
deduplicated, clipped, or imputed. All four source checkpoints were fixed at
epoch 100 before any target file was opened. The shared protocol and model
definitions are specified by `{cfg['specification']}`.

## Artifact index

- `global_freeze_manifest.json`: four checkpoint identities and freeze evidence
- `target_batch_metrics.csv`: exact stage/seed/batch metrics
- `target_predictions.csv.gz`: sample-level predictions
- `confusion_matrices.csv`: exact 6×6 confusion counts
- `source_cv_metrics.csv` and `source_cv_predictions.csv.gz`: Batch-1 CV evidence
- `dataset_validation.json`, `data_access_log.json`, and
  `leakage_target_access_audit.json`: structure and access-boundary evidence
- `failure_records.json`: execution-failure ledger
"""
    (out / "one_seed_pilot_report_v6_3.md").write_text(report, encoding="utf-8")
    summary.to_csv(out / "pilot_stage_summary.csv", index=False)


def evaluate(run_dir: str, config_path: str, selections_path: str,
             failures_path: str | None = None, seeds: tuple[int, ...] = SEEDS,
             report_kind: str = "full") -> Path:
    """Verify the global freeze, load every target once, evaluate, and report."""
    if report_kind not in ("full", "one_seed_pilot"):
        raise ValueError(f"Unknown report kind: {report_kind}")
    if report_kind == "one_seed_pilot" and len(seeds) != 1:
        raise ProtocolError("A one-seed pilot must contain exactly one seed")
    root = Path.cwd().resolve()
    out = Path(run_dir).resolve()
    config_file, cfg = read_config(root, config_path)
    device = resolve_training_device(cfg["training"])
    selections = json.loads(Path(selections_path).read_text(encoding="utf-8"))
    failures = json.loads(Path(failures_path).read_text()) if failures_path else []

    checkpoints = verify_source_runs(root, cfg, selections, seeds)
    expected_model_count = len(STAGES) * len(seeds)
    save_json(out / "global_freeze_manifest.json", {
        "frozen_utc": utc_now(), "status": "all_source_runs_frozen",
        "source_run_count": len(checkpoints), "checkpoint_epoch": 100,
        "selection_scope": "Batch 1 only", "target_batches_loaded_at_freeze": [],
        "checkpoints": checkpoints,
    })
    print(
        f"global milestone all {expected_model_count} source checkpoints verified and frozen",
        flush=True)

    access = DataAccessGuard(root, cfg["dataset"])
    source = access.load(1, "post-global-freeze dataset validation reference")
    access.freeze()
    target_data = {batch: access.load(batch, "single global post-freeze target load")
                   for batch in cfg["dataset"]["target_batches"]}
    print("target milestone Batches 2-10 loaded once after global freeze", flush=True)

    all_x = np.vstack([source.x] + [target_data[batch].x for batch in range(2, 11)])
    all_y = np.concatenate([source.y] + [target_data[batch].y for batch in range(2, 11)])
    dataset_validation = {
        "records": int(len(all_x)), "features": int(all_x.shape[1]),
        "batches": list(range(1, 11)), "gas_labels": np.unique(all_y).astype(int).tolist(),
        "batch_record_counts": {"1": len(source.x), **{
            str(batch): len(target_data[batch].x) for batch in range(2, 11)}},
        "finite_values": bool(np.isfinite(all_x).all()),
        "expected_structure_verified": bool(
            len(all_x) == 13910 and all_x.shape[1] == 128
            and np.array_equal(np.unique(all_y), np.arange(1, 7))),
        "validation_phase": "after global freeze; target files loaded once",
        "exclusions": [], "deduplication": False, "clipping": False, "imputation": False,
    }
    if not dataset_validation["expected_structure_verified"] or not dataset_validation["finite_values"]:
        raise ProtocolError("Post-freeze full dataset validation failed")
    save_json(out / "dataset_validation.json", dataset_validation)

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict] = []
    cv_frames: list[pd.DataFrame] = []
    cv_prediction_frames: list[pd.DataFrame] = []
    for entry in checkpoints:
        stage, seed = entry["stage"], entry["seed"]
        checkpoint = torch.load(root / entry["checkpoint"], map_location="cpu", weights_only=False)
        if (checkpoint["stage"] != stage or checkpoint["seed"] != seed
                or checkpoint["epoch"] != 100
                or checkpoint.get("implementation_version") != cfg["implementation_version"]
                or checkpoint.get("a3_numerical_stability") != cfg["a3_numerical_stability"]):
            raise ProtocolError(f"Checkpoint metadata mismatch: {entry['checkpoint']}")
        model = CDCNNModel(
            stage, cfg["feature_generation"]["epsilon"], cfg["loss"]["projection_dimension"],
            cfg["a3_numerical_stability"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        mean = np.asarray(checkpoint["scaler_mean"])
        scale = np.asarray(checkpoint["scaler_scale"])
        source_run = root / entry["source_run"]
        cv_part = pd.read_csv(source_run / "cv_fold_metrics.csv")
        cv_part.insert(0, "stage", stage)
        cv_frames.append(cv_part)
        cv_pred = pd.read_csv(source_run / "cv_predictions.csv.gz")
        cv_pred.insert(0, "stage", stage)
        cv_prediction_frames.append(cv_pred)
        for batch, data in target_data.items():
            scaled = (data.x - mean) / scale
            pred = predict(model, scaled, cfg["training"]["batch_size"]).argmax(1) + 1
            prediction_frames.append(pd.DataFrame({
                "stage": stage, "seed": seed, "batch": batch,
                "sample_id": [f"batch{batch}:line{int(line)}" for line in data.line_numbers],
                "line_number": data.line_numbers, "true_gas_label": data.y,
                "predicted_gas_label": pred, "correct": pred == data.y,
            }))
            metric_rows.append({
                "stage": stage, "seed": seed, "batch": batch, "samples": len(data.y),
                "correct": int((pred == data.y).sum()),
                "accuracy": float(accuracy_score(data.y, pred)),
            })
            confusion_rows.extend({"stage": stage, **row}
                                  for row in _confusion_rows(seed, batch, data.y, pred))
        print(f"target milestone evaluated stage={stage} seed={seed}", flush=True)
        del model, checkpoint

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    confusions = pd.DataFrame(confusion_rows)
    cv_metrics = pd.concat(cv_frames, ignore_index=True)
    cv_predictions = pd.concat(cv_prediction_frames, ignore_index=True)
    metrics.to_csv(out / "target_batch_metrics.csv", index=False)
    predictions.to_csv(out / "target_predictions.csv.gz", index=False, compression="gzip")
    confusions.to_csv(out / "confusion_matrices.csv", index=False)
    cv_metrics.to_csv(out / "source_cv_metrics.csv", index=False)
    cv_predictions.to_csv(out / "source_cv_predictions.csv.gz", index=False, compression="gzip")
    save_json(out / "failure_records.json", failures)
    save_json(out / "data_access_log.json", access.events)
    raw_loads = [event for event in access.events if event["event"] == "raw_file_load"]
    save_json(out / "input_manifest.json",
              _provenance_input_manifest(root, config_file, cfg) + raw_loads + [{
                  "role": "frozen_checkpoint", **entry} for entry in checkpoints])

    rebuilt = predictions.groupby(["stage", "seed", "batch"], as_index=False).agg(
        samples=("correct", "size"), correct=("correct", "sum"))
    rebuilt["accuracy"] = rebuilt.correct / rebuilt.samples
    left = metrics.sort_values(["stage", "seed", "batch"]).reset_index(drop=True)
    right = rebuilt.sort_values(["stage", "seed", "batch"]).reset_index(drop=True)
    access_batches = [row["batch"] for row in raw_loads]
    checks = {
        "all_expected_checkpoints_frozen_before_target_load": (
            len(checkpoints) == expected_model_count),
        "batch1_only_before_global_freeze": [event["batch"] for event in raw_loads
                                              if event["phase"] == "source_only"] == [1],
        "all_targets_loaded_post_freeze_once": access_batches == list(range(1, 11))
        and all(event["phase"] == "post_freeze" for event in raw_loads if event["batch"] >= 2),
        "target_prediction_coverage": len(predictions) == expected_model_count * 13465,
        "target_predictions_unique": not predictions.duplicated(["stage", "seed", "batch", "sample_id"]).any(),
        "target_metrics_recompute_from_predictions": bool(
            np.array_equal(left[["stage", "seed", "batch", "samples", "correct"]].to_numpy(),
                           right[["stage", "seed", "batch", "samples", "correct"]].to_numpy())
            and np.allclose(left.accuracy, right.accuracy, rtol=0, atol=0)),
        "confusion_matrix_coverage": len(confusions) == expected_model_count * 9 * 36,
        "source_cv_prediction_coverage": len(cv_predictions) == expected_model_count * 445,
        "dataset_structure_verified": dataset_validation["expected_structure_verified"],
        "no_exclusions": True, "raw_data_unchanged": True,
    }
    audit = {
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "target_access_influence": "none; target loading began after global checkpoint verification",
        "preprocessing_fit_scope": "fold-local Batch 1 originals; final all-Batch-1 originals",
        "target_metrics_used_for_selection": False,
    }
    save_json(out / "leakage_target_access_audit.json", audit)
    report_name = "experiment_report_v6_3.md"
    if report_kind == "one_seed_pilot":
        report_name = "one_seed_pilot_report_v6_3.md"
        write_pilot_report(
            out, metrics, predictions, cv_metrics, confusions, failures, cfg, seeds[0])
    else:
        write_report(out, metrics, predictions, cv_metrics, confusions, failures, cfg)
    save_json(out / "experiment_summary.json", {
        "status": audit["status"], "experiment_kind": report_kind,
        "implementation_version": cfg["implementation_version"],
        "stages": list(STAGES), "seeds": list(seeds),
        "epochs": 100, "source_run_count": expected_model_count,
        "target_batches": list(range(2, 11)), "target_raw_load_count": 9,
        "target_evaluation_models": expected_model_count, "report": report_name,
    })
    save_json(out / "software_versions.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "torch": torch.__version__,
        "platform": platform.platform(),
    })
    if audit["status"] != "passed":
        raise ProtocolError(f"{report_kind} experiment audit failed")
    print("experiment milestone evaluation, audit, and report complete", flush=True)
    return out


def run_attempt(root: Path, run_dir: Path, config_path: str, stage: str, seed: int,
                attempt: int) -> dict:
    task_root = run_dir / "source_runs" / stage_slug(stage) / f"seed_{seed}"
    task_root.mkdir(parents=True, exist_ok=True)
    output = task_root / f"attempt_{attempt}"
    log_path = run_dir / "logs" / f"source_{stage_slug(stage)}_seed_{seed}_attempt_{attempt}.log"
    command = [sys.executable, str(Path(__file__).resolve()), "source-worker", "--stage", stage,
               "--seed", str(seed), "--config", config_path, "--output", str(output)]
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        return_code = process.wait()
    result = {
        "stage": stage, "seed": seed, "attempt": attempt, "return_code": return_code,
        "path": str(output.relative_to(root)), "log": str(log_path.relative_to(root)),
    }
    if return_code == 0:
        result["status"] = "source_frozen"
    else:
        result["status"] = "failed"
        failure_path = output / "failure.json"
        if failure_path.exists():
            result.update(json.loads(failure_path.read_text()))
        else:
            result.update({"phase": "source_only_training", "exception_type": "ProcessFailure",
                           "message": f"source worker exited {return_code}"})
    return result


def controller(run_dir: str, config_path: str, max_workers: int, max_attempts: int,
               gpu_smoke_run: str, seeds: tuple[int, ...] = SEEDS,
               report_kind: str = "full") -> Path:
    root = Path.cwd().resolve()
    out = Path(run_dir).resolve()
    config_file, cfg = read_config(root, config_path)
    enforce_cuda_worker_strategy(cfg, max_workers)
    verified_smoke = verify_gpu_smoke_gate(root, config_file, gpu_smoke_run)
    if not seeds or any(seed not in SEEDS for seed in seeds):
        raise ProtocolError(f"Invalid source seed selection: {seeds}")
    if report_kind == "one_seed_pilot" and len(seeds) != 1:
        raise ProtocolError("The one-seed pilot controller requires exactly one seed")
    tasks = [(stage, seed) for stage in STAGES for seed in seeds]
    expected_task_count = len(tasks)

    def complete_task(task: tuple[str, int]) -> tuple[dict | None, list[dict]]:
        stage, seed = task
        failures: list[dict] = []
        for attempt in range(1, max_attempts + 1):
            result = run_attempt(root, out, config_path, stage, seed, attempt)
            if result["status"] == "source_frozen":
                return result, failures
            failures.append(result)
            print(f"source retry stage={stage} seed={seed} failed attempt={attempt}", flush=True)
        return None, failures

    print(
        f"controller started source_tasks={expected_task_count} max_workers={max_workers} "
        f"strategy=sequential_fresh_subprocess_no_multiprocessing_fork "
        f"gpu_smoke={verified_smoke}", flush=True)
    successes: list[dict] = []
    failures: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(complete_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            success, task_failures = future.result()
            failures.extend(task_failures)
            if success:
                successes.append(success)
                print(f"controller milestone source frozen {success['stage']} seed={success['seed']} "
                      f"({len(successes)}/{expected_task_count})", flush=True)

    save_json(out / "source_run_selections.json", successes)
    temporary_failures = out / "source_failure_records.json"
    save_json(temporary_failures, failures)
    if len(successes) != expected_task_count:
        save_json(out / "failure_records.json", failures)
        save_json(out / "controller_result.json", {
            "status": "failed_source_phase", "successful_source_runs": len(successes),
            "failed_tasks": expected_task_count - len(successes),
            "target_evaluation_started": False,
        })
        raise ProtocolError("At least one source task failed all attempts; target data remain unopened")

    # This verification is the global gate.  It reads no target dataset file.
    checkpoints = verify_source_runs(root, cfg, successes, seeds)
    if len(checkpoints) != expected_task_count:
        raise ProtocolError(
            f"Global source verification did not produce {expected_task_count} checkpoints")
    print("controller milestone source phase complete; starting one target phase", flush=True)
    try:
        evaluate(
            str(out), config_path, str(out / "source_run_selections.json"),
            str(temporary_failures), seeds=seeds, report_kind=report_kind)
    except BaseException as exc:
        record = save_failure(out / "evaluation_failure.json", phase="target_evaluation", exc=exc)
        failures.append(record)
        failure_path = out / "failure_records.json"
        if not failure_path.exists():
            save_json(failure_path, failures)
        else:
            save_json(out / "failure_records_controller.json", failures)
        save_json(out / "controller_result.json", {
            "status": "failed_target_phase", "successful_source_runs": expected_task_count,
            "target_evaluation_started": True,
        })
        raise
    save_json(out / "controller_result.json", {
        "status": "completed", "successful_source_runs": expected_task_count,
        "source_failed_attempts": len(failures), "target_evaluation_completed": True,
        "report": ("one_seed_pilot_report_v6_3.md" if report_kind == "one_seed_pilot"
                   else "experiment_report_v6_3.md"),
        "experiment_kind": report_kind, "completed_utc": utc_now(),
    })
    return out


def launch(config_path: str, max_workers: int, max_attempts: int,
           gpu_smoke_run: str, seeds: tuple[int, ...] = SEEDS,
           report_kind: str = "full") -> tuple[Path, int]:
    root = Path.cwd().resolve()
    config_file, cfg = read_config(root, config_path)
    enforce_cuda_worker_strategy(cfg, max_workers)
    verified_smoke = verify_gpu_smoke_gate(root, config_file, gpu_smoke_run)
    if report_kind == "one_seed_pilot" and len(seeds) != 1:
        raise ProtocolError("The one-seed pilot launcher requires exactly one seed")
    suffix = "cdcnn_v6_3_one_seed_pilot" if report_kind == "one_seed_pilot" else "cdcnn_v6_3_full"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_{suffix}"
    out = root / cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    save_json(out / "orchestration_config.json", {
        "run_id": run_id, "created_utc": utc_now(), "stages": list(STAGES),
        "implementation_version": cfg["implementation_version"],
        "seeds": list(seeds), "epochs": 100, "source_tasks": len(STAGES) * len(seeds),
        "experiment_kind": report_kind,
        "max_concurrent_source_jobs": max_workers, "max_attempts_per_source_task": max_attempts,
        "source_execution_strategy": "sequential fresh Python subprocesses; no fork-based CUDA multiprocessing",
        "verified_gpu_smoke_run": str(verified_smoke.relative_to(root)),
        "global_target_gate": (
            f"all {len(STAGES) * len(seeds)} source checkpoints frozen and verified"),
        "target_loading": "one phase; each Batch 2-10 raw file loaded once",
        "config_path": str(config_file.relative_to(root)), "config_sha256": sha256(config_file),
        "python_executable": sys.executable,
    })
    log_path = out / "logs" / "controller.log"
    controller_command = "pilot-controller" if report_kind == "one_seed_pilot" else "controller"
    command = [sys.executable, str(Path(__file__).resolve()), controller_command,
               "--run-dir", str(out),
               "--config", config_path, "--max-workers", str(max_workers),
               "--max-attempts", str(max_attempts), "--gpu-smoke-run", str(verified_smoke)]
    if report_kind == "one_seed_pilot":
        command.extend(["--seed", str(seeds[0])])
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    save_json(out / "launch_manifest.json", {
        "launched_utc": utc_now(), "controller_pid": process.pid,
        "controller_log": str(log_path.relative_to(root)), "detached": True,
        "command": command,
    })
    return out, process.pid


def run_gpu_smoke(config_path: str) -> Path:
    """Run one B0 optimization batch using Batch 1 only and prove CUDA residency."""
    root = Path.cwd().resolve()
    config_file, cfg = read_config(root, config_path)
    device = resolve_training_device(cfg["training"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_v6_3_b0_batch1_gpu_smoke"
    out = root / cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    save_json(out / "configuration.json", {**cfg, "resolved_stage": "B0"})
    save_json(out / "software_versions.json", {
        **_software_versions(), "torch_cuda_build": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(device),
    })
    access = DataAccessGuard(root, cfg["dataset"])
    device_audit: list[dict] = []
    try:
        source = access.load(1, "Batch-1-only B0 CUDA smoke test")
        fold_frame, fold_ids = _folds(root, cfg, source)
        fold = int(cfg["smoke"]["folds"][0])
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        scaler = StandardScaler().fit(source.x[train_idx])
        model, history = train_model(
            "B0", scaler.transform(source.x[train_idx]), source.y[train_idx], cfg,
            int(cfg["smoke"]["seeds"][0]), epochs=1, max_training_batches=1,
            validation=(scaler.transform(source.x[valid_idx]), source.y[valid_idx]),
            device_audit=device_audit, training_context=f"gpu_smoke_cv_fold_{fold}",
        )
        post_training_smi = capture_nvidia_smi_evidence(device)
        logits = predict(model, scaler.transform(source.x[valid_idx]), cfg["training"]["batch_size"])
        placement = device_audit[0] if device_audit else {}
        matching_rows = placement.get("nvidia_smi", {}).get("matching_process_rows", [])
        checks = {
            "stage_is_b0": True,
            "only_batch1_loaded": [
                event["batch"] for event in access.events if event["event"] == "raw_file_load"
            ] == [1],
            "no_target_batches_loaded": not any(
                event.get("batch") in range(2, 11) for event in access.events),
            "one_optimization_batch": len(history) == 1
            and history[0]["batches_seen"] == 1,
            "selected_device_is_cuda": placement.get("selected_device") == str(device),
            "model_parameter_on_selected_device": placement.get("model_parameter_device") == str(device),
            "input_tensor_on_selected_device": placement.get("input_tensor_device") == str(device),
            "label_tensor_on_selected_device": placement.get("label_tensor_device") == str(device),
            "first_training_batch_on_selected_device": set(
                placement.get("first_training_batch_device", {}).values()) == {str(device)},
            "gpu_pid_visible_through_nvidia_smi": bool(
                placement.get("gpu_pid_visible_through_nvidia_smi")),
            "nvidia_smi_reports_gpu_memory": bool(
                matching_rows and matching_rows[0]["used_memory_mib"] > 0),
            "torch_reports_gpu_memory": placement.get("nvidia_smi", {}).get(
                "torch_cuda_memory_allocated_bytes", 0) > 0,
            "validation_logits_finite": logits.shape == (len(valid_idx), 6)
            and bool(np.isfinite(logits).all()),
        }
        status = "passed" if all(checks.values()) else "failed"
        save_json(out / "device_placement_audit.json", device_audit)
        save_json(out / "post_training_nvidia_smi.json", post_training_smi)
        smi_record = placement.get("nvidia_smi", {})
        (out / "nvidia_smi.txt").write_text(
            "$ nvidia-smi --query-compute-apps=pid,process_name,used_memory "
            "--format=csv,noheader,nounits\n"
            + smi_record.get("query_stdout", "")
            + "\n$ nvidia-smi\n"
            + smi_record.get("full_stdout", ""),
            encoding="utf-8",
        )
        pd.DataFrame(history).to_csv(out / "smoke_training_history.csv", index=False)
        fold_frame.to_csv(out / "cv_fold_assignments.csv", index=False)
        save_json(out / "data_access_log.json", access.events)
        save_json(out / "input_manifest.json", _provenance_input_manifest(root, config_file, cfg) + [{
            "role": "source_raw_dataset", **next(
                event for event in access.events if event["event"] == "raw_file_load")
        }])
        result = {
            "status": status, "stage": "B0", "source_batch": 1,
            "implementation_version": cfg["implementation_version"],
            "target_batches_loaded": False, "fold": fold,
            "epochs": 1, "optimization_batches": 1,
            "selected_device": placement.get("selected_device"),
            "model_parameter_device": placement.get("model_parameter_device"),
            "input_tensor_device": placement.get("input_tensor_device"),
            "label_tensor_device": placement.get("label_tensor_device"),
            "first_training_batch_device": placement.get("first_training_batch_device"),
            "python_pid": placement.get("nvidia_smi", {}).get("python_pid"),
            "python_process_name": placement.get("nvidia_smi", {}).get("python_process_name"),
            "python_process_command": placement.get("nvidia_smi", {}).get("python_process_command"),
            "gpu_pid": placement.get("gpu_pid"),
            "gpu_pid_identification": placement.get("nvidia_smi", {}).get(
                "gpu_pid_identification"),
            "python_pid_directly_matched": placement.get("nvidia_smi", {}).get(
                "python_pid_directly_matched"),
            "gpu_pid_visible_through_nvidia_smi": placement.get(
                "gpu_pid_visible_through_nvidia_smi", False),
            "nvidia_smi_matching_process_rows": matching_rows,
            "torch_cuda_memory_allocated_bytes": placement.get("nvidia_smi", {}).get(
                "torch_cuda_memory_allocated_bytes"),
            "torch_cuda_memory_reserved_bytes": placement.get("nvidia_smi", {}).get(
                "torch_cuda_memory_reserved_bytes"),
            "config_sha256": sha256(config_file), "checks": checks,
        }
        save_json(out / "gpu_smoke_result.json", result)
        save_json(out / "run_manifest.json", {
            "run_id": run_id, "created_utc": started, "completed_utc": utc_now(),
            "implementation_version": cfg["implementation_version"],
            "status": status, "kind": "Batch-1-only B0 GPU smoke test",
            "configuration": "configuration.json", "config_sha256": sha256(config_file),
            "implementation_files": _implementation_identity(root) + [{
                "path": "scripts/run_cdcnn_v6_full.py", "sha256": sha256(Path(__file__)),
                "bytes": Path(__file__).stat().st_size,
            }],
            "preprocessing_fit_scope": "Batch 1 fold-1 training rows only",
            "outputs": ["gpu_smoke_result.json", "device_placement_audit.json",
                        "post_training_nvidia_smi.json", "nvidia_smi.txt",
                        "smoke_training_history.csv"],
            "exclusions": [], "raw_data_modified": False,
        })
        (out / "report.md").write_text(
            f"# Batch-1 B0 GPU smoke test\n\nStatus: **{status}**. One B0 optimization "
            f"batch ran on `{device}` using Batch 1 only. Container Python PID "
            f"`{result['python_pid']}` corresponds to the sole `nvidia-smi` compute PID "
            f"`{result['gpu_pid']}`; the IDs differ across the container/host PID namespace. "
            f"GPU process visibility: `{result['gpu_pid_visible_through_nvidia_smi']}`. "
            "No target batch was opened.\n",
            encoding="utf-8",
        )
        _write_output_inventory(out)
        if status != "passed":
            raise ProtocolError(f"B0 GPU smoke checks failed: {checks}")
        return out
    except BaseException as exc:
        if not (out / "data_access_log.json").exists():
            save_json(out / "data_access_log.json", access.events)
        if device_audit and not (out / "device_placement_audit.json").exists():
            save_json(out / "device_placement_audit.json", device_audit)
        save_failure(out / "failure.json", phase="b0_batch1_gpu_smoke", exc=exc)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--config", default="configs/cdcnn_v6.json")
    launch_parser.add_argument("--max-workers", type=int, default=1)
    launch_parser.add_argument("--max-attempts", type=int, default=2)
    launch_parser.add_argument("--gpu-smoke-run", required=True)
    pilot_launch = sub.add_parser("pilot-launch")
    pilot_launch.add_argument("--config", default="configs/cdcnn_v6.json")
    pilot_launch.add_argument("--seed", required=True, type=int, choices=SEEDS)
    pilot_launch.add_argument("--max-workers", type=int, default=1)
    pilot_launch.add_argument("--max-attempts", type=int, default=2)
    pilot_launch.add_argument("--gpu-smoke-run", required=True)
    smoke = sub.add_parser("gpu-smoke")
    smoke.add_argument("--config", default="configs/cdcnn_v6.json")
    worker = sub.add_parser("source-worker")
    worker.add_argument("--stage", required=True, choices=STAGES)
    worker.add_argument("--seed", required=True, type=int, choices=SEEDS)
    worker.add_argument("--config", default="configs/cdcnn_v6.json")
    worker.add_argument("--output", required=True)
    control = sub.add_parser("controller")
    control.add_argument("--run-dir", required=True)
    control.add_argument("--config", default="configs/cdcnn_v6.json")
    control.add_argument("--max-workers", type=int, default=1)
    control.add_argument("--max-attempts", type=int, default=2)
    control.add_argument("--gpu-smoke-run", required=True)
    pilot_control = sub.add_parser("pilot-controller")
    pilot_control.add_argument("--run-dir", required=True)
    pilot_control.add_argument("--config", default="configs/cdcnn_v6.json")
    pilot_control.add_argument("--seed", required=True, type=int, choices=SEEDS)
    pilot_control.add_argument("--max-workers", type=int, default=1)
    pilot_control.add_argument("--max-attempts", type=int, default=2)
    pilot_control.add_argument("--gpu-smoke-run", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run-dir", required=True)
    evaluation.add_argument("--config", default="configs/cdcnn_v6.json")
    evaluation.add_argument("--selections", required=True)
    evaluation.add_argument("--failures")
    args = parser.parse_args()
    if args.command == "launch":
        out, pid = launch(args.config, args.max_workers, args.max_attempts, args.gpu_smoke_run)
        print(json.dumps({"run_dir": str(out), "controller_pid": pid}))
    elif args.command == "pilot-launch":
        out, pid = launch(
            args.config, args.max_workers, args.max_attempts, args.gpu_smoke_run,
            seeds=(args.seed,), report_kind="one_seed_pilot")
        print(json.dumps({"run_dir": str(out), "controller_pid": pid}))
    elif args.command == "gpu-smoke":
        print(run_gpu_smoke(args.config))
    elif args.command == "source-worker":
        print(source_worker(args.stage, args.seed, args.config, args.output))
    elif args.command == "controller":
        print(controller(
            args.run_dir, args.config, args.max_workers, args.max_attempts,
            args.gpu_smoke_run))
    elif args.command == "pilot-controller":
        print(controller(
            args.run_dir, args.config, args.max_workers, args.max_attempts,
            args.gpu_smoke_run, seeds=(args.seed,), report_kind="one_seed_pilot"))
    else:
        print(evaluate(args.run_dir, args.config, args.selections, args.failures))


if __name__ == "__main__":
    main()
