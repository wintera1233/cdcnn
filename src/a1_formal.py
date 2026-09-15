"""Formal A1: canonical Conv1d ResNet plus source-only statistical input augmentation."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.cv_folds import load_folds
from src.pca_analysis import load_batch, save_json, sha256
from src.resnet_1d_baseline import GasResNet1D, parameter_counts, predict_logits, reshape


NO_SCHEDULER = {"name": "no_scheduler", "type": "none"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_config(cfg: dict) -> None:
    required_training = {
        "max_epochs": 30, "patience": 8, "batch_size": 64, "optimizer": "SGD",
        "learning_rate": 0.001, "momentum": 0.9, "weight_decay": 1e-4,
        "scheduler": "none", "device": "cpu",
    }
    for key, expected in required_training.items():
        if cfg["training"].get(key) != expected:
            raise ValueError(f"A1 requires training.{key}={expected!r}")
    required_augmentation = {
        "augmentation_ratio": 1.0, "views_per_original": 1,
        "standard_deviation_ddof": 0, "f": "identity", "target_data_allowed": False,
    }
    for key, expected in required_augmentation.items():
        if cfg["augmentation"].get(key) != expected:
            raise ValueError(f"A1 requires augmentation.{key}={expected!r}")
    perturbation_scale = cfg["augmentation"].get("perturbation_scale", 1.0)
    if not isinstance(perturbation_scale, (int, float)) or not np.isfinite(perturbation_scale):
        raise ValueError("A1 requires a finite numeric augmentation.perturbation_scale")
    if perturbation_scale < 0:
        raise ValueError("A1 requires augmentation.perturbation_scale >= 0")
    if cfg.get("experiment_id") != "A1" or cfg.get("source_batch") != 1:
        raise ValueError("This entry point runs only formal A1 with Batch 1 as source")
    if cfg.get("target_batches") != list(range(2, 11)):
        raise ValueError("Formal A1 target batches must be 2-10")
    if cfg.get("augmentation_seed") != 1042:
        raise ValueError("Formal A1 augmentation seed must be 1042")
    arch = cfg["architecture"]
    if (arch.get("convolution_dimension"), arch.get("input_layout"), arch.get("block_channels"),
            arch.get("pca"), arch.get("reshape_16x8")) != (
                1, [1, 128], [32, 64, 128, 256, 128], False, False):
        raise ValueError("Formal A1 requires the canonical 1D B0 backbone with no PCA or 16x8 reshape")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _sample_id(line_number: int) -> str:
    return f"batch1:line{line_number}"


def generate_fixed_views(
    standardized_x: np.ndarray,
    y: np.ndarray,
    source_indices: np.ndarray,
    line_numbers: np.ndarray,
    context: str,
    seed: int,
    perturbation_scale: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate exactly one fixed A1 view per supplied Batch 1 training sample."""
    if standardized_x.shape != (len(y), 128) or len(source_indices) != len(y):
        raise ValueError("Augmentation inputs have inconsistent shapes")
    rng = np.random.default_rng(seed)
    augmented = np.empty_like(standardized_x, dtype=np.float64)
    records: list[dict] = []
    for anchor_local in range(len(y)):
        candidates = np.flatnonzero(y == y[anchor_local])
        candidates = candidates[candidates != anchor_local]
        if not len(candidates):
            raise ValueError(f"No non-self same-class partner for {context} anchor {anchor_local}")
        partner_local = int(rng.choice(candidates))
        lam = float(rng.uniform(0.0, 1.0))
        anchor = standardized_x[anchor_local]
        partner = standardized_x[partner_local]
        mu_anchor, sd_anchor = float(anchor.mean()), float(anchor.std(ddof=0))
        mu_partner, sd_partner = float(partner.mean()), float(partner.std(ddof=0))
        mu_mix = lam * mu_anchor + (1.0 - lam) * mu_partner
        sd_mix = lam * sd_anchor + (1.0 - lam) * sd_partner
        perturbation = rng.normal(mu_mix, sd_mix, size=128)
        applied_perturbation = perturbation_scale * perturbation
        augmented[anchor_local] = anchor + applied_perturbation
        anchor_source = int(source_indices[anchor_local])
        partner_source = int(source_indices[partner_local])
        label = int(y[anchor_local])
        records.append({
            "context": context,
            "generated_sample_id": f"{context}:aug_of_{_sample_id(int(line_numbers[anchor_source]))}",
            "source_batch": 1,
            "anchor_source_index": anchor_source,
            "anchor_sample_id": _sample_id(int(line_numbers[anchor_source])),
            "anchor_gas_label": label,
            "partner_source_index": partner_source,
            "partner_sample_id": _sample_id(int(line_numbers[partner_source])),
            "partner_gas_label": int(y[partner_local]),
            "generated_gas_label": label,
            "same_class_pair": bool(y[anchor_local] == y[partner_local]),
            "non_self_pair": bool(anchor_source != partner_source),
            "lambda": lam,
            "anchor_mean": mu_anchor,
            "anchor_population_std": sd_anchor,
            "partner_mean": mu_partner,
            "partner_population_std": sd_partner,
            "mixed_mean": mu_mix,
            "mixed_population_std": sd_mix,
            "perturbation_mean": float(perturbation.mean()),
            "perturbation_population_std": float(perturbation.std(ddof=0)),
            "perturbation_l2_norm": float(np.linalg.norm(perturbation)),
            "perturbation_scale": float(perturbation_scale),
            "applied_perturbation_mean": float(applied_perturbation.mean()),
            "applied_perturbation_population_std": float(applied_perturbation.std(ddof=0)),
            "applied_perturbation_l2_norm": float(np.linalg.norm(applied_perturbation)),
            "augmentation_seed": seed,
        })
    return augmented, pd.DataFrame(records)


def matrix_stats(values: np.ndarray) -> dict:
    return {
        "element_count": int(values.size),
        "mean": float(values.mean()),
        "population_std": float(values.std(ddof=0)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean_sample_l2_norm": float(np.linalg.norm(values, axis=1).mean()),
    }


def sanity_for_context(
    context: str,
    original: np.ndarray,
    augmented: np.ndarray,
    y: np.ndarray,
    provenance: pd.DataFrame,
    perturbation_scale: float,
) -> dict:
    perturbations = augmented - original
    label_counts = {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    generated_counts = {
        str(int(k)): int(v) for k, v in provenance.generated_gas_label.value_counts().sort_index().items()
    }
    checks = {
        "generated_count_equals_original_count": len(augmented) == len(original),
        "exactly_one_view_per_anchor": provenance.anchor_sample_id.nunique() == len(original) == len(provenance),
        "all_source_batches_are_1": provenance.source_batch.eq(1).all(),
        "all_pairs_same_class": provenance.same_class_pair.all(),
        "all_pairs_non_self": provenance.non_self_pair.all(),
        "all_generated_labels_preserved": provenance.generated_gas_label.eq(provenance.anchor_gas_label).all(),
        "label_counts_preserved": label_counts == generated_counts,
        "all_values_finite": bool(np.isfinite(augmented).all() and np.isfinite(perturbations).all()),
        "perturbation_scale_recorded": bool(provenance.perturbation_scale.eq(perturbation_scale).all()),
        "applied_l2_equals_scaled_raw_l2": bool(np.allclose(
            provenance.applied_perturbation_l2_norm,
            perturbation_scale * provenance.perturbation_l2_norm,
            rtol=1e-12,
            atol=1e-12,
        )),
    }
    return {
        "context": context,
        "original_count": int(len(original)),
        "generated_view_count": int(len(augmented)),
        "combined_training_count": int(len(original) + len(augmented)),
        "source_label_counts": label_counts,
        "generated_label_counts": generated_counts,
        "perturbation_scale": float(perturbation_scale),
        "checks": {key: bool(value) for key, value in checks.items()},
        "perturbation": {
            **matrix_stats(perturbations),
            "global_l2_norm": float(np.linalg.norm(perturbations)),
            "minimum_view_l2_norm": float(np.linalg.norm(perturbations, axis=1).min()),
            "maximum_view_l2_norm": float(np.linalg.norm(perturbations, axis=1).max()),
        },
        "original_standardized_features": matrix_stats(original),
        "augmented_standardized_features": matrix_stats(augmented),
    }


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> tuple[TensorDataset, DataLoader]:
    dataset = TensorDataset(torch.from_numpy(reshape(x)), torch.from_numpy(y.astype(np.int64) - 1))
    generator = torch.Generator().manual_seed(seed)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_cv_model(x_train, y_train, x_valid, y_valid, training_cfg, seed):
    seed_everything(seed)
    model = GasResNet1D()
    dataset, loader = make_loader(x_train, y_train, training_cfg["batch_size"], seed, training_cfg["shuffle"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=training_cfg["learning_rate"], momentum=training_cfg["momentum"],
        weight_decay=training_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    history, best_state = [], None
    best_epoch, best_accuracy, best_loss, stale = 0, -1.0, math.inf, 0
    for epoch in range(1, training_cfg["max_epochs"] + 1):
        model.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        valid_logits = predict_logits(model, x_valid, training_cfg["batch_size"])
        valid_pred = valid_logits.argmax(1) + 1
        valid_accuracy = float(accuracy_score(y_valid, valid_pred))
        valid_loss = float(criterion(
            torch.from_numpy(valid_logits), torch.from_numpy(y_valid.astype(np.int64) - 1)).item())
        improved = valid_accuracy > best_accuracy or (valid_accuracy == best_accuracy and valid_loss < best_loss)
        if improved:
            best_accuracy, best_loss, best_epoch = valid_accuracy, valid_loss, epoch
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "training_loss": loss_sum / len(dataset),
            "training_accuracy": correct / len(dataset), "validation_loss": valid_loss,
            "validation_accuracy": valid_accuracy, "learning_rate": training_cfg["learning_rate"],
            "is_selected_epoch": False,
        })
        if stale >= training_cfg["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No CV checkpoint was selected")
    model.load_state_dict(best_state)
    for row in history:
        row["is_selected_epoch"] = row["epoch"] == best_epoch
    return model, history, best_epoch, best_accuracy, best_loss


def train_fixed_epochs(x, y, training_cfg, seed, epochs):
    seed_everything(seed)
    model = GasResNet1D()
    dataset, loader = make_loader(x, y, training_cfg["batch_size"], seed, training_cfg["shuffle"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=training_cfg["learning_rate"], momentum=training_cfg["momentum"],
        weight_decay=training_cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, correct = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
        history.append({
            "epoch": epoch, "training_loss": loss_sum / len(dataset),
            "training_accuracy": correct / len(dataset), "learning_rate": training_cfg["learning_rate"],
        })
    return model, history


def predicted_class_proportions(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby(["batch", "predicted_gas_label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(1, 7), fill_value=0)
    rows = []
    for batch, values in counts.iterrows():
        total = int(values.sum())
        for label, count in values.items():
            rows.append({"batch": int(batch), "predicted_gas_label": int(label),
                         "count": int(count), "proportion": float(count / total)})
    return pd.DataFrame(rows)


def confusion_rows(batch: int | str, true: np.ndarray, pred: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(true, pred, labels=range(1, 7))
    return [{"batch": batch, "true_gas_label": i, "predicted_gas_label": j,
             "count": int(matrix[i - 1, j - 1])}
            for i in range(1, 7) for j in range(1, 7)]


def validate_b0_configuration(root: Path, cfg: dict) -> tuple[Path, dict, dict[str, Path]]:
    baseline = (root / cfg["b0_baseline_run"]).resolve()
    paths = {name: baseline / name for name in (
        "configuration.json", "summary.json", "batch_metrics.csv", "cv_fold_results.csv")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing B0 baseline artifacts: {missing}")
    b0_cfg = json.loads(paths["configuration.json"].read_text(encoding="utf-8"))
    expected = cfg["training"]
    for key in ("max_epochs", "patience", "batch_size", "optimizer", "learning_rate", "momentum", "weight_decay"):
        if b0_cfg["training"].get(key) != expected[key]:
            raise ValueError(f"B0 baseline training.{key} does not match formal A1")
    if b0_cfg["architecture"]["convolution_dimension"] != 1 or b0_cfg["architecture"]["input_layout"] != [1, 128]:
        raise ValueError("B0 comparison is not canonical Conv1d")
    if b0_cfg.get("scheduler_candidates"):
        raise ValueError("Fixed B0 reference unexpectedly contains scheduler candidates")
    return baseline, b0_cfg, paths


def load_b0_results(paths: dict[str, Path]) -> tuple[dict, pd.DataFrame]:
    """Load measured B0 results only after the A1 checkpoint has been frozen."""
    return (json.loads(paths["summary.json"].read_text(encoding="utf-8")),
            pd.read_csv(paths["batch_metrics.csv"]))


def validate_previous_a1_configuration(root: Path, cfg: dict) -> tuple[Path, dict, dict[str, Path]]:
    previous = (root / cfg["previous_a1_run"]).resolve()
    paths = {name: previous / name for name in (
        "configuration.json", "summary.json", "batch_metrics.csv", "cv_fold_results.csv",
        "augmentation_provenance.csv.gz")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing previous A1 artifacts: {missing}")
    previous_cfg = json.loads(paths["configuration.json"].read_text(encoding="utf-8"))
    for key in ("source_batch", "target_batches", "model_seed", "augmentation_seed", "saved_cv_folds"):
        if previous_cfg.get(key) != cfg.get(key):
            raise ValueError(f"Previous A1 {key} does not match the requested run")
    for key in ("max_epochs", "patience", "batch_size", "optimizer", "learning_rate", "momentum", "weight_decay", "scheduler"):
        if previous_cfg["training"].get(key) != cfg["training"].get(key):
            raise ValueError(f"Previous A1 training.{key} does not match the requested run")
    for key in ("augmentation_ratio", "views_per_original", "partner_sampling", "lambda_distribution",
                "standard_deviation_ddof", "perturbation_distribution", "f", "target_data_allowed"):
        if previous_cfg["augmentation"].get(key) != cfg["augmentation"].get(key):
            raise ValueError(f"Previous A1 augmentation.{key} does not match the requested run")
    return previous, previous_cfg, paths


def verify_raw_perturbation_reproduction(current: pd.DataFrame, previous_path: Path) -> dict:
    previous = pd.read_csv(previous_path)
    identity_columns = [
        "context", "generated_sample_id", "anchor_source_index", "anchor_sample_id",
        "partner_source_index", "partner_sample_id", "anchor_gas_label", "partner_gas_label",
        "generated_gas_label", "same_class_pair", "non_self_pair", "augmentation_seed",
    ]
    float_columns = [
        "lambda", "anchor_mean", "anchor_population_std", "partner_mean",
        "partner_population_std", "mixed_mean", "mixed_population_std",
        "perturbation_mean", "perturbation_population_std", "perturbation_l2_norm",
    ]
    checks = {
        "row_count_matches_previous_a1": len(current) == len(previous),
        "sampling_and_labels_match_previous_a1": bool(current[identity_columns].equals(previous[identity_columns])),
        "gaussian_draw_statistics_match_previous_a1": bool(np.allclose(
            current[float_columns].to_numpy(), previous[float_columns].to_numpy(), rtol=0.0, atol=1e-14)),
    }
    return {
        "previous_a1_provenance": str(previous_path),
        "numeric_tolerance": {"relative": 0.0, "absolute": 1e-14},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def load_reference_results(paths: dict[str, Path]) -> tuple[dict, pd.DataFrame]:
    return (json.loads(paths["summary.json"].read_text(encoding="utf-8")),
            pd.read_csv(paths["batch_metrics.csv"]))


def build_comparison(summary: dict, metrics: pd.DataFrame, b0_summary: dict, b0_metrics: pd.DataFrame,
                     previous_summary: dict, previous_metrics: pd.DataFrame) -> pd.DataFrame:
    b0_by_batch = b0_metrics.set_index("batch")
    previous_by_batch = previous_metrics.set_index("batch")
    rows = [{"scope": "source_cv_mean", "batch": pd.NA,
             "current_accuracy": summary["source_cv_mean_accuracy"],
             "b0_accuracy": b0_summary["source_cv_mean_accuracy"],
             "previous_a1_accuracy": previous_summary["source_cv_mean_accuracy"]}]
    for row in metrics.itertuples():
        rows.append({"scope": "target_batch", "batch": int(row.batch), "current_accuracy": float(row.accuracy),
                     "b0_accuracy": float(b0_by_batch.loc[row.batch, "accuracy"]),
                     "previous_a1_accuracy": float(previous_by_batch.loc[row.batch, "accuracy"])})
    rows.extend([
        {"scope": "target_unweighted_mean", "batch": pd.NA,
         "current_accuracy": summary["target_unweighted_mean_accuracy"],
         "b0_accuracy": b0_summary["target_unweighted_mean_accuracy"],
         "previous_a1_accuracy": previous_summary["target_unweighted_mean_accuracy"]},
        {"scope": "target_pooled", "batch": pd.NA,
         "current_accuracy": summary["target_pooled_accuracy"],
         "b0_accuracy": b0_summary["target_pooled_accuracy"],
         "previous_a1_accuracy": previous_summary["target_pooled_accuracy"]},
    ])
    result = pd.DataFrame(rows)
    result["current_minus_b0"] = result.current_accuracy - result.b0_accuracy
    result["current_minus_previous_a1"] = result.current_accuracy - result.previous_a1_accuracy
    return result


def plot_curves(cv_history: pd.DataFrame, final_history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fold, values in cv_history.groupby("fold"):
        axes[0].plot(values.epoch, values.training_accuracy, alpha=.55, label=f"Fold {fold} train")
        axes[0].plot(values.epoch, values.validation_accuracy, linestyle="--", alpha=.8, label=f"Fold {fold} validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Batch 1 cross-validation")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    axes[1].plot(final_history.epoch, final_history.training_loss, color="tab:red", label="Training loss")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy", title="Final Batch 1 fit")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_confusions(confusions: pd.DataFrame) -> str:
    sections = []
    for batch in range(2, 11):
        matrix = confusions[confusions.batch.astype(str) == str(batch)].pivot(
            index="true_gas_label", columns="predicted_gas_label", values="count").reindex(
                index=range(1, 7), columns=range(1, 7))
        rows = "\n".join("| " + str(i) + " | " + " | ".join(str(int(v)) for v in matrix.loc[i]) + " |"
                         for i in range(1, 7))
        sections.append(f"### Batch {batch}\n\nRows are true gas labels; columns are predicted gas labels.\n\n"
                        "| True \\ Predicted | 1 | 2 | 3 | 4 | 5 | 6 |\n"
                        "|---:|---:|---:|---:|---:|---:|---:|\n" + rows)
    return "\n\n".join(sections)


def write_sanity_report(out: Path, sanity: dict) -> None:
    rows = []
    for item in sanity["contexts"]:
        rows.append(
            f"| {item['context']} | {item['original_count']} | {item['generated_view_count']} | "
            f"{item['combined_training_count']} | {item['perturbation']['mean']:.6f} | "
            f"{item['perturbation']['population_std']:.6f} | {item['perturbation']['global_l2_norm']:.6f} |")
    text = f"""# A1 pre-training augmentation sanity check

Status: **PASSED**.

Only Batch 1 was loaded. Loaded batches at this checkpoint: `{sanity['loaded_batches_before_training']}`; target batches loaded: `{sanity['target_batches_loaded_before_training']}`. The perturbation scale was `{sanity['perturbation_scale']}`. Raw Gaussian draw statistics, pairing, labels, lambda values, and RNG seeds match the previous A1 provenance exactly within the recorded numeric tolerance.

| Context | Originals | Generated views | Combined | Perturbation mean | Perturbation population SD | Perturbation global L2 norm |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Every anchor has exactly one generated view. Provenance verifies that anchors and partners are Batch 1 training samples, every partner is non-self and from the same gas class, and every generated label equals its anchor label. Detailed label provenance and per-view statistics are in `augmentation_provenance.csv.gz`; original-versus-augmented feature statistics and all checks are in `augmentation_sanity.json` and `augmentation_sanity_by_context.csv`.

This is statistical input augmentation based on the paper's equations. It is not claimed to simulate physical sensor drift.
"""
    (out / "augmentation_sanity_report.md").write_text(text, encoding="utf-8")


def write_report(out: Path, cfg: dict, summary: dict, sanity: dict, cv: pd.DataFrame,
                 metrics: pd.DataFrame, proportions: pd.DataFrame, confusions: pd.DataFrame,
                 comparison: pd.DataFrame, inputs: list[dict], inventory_names: list[str]) -> None:
    cv_rows = "\n".join(
        f"| {int(r.fold)} | {int(r.original_training_samples)} | {int(r.augmented_training_samples)} | "
        f"{int(r.validation_samples)} | {int(r.epochs_run)} | {int(r.selected_epoch)} | {r.accuracy:.6f} |"
        for r in cv.itertuples())
    target_rows = "\n".join(
        f"| {int(r.batch)} | {int(r.samples)} | {int(r.correct)} | {r.accuracy:.6f} |" for r in metrics.itertuples())
    prop_rows = []
    for batch in range(2, 11):
        values = proportions[proportions.batch == batch].sort_values("predicted_gas_label")
        prop_rows.append("| " + str(batch) + " | " + " | ".join(f"{v:.4f}" for v in values.proportion) + " |")
    comparison_rows = []
    for row in comparison.itertuples():
        scope = f"Batch {int(row.batch)}" if pd.notna(row.batch) else row.scope.replace("_", " ").title()
        comparison_rows.append(
            f"| {scope} | {row.current_accuracy:.6f} | {row.b0_accuracy:.6f} | "
            f"{row.current_minus_b0:+.6f} | {row.previous_a1_accuracy:.6f} | "
            f"{row.current_minus_previous_a1:+.6f} |")
    sanity_rows = []
    for item in sanity["contexts"]:
        sanity_rows.append(
            f"| {item['context']} | {item['generated_view_count']} | {item['perturbation']['mean']:.6f} | "
            f"{item['perturbation']['population_std']:.6f} | {item['perturbation']['global_l2_norm']:.6f} | "
            f"{item['original_standardized_features']['mean']:.6f} | {item['original_standardized_features']['population_std']:.6f} | "
            f"{item['augmented_standardized_features']['mean']:.6f} | {item['augmented_standardized_features']['population_std']:.6f} |")
    hash_rows = "\n".join(
        f"| {item['role']} | `{item['path']}` | {item['sha256']} |" for item in inputs)
    config_text = json.dumps(cfg, indent=2, sort_keys=True)
    text = f"""# Formal A1 — canonical 1D ResNet with statistical input augmentation

This run implements and evaluates **A1 only**. A2 latent feature generation and A3 contrastive learning were neither implemented nor run. A1 is statistical input augmentation based on the paper's equations; it is **not** claimed to simulate physical sensor drift. All sampling details not fully specified by the paper are recorded as project assumptions in the resolved configuration.

## Results

Batch 1 mean cross-validation accuracy: **{summary['source_cv_mean_accuracy']:.6f}** (sample SD {summary['source_cv_std_accuracy']:.6f}).

| Fold | Original train | Augmented train | Validation | Epochs run | Selected epoch | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|
{cv_rows}

The round-half-up median selected epoch was **{summary['source_selected_epochs']}**. Each scaler was fit only on the corresponding original Batch 1 training fold; augmentation was applied after scaling. Validation samples were neither augmented nor included in scaler fitting or partner selection.

| Target batch | Samples | Correct | Accuracy |
|---:|---:|---:|---:|
{target_rows}

Unweighted target-batch mean accuracy: **{summary['target_unweighted_mean_accuracy']:.6f}**. Pooled target accuracy: **{summary['target_pooled_accuracy']:.6f}** across {summary['target_total_samples']:,} samples.

## Direct comparison with fixed SGD B0 and previous A1

References: B0 `{summary['b0_baseline_run']}`; previous A1 `{summary['previous_a1_run']}`. Positive deltas favor this scaled A1 run.

| Scope | Scaled A1 | B0 | Scaled A1 - B0 | Previous A1 | Scaled A1 - previous A1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

## Pre-training augmentation sanity check

Status: **PASSED**. Before training, loaded batches were `{sanity['loaded_batches_before_training']}` and target batches loaded were `{sanity['target_batches_loaded_before_training']}`. Every detailed provenance row passed same-class, non-self, Batch-1-source, and label-preservation checks.

| Context | Views | Perturb. mean | Perturb. pop. SD | Perturb. global L2 | Original mean | Original pop. SD | Augmented mean | Augmented pop. SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(sanity_rows)}

Exactly one fixed augmented view was generated for each original training sample in each fold and for final source fitting, using seed 1042. The Gaussian draws, pairs, and lambda values were verified against the previous A1 provenance, then each entire perturbation vector was multiplied by **{cfg['augmentation']['perturbation_scale']}** before addition. Originals and generated views were combined in one shuffled training dataset. Detailed provenance is saved in `augmentation_provenance.csv.gz`.

## Predicted-class proportions

| Batch | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(prop_rows)}

## Confusion matrices

{markdown_confusions(confusions)}

## Configuration and provenance

Model seed: **{cfg['model_seed']}**; fold model seeds: **{', '.join(str(cfg['model_seed'] + f) for f in range(1, 6))}**; augmentation seed: **{cfg['augmentation_seed']}** independently initialized for every fold and the final source fit. PyTorch deterministic algorithms were enabled. The optimizer was SGD (`lr=0.001`, `momentum=0.9`, `weight_decay=1e-4`), batch size 64, maximum 30 epochs, patience 8, with no scheduler.

No PCA and no 2D 16x8 reshape were used. No sample was excluded, deduplicated, clipped, or imputed. Batch 1 alone was used for scaling, cross-validation, augmentation, early stopping, epoch selection, and final fitting. Batches 2-10 were loaded only after the final model checkpoint was frozen and were used once for final evaluation.

### Resolved configuration

```json
{config_text}
```

### Input hashes

| Role | Path | SHA-256 |
|---|---|---|
{hash_rows}

### Output inventory

{chr(10).join('- `' + name + '`' for name in inventory_names)}
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def main(config_path: str = "configs/a1_formal.json") -> Path:
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    validate_config(cfg)
    data_file = (root / cfg["pca_config"]).resolve()
    data_cfg = json.loads(data_file.read_text(encoding="utf-8"))
    spec_file = (root / cfg["specification"]).resolve()
    fold_file = (root / cfg["saved_cv_folds"]).resolve()
    baseline_dir, b0_cfg, b0_paths = validate_b0_configuration(root, cfg)
    previous_dir, previous_cfg, previous_paths = validate_previous_a1_configuration(root, cfg)
    perturbation_scale = float(cfg["augmentation"].get("perturbation_scale", 1.0))
    scale_label = str(perturbation_scale).replace(".", "p")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_formal_a1_scale_{scale_label}"
    out = root / data_cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg["training"]["num_threads"])

    started = utc_now()
    access_log = []
    inputs = []
    implementation_paths = [root / "src/a1_formal.py", root / "src/resnet_1d_baseline.py", root / "scripts/run_a1_formal.py"]
    for role, path in [("a1_configuration", config_file), ("experiment_specification", spec_file),
                       ("dataset_configuration", data_file), ("saved_cv_folds", fold_file)]:
        inputs.append({"role": role, "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                       "sha256": sha256(path)})
    b0_config_path = b0_paths["configuration.json"]
    inputs.append({"role": "fixed_sgd_b0_configuration", "path": str(b0_config_path.relative_to(root)),
                   "bytes": b0_config_path.stat().st_size, "sha256": sha256(b0_config_path),
                   "load_phase": "pre_training_configuration_validation"})
    for name in ("configuration.json", "augmentation_provenance.csv.gz"):
        path = previous_paths[name]
        inputs.append({"role": f"previous_a1_{name.split('.')[0]}", "path": str(path.relative_to(root)),
                       "bytes": path.stat().st_size, "sha256": sha256(path),
                       "load_phase": "pre_training_source_only_reproduction_validation"})

    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["resolved_dataset_configuration"] = data_cfg
    resolved_cfg["resolved_fixed_sgd_b0_configuration"] = b0_cfg
    resolved_cfg["resolved_previous_a1_configuration"] = previous_cfg
    save_json(out / "configuration.json", resolved_cfg)
    save_json(out / "software_versions.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "torch": torch.__version__, "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(), "torch_deterministic_algorithms": True,
    })

    ds_cfg = data_cfg["dataset"]
    source_path = root / ds_cfg["path"] / ds_cfg["batch_file_pattern"].format(batch_id=1)
    source_x, source_y, source_lines = load_batch(source_path, 1, 128)
    access_log.append({"batch": 1, "phase": "pre_training_source_load", "loaded_utc": utc_now(),
                       "purpose": "source validation, folds, scaling, augmentation, model selection, final fitting"})
    inputs.append({"role": "dataset_batch", "batch": 1, "path": str(source_path.relative_to(root)),
                   "bytes": source_path.stat().st_size, "sha256": sha256(source_path), "records": len(source_x),
                   "load_phase": "pre_training"})
    if source_x.shape != (445, 128) or sorted(np.unique(source_y).tolist()) != list(range(1, 7)) or not np.isfinite(source_x).all():
        raise ValueError(f"Batch 1 validation failed: shape={source_x.shape}, labels={np.unique(source_y).tolist()}")
    fold_frame, fold_ids = load_folds(fold_file, source_x, source_lines, source_y, 1)
    fold_frame.to_csv(out / "cv_fold_assignments.csv", index=False)

    shape_model = GasResNet1D()
    dummy = torch.zeros(2, 1, 128)
    with torch.no_grad():
        feature_shape = list(shape_model.forward_features(dummy).shape)
        output_shape = list(shape_model(dummy).shape)
    shape_check = {"input": list(dummy.shape), "backbone_output": feature_shape,
                   "flattened_features_per_sample": int(np.prod(feature_shape[1:])), "output": output_shape}
    expected_shape = {"input": [2, 1, 128], "backbone_output": [2, 128, 128],
                      "flattened_features_per_sample": 16384, "output": [2, 6]}
    if shape_check != expected_shape:
        raise AssertionError(f"Canonical architecture shape check failed: {shape_check}")

    prepared = {}
    provenance_frames = []
    sanity_contexts = []
    for fold in range(1, cfg["validation"]["folds"] + 1):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        if np.intersect1d(train_idx, valid_idx).size or len(np.union1d(train_idx, valid_idx)) != len(source_x):
            raise AssertionError(f"Fold {fold} isolation failed")
        scaler = StandardScaler().fit(source_x[train_idx])
        x_train = scaler.transform(source_x[train_idx])
        x_valid = scaler.transform(source_x[valid_idx])
        augmented, provenance = generate_fixed_views(
            x_train, source_y[train_idx], train_idx, source_lines, f"cv_fold_{fold}",
            cfg["augmentation_seed"], perturbation_scale)
        prepared[fold] = (train_idx, valid_idx, scaler, x_train, x_valid, augmented)
        provenance_frames.append(provenance)
        sanity_contexts.append(sanity_for_context(
            f"cv_fold_{fold}", x_train, augmented, source_y[train_idx], provenance, perturbation_scale))

    final_scaler = StandardScaler().fit(source_x)
    final_original = final_scaler.transform(source_x)
    final_augmented, final_provenance = generate_fixed_views(
        final_original, source_y, np.arange(len(source_x)), source_lines, "final_batch1_fit",
        cfg["augmentation_seed"], perturbation_scale)
    provenance_frames.append(final_provenance)
    sanity_contexts.append(sanity_for_context(
        "final_batch1_fit", final_original, final_augmented, source_y, final_provenance, perturbation_scale))
    provenance_df = pd.concat(provenance_frames, ignore_index=True)
    reproduction = verify_raw_perturbation_reproduction(
        provenance_df, previous_paths["augmentation_provenance.csv.gz"])
    provenance_df.to_csv(out / "augmentation_provenance.csv.gz", index=False, compression="gzip")
    label_counts = provenance_df.groupby(["context", "anchor_gas_label", "partner_gas_label", "generated_gas_label"]).size().rename("count").reset_index()
    label_counts.to_csv(out / "augmentation_label_provenance_counts.csv", index=False)

    target_loaded = sorted(set(item["batch"] for item in access_log) & set(cfg["target_batches"]))
    all_sanity_checks = (
        all(all(item["checks"].values()) for item in sanity_contexts)
        and reproduction["all_checks_passed"]
    )
    sanity = {
        "status": "passed" if all_sanity_checks and not target_loaded else "failed",
        "checked_utc": utc_now(), "loaded_batches_before_training": sorted(set(item["batch"] for item in access_log)),
        "target_batches_loaded_before_training": target_loaded,
        "confirmation_no_target_batch_loaded": not target_loaded,
        "target_derived_result_artifacts_loaded_before_training": False,
        "augmentation_seed": cfg["augmentation_seed"], "views_per_original": 1,
        "perturbation_scale": perturbation_scale,
        "raw_perturbation_reproduction": reproduction,
        "total_cv_generated_views": int(sum(item["generated_view_count"] for item in sanity_contexts[:-1])),
        "final_fit_generated_views": int(sanity_contexts[-1]["generated_view_count"]),
        "all_context_checks_passed": all_sanity_checks, "contexts": sanity_contexts,
    }
    save_json(out / "augmentation_sanity.json", sanity)
    flat_sanity = []
    for item in sanity_contexts:
        flat_sanity.append({
            "context": item["context"], "original_count": item["original_count"],
            "generated_view_count": item["generated_view_count"], "combined_training_count": item["combined_training_count"],
            "perturbation_mean": item["perturbation"]["mean"],
            "perturbation_population_std": item["perturbation"]["population_std"],
            "perturbation_global_l2_norm": item["perturbation"]["global_l2_norm"],
            "mean_perturbation_view_l2_norm": item["perturbation"]["mean_sample_l2_norm"],
            "original_feature_mean": item["original_standardized_features"]["mean"],
            "original_feature_population_std": item["original_standardized_features"]["population_std"],
            "augmented_feature_mean": item["augmented_standardized_features"]["mean"],
            "augmented_feature_population_std": item["augmented_standardized_features"]["population_std"],
            **item["checks"],
        })
    pd.DataFrame(flat_sanity).to_csv(out / "augmentation_sanity_by_context.csv", index=False)
    write_sanity_report(out, sanity)
    save_json(out / "data_access_log.json", access_log)
    if sanity["status"] != "passed":
        raise RuntimeError(f"Pre-training augmentation sanity check failed: {sanity}")

    cv_rows, history_rows, cv_predictions, cv_confusion = [], [], [], []
    for fold in range(1, cfg["validation"]["folds"] + 1):
        train_idx, valid_idx, scaler, x_train, x_valid, augmented = prepared[fold]
        combined_x = np.concatenate([x_train, augmented], axis=0)
        combined_y = np.concatenate([source_y[train_idx], source_y[train_idx]], axis=0)
        model, history, best_epoch, best_accuracy, best_loss = train_cv_model(
            combined_x, combined_y, x_valid, source_y[valid_idx], cfg["training"], cfg["model_seed"] + fold)
        logits = predict_logits(model, x_valid, cfg["training"]["batch_size"])
        pred = logits.argmax(1) + 1
        measured = float(accuracy_score(source_y[valid_idx], pred))
        if measured != best_accuracy:
            raise AssertionError(f"Fold {fold} restored checkpoint accuracy mismatch")
        cv_rows.append({
            "fold": fold, "original_training_samples": len(train_idx), "augmented_training_samples": len(augmented),
            "combined_training_samples": len(combined_x), "validation_samples": len(valid_idx), "overlap_samples": 0,
            "scaler_fit_batch": 1, "scaler_fit_samples": len(train_idx), "augmentation_source_batch": 1,
            "augmentation_seed": cfg["augmentation_seed"], "model_seed": cfg["model_seed"] + fold,
            "perturbation_scale": perturbation_scale,
            "epochs_run": len(history), "selected_epoch": best_epoch,
            "stopped_early": len(history) < cfg["training"]["max_epochs"], "accuracy": measured,
            "selected_validation_loss": best_loss,
        })
        history_rows.extend({"fold": fold, **row} for row in history)
        cv_predictions.append(pd.DataFrame({
            "sample_id": [_sample_id(int(source_lines[i])) for i in valid_idx], "batch": 1,
            "line_number": source_lines[valid_idx], "cv_fold": fold, "true_gas_label": source_y[valid_idx],
            "predicted_gas_label": pred, "correct": pred == source_y[valid_idx],
        }))
        cv_confusion.extend(confusion_rows(f"cv_fold_{fold}", source_y[valid_idx], pred))
    cv_df = pd.DataFrame(cv_rows)
    history_df = pd.DataFrame(history_rows)
    cv_prediction_df = pd.concat(cv_predictions, ignore_index=True)
    cv_df.to_csv(out / "cv_fold_results.csv", index=False)
    history_df.to_csv(out / "cv_training_history.csv", index=False)
    cv_prediction_df.to_csv(out / "cv_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cv_confusion).to_csv(out / "cv_confusion_matrices.csv", index=False)
    selected_epochs = int(math.floor(float(cv_df.selected_epoch.median()) + 0.5))

    final_combined_x = np.concatenate([final_original, final_augmented], axis=0)
    final_combined_y = np.concatenate([source_y, source_y], axis=0)
    final_model, final_history = train_fixed_epochs(
        final_combined_x, final_combined_y, cfg["training"], cfg["model_seed"], selected_epochs)
    checkpoint_path = out / "model.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(), "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_, "architecture": cfg["architecture"],
        "experiment_id": "A1", "selected_epochs": selected_epochs,
        "augmentation": cfg["augmentation"], "augmentation_seed": cfg["augmentation_seed"],
        "optimizer": {"name": "SGD", "learning_rate": .001, "momentum": .9, "weight_decay": 1e-4},
        "scheduler": NO_SCHEDULER,
    }, checkpoint_path)
    final_history_df = pd.DataFrame(final_history)
    final_history_df.to_csv(out / "final_training_history.csv", index=False)
    frozen_utc = utc_now()
    save_json(out / "model_freeze_manifest.json", {
        "frozen_utc": frozen_utc, "checkpoint": "model.pt", "checkpoint_sha256": sha256(checkpoint_path),
        "selected_epochs": selected_epochs, "selection_scope": "Batch 1 five-fold cross-validation only",
        "batches_loaded_at_freeze": sorted(set(item["batch"] for item in access_log)),
        "target_batches_loaded_at_freeze": sorted(set(item["batch"] for item in access_log) & set(cfg["target_batches"])),
    })

    b0_summary, b0_metrics = load_b0_results(b0_paths)
    for name in ("summary.json", "batch_metrics.csv", "cv_fold_results.csv"):
        path = b0_paths[name]
        inputs.append({"role": f"fixed_sgd_b0_{name.removesuffix('.json').removesuffix('.csv')}",
                       "path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path),
                       "load_phase": "post_model_freeze_comparison"})
    previous_summary, previous_metrics = load_reference_results(previous_paths)
    for name in ("summary.json", "batch_metrics.csv", "cv_fold_results.csv"):
        path = previous_paths[name]
        inputs.append({"role": f"previous_a1_{name.removesuffix('.json').removesuffix('.csv')}",
                       "path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path),
                       "load_phase": "post_model_freeze_comparison"})

    arrays, labels, lines = {1: source_x}, {1: source_y}, {1: source_lines}
    for batch in cfg["target_batches"]:
        path = root / ds_cfg["path"] / ds_cfg["batch_file_pattern"].format(batch_id=batch)
        arrays[batch], labels[batch], lines[batch] = load_batch(path, batch, 128)
        access_log.append({"batch": batch, "phase": "post_freeze_target_evaluation_load", "loaded_utc": utc_now(),
                           "purpose": "final frozen-model evaluation only", "model_frozen_utc": frozen_utc})
        inputs.append({"role": "dataset_batch", "batch": batch, "path": str(path.relative_to(root)),
                       "bytes": path.stat().st_size, "sha256": sha256(path), "records": len(arrays[batch]),
                       "load_phase": "post_model_freeze"})
    save_json(out / "data_access_log.json", access_log)

    all_x = np.vstack([arrays[b] for b in range(1, 11)])
    all_y = np.concatenate([labels[b] for b in range(1, 11)])
    validation = {
        "records": int(len(all_x)), "batches": list(range(1, 11)), "batch_record_counts": {str(b): int(len(arrays[b])) for b in range(1, 11)},
        "gas_labels": sorted(map(int, np.unique(all_y))), "features": int(all_x.shape[1]),
        "finite_values": bool(np.isfinite(all_x).all()), "nonfinite_value_count": int((~np.isfinite(all_x)).sum()),
        "expected_structure_verified": bool(len(all_x) == 13910 and all_x.shape[1] == 128 and len(np.unique(all_y)) == 6),
        "exclusions": [], "deduplication": False, "clipping": False, "imputation": False,
    }
    if not validation["expected_structure_verified"] or not validation["finite_values"]:
        raise ValueError(f"Full dataset validation failed after model freeze: {validation}")
    save_json(out / "dataset_validation.json", validation)

    predictions, metrics, target_confusion = [], [], []
    for batch in cfg["target_batches"]:
        scaled = final_scaler.transform(arrays[batch])
        pred = predict_logits(final_model, scaled, cfg["training"]["batch_size"]).argmax(1) + 1
        correct = pred == labels[batch]
        metrics.append({"batch": batch, "samples": len(pred), "correct": int(correct.sum()), "accuracy": float(correct.mean())})
        predictions.append(pd.DataFrame({
            "sample_id": [f"batch{batch}:line{int(v)}" for v in lines[batch]], "batch": batch,
            "line_number": lines[batch], "true_gas_label": labels[batch], "predicted_gas_label": pred,
            "correct": correct,
        }))
        target_confusion.extend(confusion_rows(batch, labels[batch], pred))
    metric_df = pd.DataFrame(metrics)
    prediction_df = pd.concat(predictions, ignore_index=True)
    confusion_df = pd.DataFrame(target_confusion)
    proportions_df = predicted_class_proportions(prediction_df)
    metric_df.to_csv(out / "batch_metrics.csv", index=False)
    prediction_df.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    confusion_df.to_csv(out / "confusion_matrices.csv", index=False)
    proportions_df.to_csv(out / "class_prediction_proportions.csv", index=False)

    summary = {
        "experiment_id": "A1", "implemented_experiments": ["A1"], "not_implemented_or_run": ["A2", "A3"],
        "description": "Statistical input augmentation based on the paper's equations; not a physical sensor-drift simulation.",
        "canonical_backbone": "1D ResNet B0", "legacy_2d_reshape_used": False, "pca_used": False,
        "parameter_counts": parameter_counts(shape_model), "tensor_shape_check": shape_check,
        "source_cv_mean_accuracy": float(cv_df.accuracy.mean()),
        "source_cv_std_accuracy": float(cv_df.accuracy.std(ddof=1)), "source_selected_epochs": selected_epochs,
        "target_unweighted_mean_accuracy": float(metric_df.accuracy.mean()),
        "target_pooled_accuracy": float(prediction_df.correct.mean()), "target_total_samples": int(len(prediction_df)),
        "b0_baseline_run": str(baseline_dir.relative_to(root)),
        "previous_a1_run": str(previous_dir.relative_to(root)),
        "perturbation_scale": perturbation_scale,
        "raw_perturbation_reproduction_status": "passed" if reproduction["all_checks_passed"] else "failed",
        "augmentation_sanity_status": sanity["status"], "target_batches_loaded_before_training": target_loaded,
        "optimizer": {"name": "SGD", "learning_rate": .001, "momentum": .9, "weight_decay": 1e-4},
        "scheduler": "none", "seeds": {"model_final": cfg["model_seed"],
            "model_cv_by_fold": {str(f): cfg["model_seed"] + f for f in range(1, 6)},
            "augmentation_each_context": cfg["augmentation_seed"]},
    }
    comparison_df = build_comparison(
        summary, metric_df, b0_summary, b0_metrics, previous_summary, previous_metrics)
    comparison_df.to_csv(out / "reference_comparison.csv", index=False)
    comparison_df[["scope", "batch", "current_accuracy", "b0_accuracy", "current_minus_b0"]].rename(
        columns={"current_accuracy": "a1_accuracy", "current_minus_b0": "a1_minus_b0"}
    ).to_csv(out / "b0_comparison.csv", index=False)
    save_json(out / "summary.json", summary)
    save_json(out / "input_manifest.json", inputs)
    plot_curves(history_df, final_history_df, out / "training_validation_curves.png")

    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = None
    implementation = [{"path": str(path.relative_to(root)), "sha256": sha256(path), "bytes": path.stat().st_size}
                      for path in implementation_paths]
    save_json(out / "run_manifest.json", {
        "run_id": run_id, "experiment_id": "A1", "created_utc": started, "completed_utc": utc_now(),
        "command": f"{sys.executable} scripts/run_a1_formal.py --config {config_path}", "git_revision": revision,
        "working_directory": str(root), "implementation_files": implementation,
        "preprocessing_fit_scope": "CV scalers: relevant original Batch 1 training fold only; final scaler: all original Batch 1 only.",
        "augmentation_scope": "One fixed view per original Batch 1 training sample; no validation or target samples.",
        "perturbation_scale": perturbation_scale,
        "selection_scope": "Batch 1 cross-validation only; target batches loaded after final checkpoint freeze.",
        "sample_policy": "All valid records retained; no removal, deduplication, clipping, or imputation.",
        "exclusions": [], "raw_data_modified": False, "a2_implemented_or_run": False, "a3_implemented_or_run": False,
        "summary": summary,
    })

    expected_inventory = sorted([
        "augmentation_label_provenance_counts.csv", "augmentation_provenance.csv.gz", "augmentation_sanity.json",
        "augmentation_sanity_by_context.csv", "augmentation_sanity_report.md", "b0_comparison.csv", "batch_metrics.csv",
        "class_prediction_proportions.csv", "configuration.json", "confusion_matrices.csv", "cv_confusion_matrices.csv",
        "cv_fold_assignments.csv", "cv_fold_results.csv", "cv_predictions.csv.gz", "cv_training_history.csv",
        "data_access_log.json", "dataset_validation.json", "final_training_history.csv", "input_manifest.json",
        "model.pt", "model_freeze_manifest.json", "output_inventory.json", "predictions.csv.gz", "report.md",
        "reference_comparison.csv", "run_manifest.json", "software_versions.json", "summary.json", "training_validation_curves.png",
    ])
    write_report(out, resolved_cfg, summary, sanity, cv_df, metric_df, proportions_df, confusion_df,
                 comparison_df, inputs, expected_inventory)
    actual_before_inventory = sorted(path.name for path in out.iterdir())
    expected_before_inventory = [name for name in expected_inventory if name != "output_inventory.json"]
    if actual_before_inventory != expected_before_inventory:
        raise AssertionError({"expected": expected_before_inventory, "actual": actual_before_inventory})
    inventory = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(out.iterdir())]
    inventory.append({"path": "output_inventory.json", "bytes": None, "sha256": None,
                      "note": "Self-entry; size and digest are null to avoid recursive self-hashing."})
    save_json(out / "output_inventory.json", inventory)
    if sorted(path.name for path in out.iterdir()) != expected_inventory:
        raise AssertionError("Final output inventory does not match actual outputs")

    # Make the completed run read-only so subsequent runs cannot overwrite it accidentally.
    for path in out.iterdir():
        path.chmod(0o444)
    out.chmod(0o555)
    return out


if __name__ == "__main__":
    print(main())
