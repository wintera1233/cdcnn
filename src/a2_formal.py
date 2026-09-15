"""Formal A2 source-only experiment for the canonical Conv1d CDCNN ablation."""

from __future__ import annotations

import copy
import hashlib
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.a1_formal import (
    generate_fixed_views, sanity_for_context, verify_raw_perturbation_reproduction,
)
from src.cv_folds import load_folds
from src.pca_analysis import load_batch, save_json, sha256
from src.resnet_1d_baseline import ResidualBlock1D, reshape


NO_SCHEDULER = {"name": "no_scheduler", "type": "none"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GateFailure(RuntimeError):
    """A conceptual or tensor-shape gate failure that forbids training."""


class A2GasResNet1D(nn.Module):
    """Canonical five-block backbone with A2 generation after block 3."""

    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        channels = (32, 64, 128, 256, 128)
        blocks, incoming = [], 1
        for outgoing in channels:
            blocks.append(ResidualBlock1D(incoming, outgoing))
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)
        self.flatten = nn.Flatten()
        self.fc128 = nn.Sequential(nn.Linear(128 * 128, 128), nn.BatchNorm1d(128))
        self.fc6 = nn.Linear(128, 6)
        self.epsilon = float(epsilon)

    def forward_to_block3(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks[:3]:
            x = block(x)
        return x

    def forward_shared_tail(self, z: torch.Tensor) -> torch.Tensor:
        for block in self.blocks[3:]:
            z = block(z)
        return self.fc6(self.fc128(self.flatten(z)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Original inference path; generation is disabled."""
        return self.forward_shared_tail(self.forward_to_block3(x))

    def decompose(self, z_s: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z_l = F.interpolate(
            F.max_pool1d(z_s, kernel_size=2, stride=2, padding=0),
            size=z_s.shape[-1], mode="nearest",
        )
        z_h = z_s - z_l
        mu_l = z_l.mean(dim=-1, keepdim=True)
        sigma_l = z_l.std(dim=-1, unbiased=False, keepdim=True) + self.epsilon
        return z_l, z_h, mu_l, sigma_l

    def generate_latent(
        self, z_s: torch.Tensor, generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z_l, z_h, mu_l, sigma_l = self.decompose(z_s)
        detached_mu = mu_l.detach()
        detached_log_sigma = sigma_l.detach().log()
        mu_center = detached_mu.mean(dim=0, keepdim=True)
        mu_spread = detached_mu.std(dim=0, unbiased=False, keepdim=True)
        log_center = detached_log_sigma.mean(dim=0, keepdim=True)
        log_spread = detached_log_sigma.std(dim=0, unbiased=False, keepdim=True)
        noise_mu = torch.randn(detached_mu.shape, generator=generator, dtype=z_s.dtype, device=z_s.device)
        noise_log = torch.randn(detached_log_sigma.shape, generator=generator, dtype=z_s.dtype, device=z_s.device)
        generated_mu = mu_center + mu_spread * noise_mu
        generated_log_sigma = log_center + log_spread * noise_log
        generated_sigma = generated_log_sigma.exp()
        generated_low = generated_sigma * ((z_l - mu_l) / sigma_l) + generated_mu
        generated = z_h + generated_low
        details = {
            "z_l": z_l, "z_h": z_h, "mu_l": mu_l, "sigma_l": sigma_l,
            "generated_mu": generated_mu, "generated_log_sigma": generated_log_sigma,
            "generated_sigma": generated_sigma, "generated_low": generated_low,
        }
        return generated, details

    def forward_training(
        self, x: torch.Tensor, generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        z_s = self.forward_to_block3(x)
        generated, details = self.generate_latent(z_s, generator)
        # One call through the same tail/head objects also gives BatchNorm a
        # symmetric combined branch batch.
        logits = self.forward_shared_tail(torch.cat([z_s, generated], dim=0))
        original_logits, generated_logits = logits.split(len(x), dim=0)
        details["z_s"] = z_s
        details["generated"] = generated
        return original_logits, generated_logits, details


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def parameter_counts(model: A2GasResNet1D) -> dict:
    count = lambda module: sum(p.numel() for p in module.parameters())
    result = {
        "resnet_blocks_1_to_3": sum(count(block) for block in model.blocks[:3]),
        "shared_resnet_blocks_4_and_5": sum(count(block) for block in model.blocks[3:]),
        "FC128_including_batch_norm": count(model.fc128),
        "FC6": count(model.fc6),
        "latent_generator_trainable": 0,
        "total": count(model),
    }
    subtotal = sum(result[k] for k in (
        "resnet_blocks_1_to_3", "shared_resnet_blocks_4_and_5",
        "FC128_including_batch_norm", "FC6", "latent_generator_trainable"))
    if subtotal != result["total"]:
        raise AssertionError("Parameter accounting mismatch")
    return result


def validate_config(cfg: dict) -> None:
    expected_training = {
        "max_epochs": 30, "patience": 8, "batch_size": 64,
        "optimizer": "SGD", "learning_rate": 0.001, "momentum": 0.9,
        "weight_decay": 1e-4, "scheduler": "none", "device": "cpu",
    }
    checks = {k: cfg["training"].get(k) == v for k, v in expected_training.items()}
    if not all(checks.values()):
        raise GateFailure(f"Required SGD configuration failed: {checks}")
    if cfg.get("experiment_id") != "A2" or cfg.get("source_batch") != 1:
        raise GateFailure("A2 must use Batch 1 as its source")
    if cfg.get("target_batches") != list(range(2, 11)):
        raise GateFailure("A2 targets must be Batches 2-10")
    arch = cfg["architecture"]
    arch_tuple = (
        arch.get("convolution_dimension"), arch.get("input_layout"),
        arch.get("block_channels"), arch.get("pca"), arch.get("reshape_16x8"),
    )
    if arch_tuple != (1, [1, 128], [32, 64, 128, 256, 128], False, False):
        raise GateFailure("A2 configuration is not the canonical Conv1d architecture")
    aug = cfg["augmentation"]
    if aug.get("perturbation_scale") != 0.5 or aug.get("target_data_allowed") is not False:
        raise GateFailure("A2 must inherit fixed A1 perturbation_scale=0.5 source-only augmentation")
    latent = cfg["latent_feature_generation"]
    required_latent = {
        "ratio": 1.0, "views_per_active_training_latent": 1,
        "standard_deviation_unbiased": False, "epsilon": 1e-5,
        "mse_loss": False, "contrastive_loss": False, "target_data_allowed": False,
    }
    latent_checks = {k: latent.get(k) == v for k, v in required_latent.items()}
    if not all(latent_checks.values()):
        raise GateFailure(f"A2 latent configuration failed: {latent_checks}")


def _artifact(root: Path, path: Path, role: str, phase: str, **extra) -> dict:
    return {
        "role": role, "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
        "sha256": sha256(path), "load_phase": phase, **extra,
    }


def _sample_id(line_number: int) -> str:
    return f"batch1:line{line_number}"


def make_loader(
    x: np.ndarray, y: np.ndarray, input_types: np.ndarray,
    batch_size: int, seed: int, shuffle: bool,
) -> tuple[TensorDataset, DataLoader]:
    type_codes = np.asarray([0 if value == "original" else 1 for value in input_types], dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(reshape(x)), torch.from_numpy(y.astype(np.int64) - 1),
        torch.from_numpy(type_codes),
    )
    generator = torch.Generator().manual_seed(seed)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def predict_logits(model: A2GasResNet1D, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        chunks.append(model(torch.from_numpy(reshape(x[start:start + batch_size]))).cpu())
    return torch.cat(chunks).numpy()


def _stats_accumulator() -> dict:
    return {
        "generated_sigma_min": math.inf, "generated_sigma_max": -math.inf,
        "generated_mu_min": math.inf, "generated_mu_max": -math.inf,
        "nonfinite_tensor_count": 0,
    }


def _update_stats(acc: dict, details: dict[str, torch.Tensor]) -> None:
    sigma = details["generated_sigma"].detach()
    mu = details["generated_mu"].detach()
    acc["generated_sigma_min"] = min(acc["generated_sigma_min"], float(sigma.min()))
    acc["generated_sigma_max"] = max(acc["generated_sigma_max"], float(sigma.max()))
    acc["generated_mu_min"] = min(acc["generated_mu_min"], float(mu.min()))
    acc["generated_mu_max"] = max(acc["generated_mu_max"], float(mu.max()))
    acc["nonfinite_tensor_count"] += sum(
        0 if torch.isfinite(value).all() else 1 for value in details.values())


def train_cv_model(
    x_train: np.ndarray, y_train: np.ndarray, input_types: np.ndarray,
    x_valid: np.ndarray, y_valid: np.ndarray, cfg: dict, seed: int,
    latent_seed: int,
) -> tuple[A2GasResNet1D, list[dict], int, float, float]:
    seed_everything(seed)
    model = A2GasResNet1D(cfg["latent_feature_generation"]["epsilon"])
    dataset, loader = make_loader(
        x_train, y_train, input_types, cfg["training"]["batch_size"], seed,
        cfg["training"]["shuffle"],
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"], weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()
    latent_generator = torch.Generator().manual_seed(latent_seed)
    history, best_state = [], None
    best_epoch, best_accuracy, best_loss, stale = 0, -1.0, math.inf, 0
    for epoch in range(1, cfg["training"]["max_epochs"] + 1):
        model.train()
        loss_sum = original_loss_sum = generated_loss_sum = 0.0
        original_correct = generated_correct = original_seen = augmented_seen = 0
        stats = _stats_accumulator()
        for xb, yb, type_code in loader:
            optimizer.zero_grad(set_to_none=True)
            original_logits, generated_logits, details = model.forward_training(xb, latent_generator)
            original_loss = criterion(original_logits, yb)
            generated_loss = criterion(generated_logits, yb)
            loss = 0.5 * (original_loss + generated_loss)
            loss.backward()
            optimizer.step()
            n = len(yb)
            loss_sum += float(loss.item()) * n
            original_loss_sum += float(original_loss.item()) * n
            generated_loss_sum += float(generated_loss.item()) * n
            original_correct += int((original_logits.argmax(1) == yb).sum())
            generated_correct += int((generated_logits.argmax(1) == yb).sum())
            original_seen += int((type_code == 0).sum())
            augmented_seen += int((type_code == 1).sum())
            _update_stats(stats, details)
        valid_logits = predict_logits(model, x_valid, cfg["training"]["batch_size"])
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
            "original_branch_loss": original_loss_sum / len(dataset),
            "generated_branch_loss": generated_loss_sum / len(dataset),
            "original_branch_accuracy": original_correct / len(dataset),
            "generated_branch_accuracy": generated_correct / len(dataset),
            "validation_loss": valid_loss, "validation_accuracy": valid_accuracy,
            "learning_rate": cfg["training"]["learning_rate"], "scheduler_step_applied": False,
            "a1_original_inputs_seen": original_seen, "a1_augmented_inputs_seen": augmented_seen,
            "original_branch_views": len(dataset), "generated_branch_views": len(dataset),
            "equal_branch_weight": 0.5, **stats, "is_selected_epoch": False,
        })
        if stale >= cfg["training"]["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No A2 CV checkpoint was selected")
    model.load_state_dict(best_state)
    for row in history:
        row["is_selected_epoch"] = row["epoch"] == best_epoch
    return model, history, best_epoch, best_accuracy, best_loss


def train_fixed_epochs(
    x: np.ndarray, y: np.ndarray, input_types: np.ndarray, cfg: dict,
    seed: int, latent_seed: int, epochs: int,
) -> tuple[A2GasResNet1D, list[dict]]:
    seed_everything(seed)
    model = A2GasResNet1D(cfg["latent_feature_generation"]["epsilon"])
    dataset, loader = make_loader(
        x, y, input_types, cfg["training"]["batch_size"], seed, cfg["training"]["shuffle"])
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg["training"]["learning_rate"],
        momentum=cfg["training"]["momentum"], weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    latent_generator = torch.Generator().manual_seed(latent_seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = original_loss_sum = generated_loss_sum = 0.0
        original_correct = generated_correct = original_seen = augmented_seen = 0
        stats = _stats_accumulator()
        for xb, yb, type_code in loader:
            optimizer.zero_grad(set_to_none=True)
            original_logits, generated_logits, details = model.forward_training(xb, latent_generator)
            original_loss = criterion(original_logits, yb)
            generated_loss = criterion(generated_logits, yb)
            loss = 0.5 * (original_loss + generated_loss)
            loss.backward()
            optimizer.step()
            n = len(yb)
            loss_sum += float(loss.item()) * n
            original_loss_sum += float(original_loss.item()) * n
            generated_loss_sum += float(generated_loss.item()) * n
            original_correct += int((original_logits.argmax(1) == yb).sum())
            generated_correct += int((generated_logits.argmax(1) == yb).sum())
            original_seen += int((type_code == 0).sum())
            augmented_seen += int((type_code == 1).sum())
            _update_stats(stats, details)
        history.append({
            "epoch": epoch, "training_loss": loss_sum / len(dataset),
            "original_branch_loss": original_loss_sum / len(dataset),
            "generated_branch_loss": generated_loss_sum / len(dataset),
            "original_branch_accuracy": original_correct / len(dataset),
            "generated_branch_accuracy": generated_correct / len(dataset),
            "learning_rate": cfg["training"]["learning_rate"], "scheduler_step_applied": False,
            "a1_original_inputs_seen": original_seen, "a1_augmented_inputs_seen": augmented_seen,
            "original_branch_views": len(dataset), "generated_branch_views": len(dataset),
            "equal_branch_weight": 0.5, **stats,
        })
    return model, history


def confusion_rows(scope: int | str, true: np.ndarray, pred: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(true, pred, labels=range(1, 7))
    return [{
        "batch": scope, "true_gas_label": true_label,
        "predicted_gas_label": predicted_label,
        "count": int(matrix[true_label - 1, predicted_label - 1]),
    } for true_label in range(1, 7) for predicted_label in range(1, 7)]


def predicted_class_proportions(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby(["batch", "predicted_gas_label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(1, 7), fill_value=0)
    rows = []
    for batch, values in counts.iterrows():
        total = int(values.sum())
        for label, count in values.items():
            rows.append({
                "batch": int(batch), "predicted_gas_label": int(label),
                "count": int(count), "proportion": float(count / total),
            })
    return pd.DataFrame(rows)


def validate_reference_configurations(
    root: Path, cfg: dict,
) -> tuple[Path, dict, Path, dict, dict[str, Path], dict[str, Path]]:
    b0_dir = (root / cfg["b0_baseline_run"]).resolve()
    a1_dir = (root / cfg["a1_parent_run"]).resolve()
    b0_names = ("configuration.json", "summary.json", "batch_metrics.csv", "cv_fold_results.csv")
    a1_names = (
        "configuration.json", "summary.json", "batch_metrics.csv", "cv_fold_results.csv",
        "augmentation_provenance.csv.gz", "dataset_validation.json", "input_manifest.json",
    )
    b0_paths = {name: b0_dir / name for name in b0_names}
    a1_paths = {name: a1_dir / name for name in a1_names}
    missing = [str(path) for path in [*b0_paths.values(), *a1_paths.values()] if not path.is_file()]
    if missing:
        raise GateFailure(f"Reference artifacts are missing: {missing}")
    b0_cfg = json.loads(b0_paths["configuration.json"].read_text(encoding="utf-8"))
    a1_cfg = json.loads(a1_paths["configuration.json"].read_text(encoding="utf-8"))
    if b0_cfg.get("experiment_id") != "B0" or a1_cfg.get("experiment_id") != "A1":
        raise GateFailure("Reference experiment identifiers are not B0 and A1")
    if a1_cfg.get("augmentation", {}).get("perturbation_scale") != 0.5:
        raise GateFailure("The A1 parent does not use perturbation_scale=0.5")
    for reference_name, reference in (("B0", b0_cfg), ("A1", a1_cfg)):
        training = reference["training"]
        expected = cfg["training"]
        for key in ("max_epochs", "patience", "batch_size", "optimizer", "learning_rate", "momentum", "weight_decay"):
            if training.get(key) != expected[key]:
                raise GateFailure(f"{reference_name} training.{key} differs from A2")
        scheduler = training.get("scheduler", "none")
        if scheduler != "none":
            raise GateFailure(f"{reference_name} is not a no-scheduler reference")
        arch = reference["architecture"]
        if arch.get("convolution_dimension") != 1 or arch.get("input_layout") != [1, 128]:
            raise GateFailure(f"{reference_name} is not the canonical Conv1d model")
    if a1_cfg.get("saved_cv_folds") != cfg.get("saved_cv_folds"):
        raise GateFailure("A1 and A2 fold-assignment paths differ")
    return b0_dir, b0_cfg, a1_dir, a1_cfg, b0_paths, a1_paths


def run_preflight(
    root: Path, cfg: dict, source_x: np.ndarray, source_y: np.ndarray,
    source_lines: np.ndarray, fold_file: Path, a1_paths: dict[str, Path],
    b0_cfg: dict, a1_cfg: dict, access_log: list[dict],
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    fold_frame, fold_ids = load_folds(fold_file, source_x, source_lines, source_y, 1)
    prior_validation = json.loads(a1_paths["dataset_validation.json"].read_text(encoding="utf-8"))
    expected_counts = {"1": 445, "2": 1244, "3": 1586, "4": 161, "5": 197,
                       "6": 2300, "7": 3613, "8": 294, "9": 470, "10": 3600}
    model = A2GasResNet1D(cfg["latent_feature_generation"]["epsilon"])
    dummy = torch.zeros(2, 1, 128)
    with torch.no_grad():
        block3 = model.forward_to_block3(dummy)
        final_features = block3
        for block in model.blocks[3:]:
            final_features = block(final_features)
        output = model(dummy)
    conv2d_count = sum(isinstance(module, nn.Conv2d) for module in model.modules())
    training_expected = {
        "learning_rate": 0.001, "momentum": 0.9, "weight_decay": 1e-4,
        "batch_size": 64, "max_epochs": 30, "patience": 8, "scheduler": "none",
    }
    training_checks = {key: cfg["training"].get(key) == value for key, value in training_expected.items()}
    target_raw_accesses = [row for row in access_log if row["batch"] in cfg["target_batches"]]
    checks = {
        "full_dataset_has_13910_records_from_saved_validation": prior_validation.get("records") == 13910,
        "full_dataset_has_10_batches_from_saved_validation": prior_validation.get("batches") == list(range(1, 11)),
        "full_dataset_batch_counts_match_from_saved_validation": prior_validation.get("batch_record_counts") == expected_counts,
        "full_dataset_has_6_labels_from_saved_validation": prior_validation.get("gas_labels") == list(range(1, 7)),
        "full_dataset_has_128_features_from_saved_validation": prior_validation.get("features") == 128,
        "full_dataset_is_finite_from_saved_validation": prior_validation.get("finite_values") is True,
        "batch1_direct_shape": source_x.shape == (445, 128),
        "batch1_direct_has_6_labels": np.array_equal(np.unique(source_y), np.arange(1, 7)),
        "batch1_direct_is_finite": bool(np.isfinite(source_x).all()),
        "folds_cover_every_batch1_sample_once": len(fold_frame) == len(source_x) and not fold_frame.sample_id.duplicated().any(),
        "fold_ids_are_1_to_5": np.array_equal(np.unique(fold_ids), np.arange(1, 6)),
        "fold_validation_sets_are_non_overlapping": sum(int((fold_ids == fold).sum()) for fold in range(1, 6)) == len(source_x),
        "canonical_input_shape_is_B_1_128": list(dummy.shape) == [2, 1, 128],
        "block3_output_shape_is_B_128_128": list(block3.shape) == [2, 128, 128],
        "block5_output_shape_is_B_128_128": list(final_features.shape) == [2, 128, 128],
        "classifier_output_shape_is_B_6": list(output.shape) == [2, 6],
        "no_conv2d_modules": conv2d_count == 0,
        "no_16x8_reshape_configured": cfg["architecture"].get("reshape_16x8") is False,
        "b0_configuration_loaded": b0_cfg.get("experiment_id") == "B0",
        "a1_configuration_loaded": a1_cfg.get("experiment_id") == "A1",
        "a1_parent_perturbation_scale_is_0p5": a1_cfg["augmentation"].get("perturbation_scale") == 0.5,
        "all_required_sgd_settings_match": all(training_checks.values()),
        "no_target_raw_files_accessed": not target_raw_accesses,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checked_utc": utc_now(),
        "full_dataset_validation_evidence": {
            "artifact": str(a1_paths["dataset_validation.json"].relative_to(root)),
            "artifact_sha256": sha256(a1_paths["dataset_validation.json"]),
            "reason": "Uses the completed A1 structural validation so raw target files remain untouched before A2 checkpoint freeze.",
        },
        "directly_loaded_batches": sorted({row["batch"] for row in access_log}),
        "target_raw_files_accessed": bool(target_raw_accesses),
        "source_shape": list(source_x.shape), "source_labels": np.unique(source_y).astype(int).tolist(),
        "fold_counts": {str(fold): int((fold_ids == fold).sum()) for fold in range(1, 6)},
        "tensor_shapes": {
            "input": list(dummy.shape), "after_block3": list(block3.shape),
            "after_block5": list(final_features.shape), "classifier_output": list(output.shape),
        },
        "legacy_module_audit": {"conv2d_module_count": conv2d_count, "reshape_16x8": False},
        "training_settings": {**training_expected, "optimizer": cfg["training"]["optimizer"]},
        "training_setting_checks": training_checks,
        "checks": {key: bool(value) for key, value in checks.items()},
    }
    if result["status"] != "passed":
        raise GateFailure(f"A2 preflight failed: {[k for k, v in checks.items() if not v]}")
    return result, fold_frame, fold_ids


def latent_sanity(
    cfg: dict, sample_x: np.ndarray, sample_y: np.ndarray, access_log: list[dict],
) -> dict:
    seed_everything(cfg["model_seed"])
    model = A2GasResNet1D(cfg["latent_feature_generation"]["epsilon"])
    model.train()
    xb = torch.from_numpy(reshape(sample_x[:8]))
    yb = torch.from_numpy(sample_y[:8].astype(np.int64) - 1)
    latent_generator = torch.Generator().manual_seed(cfg["latent_feature_seed"])
    z_s = model.forward_to_block3(xb)
    generated, details = model.generate_latent(z_s, latent_generator)
    z_l, z_h = details["z_l"], details["z_h"]
    mu_l, sigma_l = details["mu_l"], details["sigma_l"]
    reconstruction = z_l + z_h
    identity_low = sigma_l * ((z_l - mu_l) / sigma_l) + mu_l
    identity_reconstruction = z_h + identity_low
    original_logits, generated_logits = model.forward_shared_tail(
        torch.cat([z_s, generated], dim=0)).split(len(xb), dim=0)
    loss = 0.5 * (
        F.cross_entropy(original_logits, yb) + F.cross_entropy(generated_logits, yb))
    model.zero_grad(set_to_none=True)
    loss.backward()
    block_gradients = {}
    for block_index, block in enumerate(model.blocks, 1):
        grads = [parameter.grad for parameter in block.parameters()]
        block_gradients[str(block_index)] = {
            "all_present": all(gradient is not None for gradient in grads),
            "all_finite": all(gradient is not None and torch.isfinite(gradient).all() for gradient in grads),
            "total_absolute_gradient": float(sum(
                gradient.detach().abs().sum() for gradient in grads if gradient is not None)),
        }
    shared_original_blocks = [model.blocks[3], model.blocks[4]]
    shared_generated_blocks = [model.blocks[3], model.blocks[4]]
    original_parameter_ids = [[id(parameter) for parameter in block.parameters()] for block in shared_original_blocks]
    generated_parameter_ids = [[id(parameter) for parameter in block.parameters()] for block in shared_generated_blocks]
    finite_tensors = {
        name: bool(torch.isfinite(value).all())
        for name, value in {"z_s": z_s, "generated": generated, **details}.items()
    }
    target_accesses = [row for row in access_log if row["batch"] in cfg["target_batches"]]
    reconstruction_max_error = float((reconstruction - z_s).detach().abs().max())
    identity_max_error = float((identity_reconstruction - z_s).detach().abs().max())
    checks = {
        "z_s_shape": list(z_s.shape) == [8, 128, 128],
        "z_l_shape": list(z_l.shape) == [8, 128, 128],
        "z_h_shape": list(z_h.shape) == [8, 128, 128],
        "generated_shape": list(generated.shape) == [8, 128, 128],
        "low_plus_high_reconstructs": bool(torch.allclose(reconstruction, z_s, rtol=1e-5, atol=1e-5)),
        "mu_l_shape": list(mu_l.shape) == [8, 128, 1],
        "sigma_l_shape": list(sigma_l.shape) == [8, 128, 1],
        "sigma_uses_unbiased_false": cfg["latent_feature_generation"]["standard_deviation_unbiased"] is False,
        "sigma_epsilon_is_1e_5": model.epsilon == 1e-5,
        "all_tensors_finite": all(finite_tensors.values()),
        "identity_statistics_reconstruct": bool(torch.allclose(identity_reconstruction, z_s, rtol=1e-5, atol=1e-5)),
        "generated_standard_deviations_positive": bool((details["generated_sigma"] > 0).all()),
        "sampled_statistics_are_detached": not details["generated_mu"].requires_grad
        and not details["generated_sigma"].requires_grad,
        "block4_object_shared": shared_original_blocks[0] is shared_generated_blocks[0],
        "block5_object_shared": shared_original_blocks[1] is shared_generated_blocks[1],
        "block4_and_5_parameter_ids_identical": original_parameter_ids == generated_parameter_ids,
        "gradients_reach_blocks_1_to_3": all(block_gradients[str(i)]["total_absolute_gradient"] > 0 for i in range(1, 4)),
        "gradients_reach_shared_blocks_4_and_5": all(block_gradients[str(i)]["total_absolute_gradient"] > 0 for i in (4, 5)),
        "all_backbone_gradients_finite": all(value["all_finite"] for value in block_gradients.values()),
        "no_target_batch_files_accessed": not target_accesses,
        "loss_has_no_mse_or_contrastive_component": not cfg["latent_feature_generation"]["mse_loss"] and not cfg["latent_feature_generation"]["contrastive_loss"],
    }
    return {
        "status": "passed" if all(checks.values()) else "failed", "checked_utc": utc_now(),
        "source_batches_used": [1], "target_batch_files_accessed": bool(target_accesses),
        "input_shape": list(xb.shape),
        "tensor_shapes": {name: list(value.shape) for name, value in {
            "z_s": z_s, "z_L": z_l, "z_H": z_h, "mu_L": mu_l,
            "sigma_L": sigma_l, "generated_mu": details["generated_mu"],
            "generated_sigma": details["generated_sigma"], "generated_z": generated,
        }.items()},
        "numeric_tolerance": {"relative": 1e-5, "absolute": 1e-5},
        "low_plus_high_max_absolute_error": reconstruction_max_error,
        "identity_statistics_max_absolute_error": identity_max_error,
        "sigma": {"unbiased": False, "epsilon": model.epsilon,
                  "minimum": float(sigma_l.detach().min()), "maximum": float(sigma_l.detach().max())},
        "generated_sigma": {"minimum": float(details["generated_sigma"].detach().min()),
                            "maximum": float(details["generated_sigma"].detach().max())},
        "finite_tensors": finite_tensors,
        "sampling_rule": cfg["latent_feature_generation"]["sampling_rule"],
        "sampled_statistics_require_grad": {
            "generated_mu": details["generated_mu"].requires_grad,
            "generated_sigma": details["generated_sigma"].requires_grad,
        },
        "parameter_sharing": {
            "block4_module_object_id_matches": checks["block4_object_shared"],
            "block5_module_object_id_matches": checks["block5_object_shared"],
            "parameter_object_ids_match": checks["block4_and_5_parameter_ids_identical"],
        },
        "gradient_audit": block_gradients,
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def plot_curves(cv_history: pd.DataFrame, final_history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for fold, values in cv_history.groupby("fold"):
        axes[0].plot(values.epoch, values.original_branch_accuracy, alpha=.55,
                     label=f"Fold {fold} original branch")
        axes[0].plot(values.epoch, values.generated_branch_accuracy, alpha=.4, linestyle=":",
                     label=f"Fold {fold} generated branch")
        axes[0].plot(values.epoch, values.validation_accuracy, alpha=.8, linestyle="--",
                     label=f"Fold {fold} validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Batch 1 source-only cross-validation")
    axes[0].legend(fontsize=6, ncol=3, frameon=False)
    axes[1].plot(final_history.epoch, final_history.original_branch_loss,
                 label="Original branch CE")
    axes[1].plot(final_history.epoch, final_history.generated_branch_loss,
                 label="Generated branch CE")
    axes[1].plot(final_history.epoch, final_history.training_loss,
                 color="black", linestyle="--", label="Equal-weight mean CE")
    axes[1].set(xlabel="Epoch", ylabel="Cross-entropy", title="Final all-Batch-1 fit")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_comparison(
    summary: dict, target_metrics: pd.DataFrame,
    b0_summary: dict, b0_metrics: pd.DataFrame,
    a1_summary: dict, a1_metrics: pd.DataFrame,
) -> pd.DataFrame:
    b0_by_batch = b0_metrics.set_index("batch")
    a1_by_batch = a1_metrics.set_index("batch")
    rows = [{
        "scope": "source_cv_mean", "batch": pd.NA,
        "b0_accuracy": b0_summary["source_cv_mean_accuracy"],
        "a1_accuracy": a1_summary["source_cv_mean_accuracy"],
        "a2_accuracy": summary["source_cv_mean_accuracy"],
    }]
    for row in target_metrics.itertuples():
        rows.append({
            "scope": "target_batch", "batch": int(row.batch),
            "b0_accuracy": float(b0_by_batch.loc[row.batch, "accuracy"]),
            "a1_accuracy": float(a1_by_batch.loc[row.batch, "accuracy"]),
            "a2_accuracy": float(row.accuracy),
        })
    rows.extend([
        {"scope": "target_unweighted_mean", "batch": pd.NA,
         "b0_accuracy": b0_summary["target_unweighted_mean_accuracy"],
         "a1_accuracy": a1_summary["target_unweighted_mean_accuracy"],
         "a2_accuracy": summary["target_unweighted_mean_accuracy"]},
        {"scope": "target_pooled", "batch": pd.NA,
         "b0_accuracy": b0_summary["target_pooled_accuracy"],
         "a1_accuracy": a1_summary["target_pooled_accuracy"],
         "a2_accuracy": summary["target_pooled_accuracy"]},
    ])
    result = pd.DataFrame(rows)
    result["a1_minus_b0"] = result.a1_accuracy - result.b0_accuracy
    result["a2_minus_a1"] = result.a2_accuracy - result.a1_accuracy
    result["a2_minus_b0"] = result.a2_accuracy - result.b0_accuracy
    return result


def recompute_from_saved_predictions(out: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv_predictions = pd.read_csv(out / "cv_predictions.csv.gz")
    target_predictions = pd.read_csv(out / "predictions.csv.gz")
    fold_metrics = cv_predictions.groupby("cv_fold").correct.agg(["count", "sum", "mean"]).reset_index()
    fold_metrics.columns = ["fold", "samples", "correct", "accuracy"]
    fold_metrics["fold"] = fold_metrics.fold.astype(int)
    fold_metrics["samples"] = fold_metrics.samples.astype(int)
    fold_metrics["correct"] = fold_metrics.correct.astype(int)
    target_metrics = target_predictions.groupby("batch").correct.agg(["count", "sum", "mean"]).reset_index()
    target_metrics.columns = ["batch", "samples", "correct", "accuracy"]
    target_metrics[["batch", "samples", "correct"]] = target_metrics[["batch", "samples", "correct"]].astype(int)
    confusions = pd.DataFrame([
        record for batch, values in target_predictions.groupby("batch", sort=True)
        for record in confusion_rows(int(batch), values.true_gas_label.to_numpy(),
                                     values.predicted_gas_label.to_numpy())
    ])
    proportions = predicted_class_proportions(target_predictions)
    summary = {
        "source_cv_mean_accuracy": float(fold_metrics.accuracy.mean()),
        "source_cv_std_accuracy": float(fold_metrics.accuracy.std(ddof=1)),
        "target_unweighted_mean_accuracy": float(target_metrics.accuracy.mean()),
        "target_pooled_accuracy": float(target_predictions.correct.mean()),
        "target_total_samples": int(len(target_predictions)),
    }
    return summary, fold_metrics, target_metrics, confusions, proportions


def replay_checkpoint(
    checkpoint_path: Path, cfg: dict, arrays: dict[int, np.ndarray],
    primary_predictions: pd.DataFrame,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    replay_model = A2GasResNet1D(checkpoint["latent_feature_generation"]["epsilon"])
    replay_model.load_state_dict(checkpoint["model_state_dict"])
    replay_scaler = StandardScaler()
    replay_scaler.mean_ = np.asarray(checkpoint["scaler_mean"], dtype=np.float64)
    replay_scaler.scale_ = np.asarray(checkpoint["scaler_scale"], dtype=np.float64)
    replay_scaler.var_ = replay_scaler.scale_ ** 2
    replay_scaler.n_features_in_ = 128
    replay_scaler.n_samples_seen_ = int(checkpoint["scaler_fit_samples"])
    predicted = []
    for batch in cfg["target_batches"]:
        scaled = replay_scaler.transform(arrays[batch])
        pred = predict_logits(replay_model, scaled, cfg["training"]["batch_size"]).argmax(1) + 1
        batch_lines = primary_predictions.loc[primary_predictions.batch == batch, "line_number"].to_numpy()
        predicted.append(pd.DataFrame({"batch": batch, "line_number": batch_lines,
                                       "replayed_prediction": pred}))
    replayed = pd.concat(predicted, ignore_index=True)
    primary = primary_predictions[["batch", "line_number", "predicted_gas_label"]].reset_index(drop=True)
    identities_match = np.array_equal(primary[["batch", "line_number"]].to_numpy(),
                                      replayed[["batch", "line_number"]].to_numpy())
    predictions_match = np.array_equal(primary.predicted_gas_label.to_numpy(),
                                       replayed.replayed_prediction.to_numpy())
    replay_correct = replayed.replayed_prediction.to_numpy() == primary_predictions.true_gas_label.to_numpy()
    batch_accuracies = {
        str(batch): float(replay_correct[replayed.batch.to_numpy() == batch].mean())
        for batch in cfg["target_batches"]
    }
    return {
        "status": "passed" if identities_match and predictions_match else "failed",
        "checkpoint": checkpoint_path.name, "checkpoint_sha256": sha256(checkpoint_path),
        "raw_inputs_reused_in_memory": True, "additional_target_file_loads": 0,
        "generation_disabled": True, "inference_path": "original branch only",
        "sample_identity_order_matches": bool(identities_match),
        "all_predictions_match": bool(predictions_match),
        "prediction_mismatch_count": int((primary.predicted_gas_label.to_numpy() != replayed.replayed_prediction.to_numpy()).sum()),
        "batch_accuracies": batch_accuracies,
        "target_unweighted_mean_accuracy": float(np.mean(list(batch_accuracies.values()))),
        "target_pooled_accuracy": float(replay_correct.mean()),
    }


def markdown_confusions(confusions: pd.DataFrame) -> str:
    sections = []
    for batch in range(2, 11):
        values = confusions[confusions.batch == batch].pivot(
            index="true_gas_label", columns="predicted_gas_label", values="count").reindex(
                index=range(1, 7), columns=range(1, 7))
        rows = "\n".join(
            "| " + str(label) + " | " + " | ".join(str(int(value)) for value in values.loc[label]) + " |"
            for label in range(1, 7))
        sections.append(
            f"### Batch {batch}\n\nRows are true labels; columns are predictions.\n\n"
            "| True / predicted | 1 | 2 | 3 | 4 | 5 | 6 |\n"
            "|---:|---:|---:|---:|---:|---:|---:|\n" + rows)
    return "\n\n".join(sections)


def write_report(
    out: Path, cfg: dict, summary: dict, preflight: dict, sanity: dict,
    cv: pd.DataFrame, metrics: pd.DataFrame, proportions: pd.DataFrame,
    confusions: pd.DataFrame, comparison: pd.DataFrame, audit: dict,
    inputs: list[dict], inventory_names: list[str],
) -> None:
    cv_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.original_training_samples)} | {int(row.a1_augmented_training_samples)} | "
        f"{int(row.original_branch_views_per_epoch)} | {int(row.generated_branch_views_per_epoch)} | "
        f"{int(row.selected_epoch)} | {row.accuracy:.6f} |"
        for row in cv.itertuples())
    target_rows = "\n".join(
        f"| {int(row.batch)} | {int(row.samples)} | {int(row.correct)} | {row.accuracy:.6f} |"
        for row in metrics.itertuples())
    comparison_rows = []
    for row in comparison.itertuples():
        label = f"Batch {int(row.batch)}" if pd.notna(row.batch) else row.scope.replace("_", " ").title()
        comparison_rows.append(
            f"| {label} | {row.b0_accuracy:.6f} | {row.a1_accuracy:.6f} | {row.a2_accuracy:.6f} | "
            f"{row.a1_minus_b0:+.6f} | {row.a2_minus_a1:+.6f} |")
    prop_rows = []
    for batch in range(2, 11):
        values = proportions[proportions.batch == batch].sort_values("predicted_gas_label")
        prop_rows.append("| " + str(batch) + " | " + " | ".join(f"{value:.4f}" for value in values.proportion) + " |")
    hash_rows = "\n".join(
        f"| {item['role']} | `{item['path']}` | {item['sha256']} | {item['load_phase']} |"
        for item in inputs)
    text = f"""# A2 — A1 input augmentation plus latent feature generation

Status: **COMPLETED**. This run implements A2 only. A3 was neither implemented nor run.

## Results

Batch-1 CV accuracy is `{summary['source_cv_mean_accuracy']:.6f} ± {summary['source_cv_std_accuracy']:.6f}` (sample standard deviation across five saved folds). The unweighted mean of target-batch accuracies is `{summary['target_unweighted_mean_accuracy']:.6f}` and pooled target accuracy is `{summary['target_pooled_accuracy']:.6f}`.

| Fold | Original source inputs | Fixed A1 views | Original latent branch / epoch | Generated latent branch / epoch | Selected epoch | Validation accuracy |
|---:|---:|---:|---:|---:|---:|---:|
{cv_rows}

| Target batch | Samples | Correct | Accuracy |
|---:|---:|---:|---:|
{target_rows}

## Canonical ablation comparison

| Scope | Canonical SGD B0 | A1 scale 0.5 | A2 | A1 - B0 | A2 - A1 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

The A1-minus-B0 column measures the fixed input-augmentation ablation difference; A2-minus-A1 measures the latent-feature-generation ablation difference under the documented implementation. These measured distribution-generalization differences do not establish a physical mechanism of sensor drift.

## Method and validation

The canonical input is `[B,1,128]`; the block-3 tensor, `z_L`, `z_H`, and generated tensor are `[B,128,128]`. `z_L` uses `MaxPool1d(kernel_size=2,stride=2,padding=0)` and nearest interpolation to length 128. `sigma_L` is the population standard deviation over length plus epsilon `1e-5`. The original and generated tensors pass through the exact same ResNet4 and ResNet5 objects. Their mean cross-entropies are averaged with weights 0.5 and 0.5. No MSE or contrastive term is present.

Preflight status: `{preflight['status']}`. Latent sanity status: `{sanity['status']}`. Low/high maximum reconstruction error was `{sanity['low_plus_high_max_absolute_error']:.3e}`; identity-statistics maximum reconstruction error was `{sanity['identity_statistics_max_absolute_error']:.3e}`. No target raw file was accessed before the model was frozen.

Each scaler was fit only on original samples from the active Batch-1 training fold; the final scaler was fit on all 445 original Batch-1 samples. Fixed A1 scale-0.5 views were generated only for active training inputs. Every optimization encounter generated exactly one latent view for each original-branch training input. Validation and target inference disabled both A1 augmentation and latent generation and used only the original path. Early stopping and final epoch selection used only Batch-1 validation accuracy, with cross-entropy as tie-breaker. Batches 2–10 were loaded once after checkpoint freeze and received one scored target evaluation; the required replay reused those raw arrays in memory.

## Paper-supported details versus project assumptions

Paper-supported details are the five residual blocks; kernel-3 two-convolution blocks with 1x1 shortcuts; the feature-generation location between blocks 3 and 4; an input-normalization stage; FC128, BatchNorm, and FC6; SGD; source Batch 1 and target Batches 2–10; and the operation-level latent feature manipulation described in Eqs. (8)–(16).

Project assumptions are StandardScaler; the Conv1d `[1,128]` interpretation; channel widths; no backbone pooling; flattening to 16,384; momentum 0.9; fixed learning rate 0.001 without scheduler; all A1 sampling details and scale 0.5; the 1D max-pool/interpolation geometry; epsilon and reduction axes; detached active-mini-batch per-channel Gaussian estimation using `mu_L` and `log(sigma_L)`; independent channel sampling; exponentiated log-sigma; stochastic per-optimization latent views; branch concatenation through the shared tail/head; equal 0.5 branch-loss weights; and the median selected-epoch rule. Treating low-frequency statistics as domain/drift style and high-frequency structure as gas identity is an unproven assumption. The generator is not claimed to physically reproduce sensor drift.

## Audit

Saved-prediction recomputation, parameter sharing, finite tensors and gradients, fold coverage, scaler scope, generated-view counts, target timing, frozen-checkpoint replay, and report/CSV/summary agreement all passed: `{audit['all_checks_passed']}`. Replay prediction mismatches: `{audit['checkpoint_replay']['prediction_mismatch_count']}`. Exclusions: none. There was no deduplication, clipping, imputation, PCA, Conv2d, or 16x8 reshape.

## Predicted-class proportions

| Batch | Class 1 | Class 2 | Class 3 | Class 4 | Class 5 | Class 6 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(prop_rows)}

## Confusion matrices

{markdown_confusions(confusions)}

## Input identities

| Role | Path | SHA-256 | Load phase |
|---|---|---|---|
{hash_rows}

## Output inventory

{chr(10).join('- `' + name + '`' for name in inventory_names)}
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def _ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _make_training_input_provenance(
    context: str, indices: np.ndarray, lines: np.ndarray, labels: np.ndarray,
    augmentation: pd.DataFrame,
) -> pd.DataFrame:
    originals = pd.DataFrame({
        "context": context, "training_input_position": np.arange(len(indices)),
        "input_view_type": "original", "source_batch": 1,
        "anchor_source_index": indices, "anchor_sample_id": [_sample_id(int(lines[i])) for i in indices],
        "gas_label": labels[indices], "a1_generated_sample_id": pd.NA,
        "latent_views_per_optimization_encounter": 1,
    })
    augmented = pd.DataFrame({
        "context": context, "training_input_position": np.arange(len(indices), 2 * len(indices)),
        "input_view_type": "a1_fixed_augmented", "source_batch": 1,
        "anchor_source_index": augmentation.anchor_source_index.to_numpy(),
        "anchor_sample_id": augmentation.anchor_sample_id.to_numpy(),
        "gas_label": augmentation.generated_gas_label.to_numpy(),
        "a1_generated_sample_id": augmentation.generated_sample_id.to_numpy(),
        "latent_views_per_optimization_encounter": 1,
    })
    return pd.concat([originals, augmented], ignore_index=True)


def _failure_finalize(out: Path, started: str, exc: Exception) -> None:
    diagnosis = {
        "status": "failed", "failed_utc": utc_now(), "exception_type": type(exc).__name__,
        "diagnosis": str(exc), "traceback": traceback.format_exc(),
        "training_was_not_forced_past_failed_gate": isinstance(exc, GateFailure),
        "method_changed": False, "a3_implemented_or_run": False,
    }
    save_json(out / "failure.json", diagnosis)
    (out / "failure_report.md").write_text(
        "# A2 failure report\n\n"
        f"Status: **FAILED** at `{diagnosis['failed_utc']}`.\n\n"
        f"Exception: `{diagnosis['exception_type']}`\n\n"
        f"Exact diagnosis: {diagnosis['diagnosis']}\n\n"
        "No method substitution was made. A3 was not implemented or run. All partial artifacts in this run were preserved.\n",
        encoding="utf-8")
    if not (out / "run_manifest.json").exists():
        save_json(out / "run_manifest.json", {
            "run_id": out.name, "experiment_id": "A2", "status": "failed",
            "created_utc": started, "failed_utc": diagnosis["failed_utc"],
            "raw_data_modified": False, "a3_implemented_or_run": False,
        })
    files = sorted(path for path in out.iterdir() if path.name != "output_inventory.json")
    inventory = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
    inventory.append({"path": "output_inventory.json", "bytes": None, "sha256": None,
                      "note": "Self-entry; size and digest omitted to avoid recursive self-hashing."})
    save_json(out / "output_inventory.json", inventory)


def _execute(root: Path, config_file: Path, cfg: dict, out: Path, started: str) -> None:
    validate_config(cfg)
    torch.set_num_threads(cfg["training"]["num_threads"])
    data_file = (root / cfg["pca_config"]).resolve()
    data_cfg = json.loads(data_file.read_text(encoding="utf-8"))
    spec_file = (root / cfg["specification"]).resolve()
    fold_file = (root / cfg["saved_cv_folds"]).resolve()
    b0_dir, b0_cfg, a1_dir, a1_cfg, b0_paths, a1_paths = validate_reference_configurations(root, cfg)
    reference_initial_hashes = {
        "b0_configuration": sha256(b0_paths["configuration.json"]),
        "a1_configuration": sha256(a1_paths["configuration.json"]),
    }
    resolved_cfg = copy.deepcopy(cfg)
    resolved_cfg["resolved_dataset_configuration"] = data_cfg
    resolved_cfg["resolved_b0_configuration"] = b0_cfg
    resolved_cfg["resolved_a1_parent_configuration"] = a1_cfg
    save_json(out / "configuration.json", resolved_cfg)
    save_json(out / "software_versions.json", {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "torch": torch.__version__, "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_available": torch.cuda.is_available(), "device": "cpu",
    })
    save_json(out / "implementation_fix_log.json", {
        "prior_failed_run": "runs/20260908T183933746912Z_formal_a2",
        "issue": "The completed first run failed only because DataFrame.equals required bit-exact floating-point round trips for batch_metrics.csv.",
        "fix": "Audit integer identity/count columns exactly and serialized accuracy values with zero relative tolerance and 1e-15 absolute tolerance.",
        "method_or_configuration_changed": False,
        "sanity_gates_rerun": True,
    })
    inputs = [
        _artifact(root, config_file, "a2_configuration", "preflight"),
        _artifact(root, spec_file, "experiment_specification", "preflight"),
        _artifact(root, data_file, "dataset_configuration", "preflight"),
        _artifact(root, fold_file, "saved_cv_folds", "preflight"),
        _artifact(root, b0_paths["configuration.json"], "canonical_sgd_b0_configuration", "preflight"),
        _artifact(root, a1_paths["configuration.json"], "a1_scale_0p5_parent_configuration", "preflight"),
        _artifact(root, a1_paths["dataset_validation.json"], "prior_full_dataset_validation_evidence", "preflight"),
        _artifact(root, a1_paths["augmentation_provenance.csv.gz"], "a1_parent_augmentation_provenance", "source_only_sanity"),
    ]
    implementation_paths = [root / "src/a2_formal.py", root / "scripts/run_a2_formal.py"]

    access_log: list[dict] = []
    ds_cfg = data_cfg["dataset"]
    source_path = root / ds_cfg["path"] / ds_cfg["batch_file_pattern"].format(batch_id=1)
    source_x, source_y, source_lines = load_batch(source_path, 1, 128)
    access_log.append({
        "batch": 1, "file": str(source_path.relative_to(root)), "access_number_for_file": 1,
        "phase": "pre_training_source_load", "loaded_utc": utc_now(),
        "purpose": "preflight, source-only sanity, CV, and final source fit",
    })
    inputs.append(_artifact(root, source_path, "dataset_batch", "pre_training", batch=1, records=len(source_x)))
    save_json(out / "data_access_log.json", access_log)

    preflight, fold_frame, fold_ids = run_preflight(
        root, cfg, source_x, source_y, source_lines, fold_file, a1_paths,
        b0_cfg, a1_cfg, access_log)
    fold_frame.to_csv(out / "cv_fold_assignments.csv", index=False)
    save_json(out / "preflight_validation.json", preflight)

    prepared: dict[int, dict] = {}
    augmentation_frames, augmentation_sanity_contexts = [], []
    training_provenance, scaler_scopes = [], []
    for fold in range(1, cfg["validation"]["folds"] + 1):
        train_idx = np.flatnonzero(fold_ids != fold)
        valid_idx = np.flatnonzero(fold_ids == fold)
        if np.intersect1d(train_idx, valid_idx).size or len(np.union1d(train_idx, valid_idx)) != len(source_x):
            raise GateFailure(f"Fold {fold} train/validation isolation failed")
        scaler = StandardScaler().fit(source_x[train_idx])
        x_train = scaler.transform(source_x[train_idx])
        x_valid = scaler.transform(source_x[valid_idx])
        context = f"cv_fold_{fold}"
        augmented, provenance = generate_fixed_views(
            x_train, source_y[train_idx], train_idx, source_lines, context,
            cfg["augmentation_seed"], cfg["augmentation"]["perturbation_scale"])
        prepared[fold] = {
            "train_idx": train_idx, "valid_idx": valid_idx, "scaler": scaler,
            "x_train": x_train, "x_valid": x_valid, "augmented": augmented,
        }
        augmentation_frames.append(provenance)
        augmentation_sanity_contexts.append(sanity_for_context(
            context, x_train, augmented, source_y[train_idx], provenance,
            cfg["augmentation"]["perturbation_scale"]))
        training_provenance.append(_make_training_input_provenance(
            context, train_idx, source_lines, source_y, provenance))
        fit_ids = [_sample_id(int(source_lines[index])) for index in train_idx]
        scaler_scopes.append({
            "context": context, "fit_batch": 1, "fit_original_samples": len(train_idx),
            "fit_augmented_samples": 0, "fit_validation_samples": 0, "fit_target_samples": 0,
            "fit_sample_ids_sha256": _ids_digest(fit_ids),
            "scope": "active original Batch-1 training fold only",
        })

    final_scaler = StandardScaler().fit(source_x)
    final_original = final_scaler.transform(source_x)
    final_context = "final_batch1_fit"
    final_augmented, final_augmentation_provenance = generate_fixed_views(
        final_original, source_y, np.arange(len(source_x)), source_lines, final_context,
        cfg["augmentation_seed"], cfg["augmentation"]["perturbation_scale"])
    augmentation_frames.append(final_augmentation_provenance)
    augmentation_sanity_contexts.append(sanity_for_context(
        final_context, final_original, final_augmented, source_y,
        final_augmentation_provenance, cfg["augmentation"]["perturbation_scale"]))
    training_provenance.append(_make_training_input_provenance(
        final_context, np.arange(len(source_x)), source_lines, source_y,
        final_augmentation_provenance))
    final_ids = [_sample_id(int(line)) for line in source_lines]
    scaler_scopes.append({
        "context": final_context, "fit_batch": 1, "fit_original_samples": len(source_x),
        "fit_augmented_samples": 0, "fit_validation_samples": 0, "fit_target_samples": 0,
        "fit_sample_ids_sha256": _ids_digest(final_ids), "scope": "all original Batch 1 only",
    })
    augmentation_df = pd.concat(augmentation_frames, ignore_index=True)
    augmentation_df.to_csv(out / "a1_augmentation_provenance.csv.gz", index=False, compression="gzip")
    pd.concat(training_provenance, ignore_index=True).to_csv(
        out / "training_input_provenance.csv.gz", index=False, compression="gzip")
    pd.DataFrame(scaler_scopes).to_csv(out / "scaler_fit_scopes.csv", index=False)
    reproduction = verify_raw_perturbation_reproduction(
        augmentation_df, a1_paths["augmentation_provenance.csv.gz"])
    augmentation_sanity = {
        "status": "passed" if all(all(item["checks"].values()) for item in augmentation_sanity_contexts)
        and reproduction["all_checks_passed"] else "failed",
        "source_batch_only": True, "perturbation_scale": 0.5,
        "raw_perturbation_reproduction_against_a1_parent": reproduction,
        "contexts": augmentation_sanity_contexts,
    }
    save_json(out / "a1_augmentation_sanity.json", augmentation_sanity)
    flat_augmentation = [{
        "context": item["context"], "original_count": item["original_count"],
        "generated_view_count": item["generated_view_count"],
        "combined_training_count": item["combined_training_count"],
        **item["checks"],
    } for item in augmentation_sanity_contexts]
    pd.DataFrame(flat_augmentation).to_csv(out / "a1_augmentation_sanity_by_context.csv", index=False)
    if augmentation_sanity["status"] != "passed":
        raise GateFailure("Inherited A1 source-only augmentation sanity or parent reproduction failed")

    sanity = latent_sanity(
        cfg, prepared[1]["x_train"], source_y[prepared[1]["train_idx"]], access_log)
    save_json(out / "latent_feature_sanity.json", sanity)
    (out / "latent_feature_sanity_report.md").write_text(
        "# A2 latent-feature sanity validation\n\n"
        f"Status: **{sanity['status'].upper()}**. Only Batch 1 was loaded. "
        f"Low/high reconstruction max error: `{sanity['low_plus_high_max_absolute_error']:.3e}`; "
        f"identity-statistics reconstruction max error: `{sanity['identity_statistics_max_absolute_error']:.3e}`. "
        "Tensor shapes, finite values, positive generated standard deviations, exact ResNet4/5 parameter sharing, "
        "and finite nonzero shared-backbone gradients are recorded in `latent_feature_sanity.json`.\n",
        encoding="utf-8")
    save_json(out / "data_access_log.json", access_log)
    if sanity["status"] != "passed":
        raise GateFailure(f"A2 latent sanity failed: {[k for k, v in sanity['checks'].items() if not v]}")

    cv_rows, history_rows, cv_predictions, cv_confusions = [], [], [], []
    latent_count_rows = []
    for fold in range(1, cfg["validation"]["folds"] + 1):
        item = prepared[fold]
        train_idx, valid_idx = item["train_idx"], item["valid_idx"]
        combined_x = np.concatenate([item["x_train"], item["augmented"]])
        combined_y = np.concatenate([source_y[train_idx], source_y[train_idx]])
        input_types = np.asarray(["original"] * len(train_idx) + ["a1_fixed_augmented"] * len(train_idx))
        latent_seed = cfg["latent_feature_seed"] + fold
        model, history, best_epoch, best_accuracy, best_loss = train_cv_model(
            combined_x, combined_y, input_types, item["x_valid"], source_y[valid_idx],
            cfg, cfg["model_seed"] + fold, latent_seed)
        logits = predict_logits(model, item["x_valid"], cfg["training"]["batch_size"])
        pred = logits.argmax(1) + 1
        measured = float(accuracy_score(source_y[valid_idx], pred))
        if not math.isclose(measured, best_accuracy, rel_tol=0.0, abs_tol=0.0):
            raise AssertionError(f"Fold {fold} restored checkpoint accuracy mismatch")
        cv_rows.append({
            "fold": fold, "original_training_samples": len(train_idx),
            "a1_augmented_training_samples": len(item["augmented"]),
            "combined_training_inputs": len(combined_x), "validation_samples": len(valid_idx),
            "overlap_samples": 0, "scaler_fit_batch": 1, "scaler_fit_original_samples": len(train_idx),
            "augmentation_source_batch": 1, "augmentation_seed": cfg["augmentation_seed"],
            "perturbation_scale": 0.5, "model_seed": cfg["model_seed"] + fold,
            "latent_feature_seed": latent_seed,
            "original_branch_views_per_epoch": len(combined_x),
            "generated_branch_views_per_epoch": len(combined_x),
            "epochs_run": len(history), "selected_epoch": best_epoch,
            "stopped_early": len(history) < cfg["training"]["max_epochs"],
            "accuracy": measured, "selected_validation_loss": best_loss,
        })
        history_rows.extend({"fold": fold, **row} for row in history)
        latent_count_rows.extend({
            "context": f"cv_fold_{fold}", "epoch": row["epoch"],
            "source_batch": 1, "training_inputs": len(combined_x),
            "original_branch_views": row["original_branch_views"],
            "generated_branch_views": row["generated_branch_views"],
            "views_per_training_latent": 1, "target_inputs": 0,
        } for row in history)
        fold_prediction = pd.DataFrame({
            "sample_id": [_sample_id(int(source_lines[index])) for index in valid_idx],
            "batch": 1, "line_number": source_lines[valid_idx], "cv_fold": fold,
            "true_gas_label": source_y[valid_idx], "predicted_gas_label": pred,
            "correct": pred == source_y[valid_idx], "inference_path": "original_only",
            "a1_augmentation_enabled": False, "latent_generation_enabled": False,
        })
        cv_predictions.append(fold_prediction)
        cv_confusions.extend(confusion_rows(f"cv_fold_{fold}", source_y[valid_idx], pred))
    cv_df = pd.DataFrame(cv_rows)
    history_df = pd.DataFrame(history_rows)
    cv_prediction_df = pd.concat(cv_predictions, ignore_index=True)
    cv_df.to_csv(out / "cv_fold_results.csv", index=False)
    history_df.to_csv(out / "cv_training_history.csv", index=False)
    cv_prediction_df.to_csv(out / "cv_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(cv_confusions).to_csv(out / "cv_confusion_matrices.csv", index=False)
    selected_epochs = int(math.floor(float(cv_df.selected_epoch.median()) + 0.5))

    final_combined_x = np.concatenate([final_original, final_augmented])
    final_combined_y = np.concatenate([source_y, source_y])
    final_input_types = np.asarray(["original"] * len(source_y) + ["a1_fixed_augmented"] * len(source_y))
    final_model, final_history = train_fixed_epochs(
        final_combined_x, final_combined_y, final_input_types, cfg,
        cfg["model_seed"], cfg["latent_feature_seed"], selected_epochs)
    final_history_df = pd.DataFrame(final_history)
    final_history_df.to_csv(out / "final_training_history.csv", index=False)
    latent_count_rows.extend({
        "context": final_context, "epoch": row["epoch"], "source_batch": 1,
        "training_inputs": len(final_combined_x),
        "original_branch_views": row["original_branch_views"],
        "generated_branch_views": row["generated_branch_views"],
        "views_per_training_latent": 1, "target_inputs": 0,
    } for row in final_history)
    latent_counts_df = pd.DataFrame(latent_count_rows)
    latent_counts_df.to_csv(out / "latent_generation_counts.csv", index=False)

    checkpoint_path = out / "model.pt"
    torch.save({
        "model_state_dict": final_model.state_dict(), "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_, "scaler_fit_samples": len(source_x),
        "architecture": cfg["architecture"], "experiment_id": "A2",
        "selected_epochs": selected_epochs, "training": cfg["training"],
        "augmentation": cfg["augmentation"],
        "latent_feature_generation": cfg["latent_feature_generation"],
        "seeds": {"model": cfg["model_seed"], "augmentation": cfg["augmentation_seed"],
                  "latent_feature": cfg["latent_feature_seed"]},
        "inference_path": "original only; A1 augmentation and latent generation disabled",
        "optimizer": {"name": "SGD", "learning_rate": .001, "momentum": .9, "weight_decay": 1e-4},
        "scheduler": NO_SCHEDULER,
    }, checkpoint_path)
    frozen_utc = utc_now()
    freeze_manifest = {
        "frozen_utc": frozen_utc, "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256(checkpoint_path), "selected_epochs": selected_epochs,
        "selection_scope": "Batch 1 five-fold cross-validation only",
        "batches_loaded_at_freeze": sorted({row["batch"] for row in access_log}),
        "target_batches_loaded_at_freeze": sorted({row["batch"] for row in access_log} & set(cfg["target_batches"])),
        "inference_generation_enabled": False,
    }
    save_json(out / "model_freeze_manifest.json", freeze_manifest)
    if freeze_manifest["target_batches_loaded_at_freeze"]:
        raise GateFailure("Target batch was loaded before checkpoint freeze")

    # Reference outcome artifacts are deliberately loaded only after A2 is frozen.
    b0_summary = json.loads(b0_paths["summary.json"].read_text(encoding="utf-8"))
    b0_metrics = pd.read_csv(b0_paths["batch_metrics.csv"])
    a1_summary = json.loads(a1_paths["summary.json"].read_text(encoding="utf-8"))
    a1_metrics = pd.read_csv(a1_paths["batch_metrics.csv"])
    for prefix, paths in (("canonical_sgd_b0", b0_paths), ("a1_scale_0p5", a1_paths)):
        for name in ("summary.json", "batch_metrics.csv", "cv_fold_results.csv"):
            inputs.append(_artifact(root, paths[name], f"{prefix}_{name.rsplit('.', 1)[0]}", "post_freeze_comparison"))

    arrays, labels, lines = {1: source_x}, {1: source_y}, {1: source_lines}
    for batch in cfg["target_batches"]:
        path = root / ds_cfg["path"] / ds_cfg["batch_file_pattern"].format(batch_id=batch)
        arrays[batch], labels[batch], lines[batch] = load_batch(path, batch, 128)
        access_log.append({
            "batch": batch, "file": str(path.relative_to(root)), "access_number_for_file": 1,
            "phase": "post_freeze_target_evaluation_load", "loaded_utc": utc_now(),
            "purpose": "one scored target evaluation plus mandatory in-memory checkpoint replay",
            "model_frozen_utc": frozen_utc,
        })
        inputs.append(_artifact(root, path, "dataset_batch", "post_freeze_target_evaluation",
                                batch=batch, records=len(arrays[batch])))
    save_json(out / "data_access_log.json", access_log)

    all_x = np.vstack([arrays[batch] for batch in range(1, 11)])
    all_y = np.concatenate([labels[batch] for batch in range(1, 11)])
    dataset_validation = {
        "records": int(len(all_x)), "batches": list(range(1, 11)),
        "batch_record_counts": {str(batch): int(len(arrays[batch])) for batch in range(1, 11)},
        "gas_labels": np.unique(all_y).astype(int).tolist(), "features": int(all_x.shape[1]),
        "finite_values": bool(np.isfinite(all_x).all()),
        "nonfinite_value_count": int((~np.isfinite(all_x)).sum()),
        "expected_structure_verified": bool(len(all_x) == 13910 and all_x.shape[1] == 128
                                             and np.array_equal(np.unique(all_y), np.arange(1, 7))),
        "exclusions": [], "deduplication": False, "clipping": False, "imputation": False,
        "validation_phase": "post checkpoint freeze; each target raw file loaded once",
    }
    save_json(out / "dataset_validation.json", dataset_validation)
    if not dataset_validation["expected_structure_verified"] or not dataset_validation["finite_values"]:
        raise GateFailure(f"Post-freeze direct full-dataset validation failed: {dataset_validation}")

    predictions = []
    for batch in cfg["target_batches"]:
        scaled = final_scaler.transform(arrays[batch])
        pred = predict_logits(final_model, scaled, cfg["training"]["batch_size"]).argmax(1) + 1
        predictions.append(pd.DataFrame({
            "sample_id": [f"batch{batch}:line{int(line)}" for line in lines[batch]],
            "batch": batch, "line_number": lines[batch], "true_gas_label": labels[batch],
            "predicted_gas_label": pred, "correct": pred == labels[batch],
            "inference_path": "original_only", "a1_augmentation_enabled": False,
            "latent_generation_enabled": False, "target_scored_evaluation_number": 1,
        }))
    prediction_df = pd.concat(predictions, ignore_index=True)
    prediction_df.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")

    recomputed, recomputed_folds, metric_df, confusion_df, proportions_df = recompute_from_saved_predictions(out)
    metric_df.to_csv(out / "batch_metrics.csv", index=False)
    confusion_df.to_csv(out / "confusion_matrices.csv", index=False)
    proportions_df.to_csv(out / "class_prediction_proportions.csv", index=False)
    recomputed_folds.to_csv(out / "recomputed_cv_fold_metrics.csv", index=False)
    fold_recompute_match = np.allclose(
        cv_df.sort_values("fold").accuracy.to_numpy(),
        recomputed_folds.sort_values("fold").accuracy.to_numpy(), rtol=0.0, atol=0.0)

    summary = {
        "experiment_id": "A2", "status": "completed", "implemented_experiments": ["A2"],
        "not_implemented_or_run": ["A3"],
        "description": "A1 scale-0.5 input augmentation plus source-only latent feature generation; not a physical sensor-drift simulation.",
        "canonical_backbone": "1D ResNet B0", "legacy_2d_reshape_used": False,
        "pca_used": False, "parameter_counts": parameter_counts(A2GasResNet1D(1e-5)),
        "tensor_shape_check": preflight["tensor_shapes"],
        **recomputed, "source_selected_epochs": selected_epochs,
        "b0_baseline_run": str(b0_dir.relative_to(root)), "a1_parent_run": str(a1_dir.relative_to(root)),
        "perturbation_scale": 0.5, "a1_augmentation_sanity_status": augmentation_sanity["status"],
        "latent_feature_sanity_status": sanity["status"],
        "optimizer": {"name": "SGD", "learning_rate": .001, "momentum": .9, "weight_decay": 1e-4},
        "scheduler": "none", "branch_loss": "0.5 * (mean CE original + mean CE generated)",
        "mse_loss_used": False, "contrastive_loss_used": False,
        "total_cv_generated_latent_views": int(latent_counts_df[latent_counts_df.context.str.startswith("cv_")].generated_branch_views.sum()),
        "total_final_fit_generated_latent_views": int(latent_counts_df[latent_counts_df.context == final_context].generated_branch_views.sum()),
        "seeds": {"model_final": cfg["model_seed"],
                  "model_cv_by_fold": {str(fold): cfg["model_seed"] + fold for fold in range(1, 6)},
                  "augmentation_each_context": cfg["augmentation_seed"],
                  "latent_final": cfg["latent_feature_seed"],
                  "latent_cv_by_fold": {str(fold): cfg["latent_feature_seed"] + fold for fold in range(1, 6)}},
    }
    save_json(out / "summary.json", summary)
    comparison_df = build_comparison(summary, metric_df, b0_summary, b0_metrics, a1_summary, a1_metrics)
    comparison_df.to_csv(out / "b0_a1_a2_comparison.csv", index=False)

    replay = replay_checkpoint(checkpoint_path, cfg, arrays, prediction_df)
    save_json(out / "checkpoint_replay.json", replay)
    target_load_counts = pd.Series([row["batch"] for row in access_log]).value_counts().to_dict()
    reference_final_hashes = {
        "b0_configuration": sha256(b0_paths["configuration.json"]),
        "a1_configuration": sha256(a1_paths["configuration.json"]),
    }
    report_inventory = sorted([
        "a1_augmentation_provenance.csv.gz", "a1_augmentation_sanity.json",
        "a1_augmentation_sanity_by_context.csv", "b0_a1_a2_comparison.csv", "batch_metrics.csv",
        "checkpoint_replay.json", "class_prediction_proportions.csv", "configuration.json",
        "confusion_matrices.csv", "cv_confusion_matrices.csv", "cv_fold_assignments.csv",
        "cv_fold_results.csv", "cv_predictions.csv.gz", "cv_training_history.csv", "data_access_log.json",
        "dataset_validation.json", "final_training_history.csv", "input_manifest.json",
        "implementation_fix_log.json",
        "latent_feature_sanity.json", "latent_feature_sanity_report.md", "latent_generation_counts.csv",
        "model.pt", "model_freeze_manifest.json", "output_inventory.json", "post_run_audit.json",
        "predictions.csv.gz", "preflight_validation.json", "recomputed_cv_fold_metrics.csv",
        "report.md", "run_manifest.json", "scaler_fit_scopes.csv", "software_versions.json",
        "summary.json", "training_input_provenance.csv.gz", "training_validation_curves.png",
    ])
    audit_checks = {
        "preflight_passed": preflight["status"] == "passed",
        "latent_sanity_passed": sanity["status"] == "passed",
        "a1_augmentation_sanity_passed": augmentation_sanity["status"] == "passed",
        "fold_coverage_once": len(cv_prediction_df) == len(source_x) and not cv_prediction_df.sample_id.duplicated().any(),
        "saved_cv_predictions_match_fold_results": bool(fold_recompute_match),
        "all_scalers_source_original_only": all(row["fit_batch"] == 1 and row["fit_augmented_samples"] == 0
                                                   and row["fit_target_samples"] == 0 for row in scaler_scopes),
        "all_latent_generation_source_only": bool(latent_counts_df.target_inputs.eq(0).all()),
        "one_generated_view_per_training_latent": bool((latent_counts_df.generated_branch_views == latent_counts_df.training_inputs).all()),
        "branch_counts_equal": bool((latent_counts_df.generated_branch_views == latent_counts_df.original_branch_views).all()),
        "training_tensors_finite": bool(history_df.nonfinite_tensor_count.eq(0).all()
                                        and final_history_df.nonfinite_tensor_count.eq(0).all()),
        "generated_sigmas_positive_during_training": bool(history_df.generated_sigma_min.gt(0).all()
                                                           and final_history_df.generated_sigma_min.gt(0).all()),
        "targets_absent_at_checkpoint_freeze": not freeze_manifest["target_batches_loaded_at_freeze"],
        "each_raw_batch_file_loaded_once": all(target_load_counts.get(batch, 0) == 1 for batch in range(1, 11)),
        "checkpoint_replay_agrees": replay["status"] == "passed",
        "summary_matches_saved_prediction_recomputation": all(
            math.isclose(summary[key], recomputed[key], rel_tol=0.0, abs_tol=0.0) for key in recomputed),
        "reference_configurations_unchanged": reference_initial_hashes == reference_final_hashes,
        "no_a3": summary["not_implemented_or_run"] == ["A3"],
    }
    audit = {
        "status": "passed" if all(audit_checks.values()) else "failed",
        "checked_utc": utc_now(), "checks": audit_checks,
        "independently_recomputed_from_saved_predictions": recomputed,
        "checkpoint_replay": replay, "input_sha256_manifest": "input_manifest.json",
        "loaded_batch_history": access_log, "scaler_fit_scope_artifact": "scaler_fit_scopes.csv",
        "fold_coverage": {"samples": len(cv_prediction_df), "unique_samples": cv_prediction_df.sample_id.nunique()},
        "generated_view_counts": {
            "cv_total": summary["total_cv_generated_latent_views"],
            "final_fit_total": summary["total_final_fit_generated_latent_views"]},
        "tensor_shapes": sanity["tensor_shapes"], "parameter_sharing": sanity["parameter_sharing"],
        "nan_inf_checks": {"sanity_all_finite": sanity["checks"]["all_tensors_finite"],
                           "training_nonfinite_tensor_count": int(history_df.nonfinite_tensor_count.sum()
                                                                   + final_history_df.nonfinite_tensor_count.sum())},
        "target_access_timing": {"frozen_utc": frozen_utc,
                                 "first_target_load_utc": next(row["loaded_utc"] for row in access_log if row["batch"] == 2),
                                 "target_loads_after_freeze": all(row.get("model_frozen_utc") == frozen_utc
                                                                  for row in access_log if row["batch"] in cfg["target_batches"])},
        "reference_configuration_hashes_before": reference_initial_hashes,
        "reference_configuration_hashes_after": reference_final_hashes,
        "all_checks_passed": all(audit_checks.values()),
    }
    save_json(out / "input_manifest.json", inputs)
    plot_curves(history_df, final_history_df, out / "training_validation_curves.png")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = None
    implementation = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                       "sha256": sha256(path)} for path in implementation_paths]
    save_json(out / "run_manifest.json", {
        "run_id": out.name, "experiment_id": "A2", "status": "completed",
        "created_utc": started, "completed_utc": utc_now(),
        "command": f"{sys.executable} scripts/run_a2_formal.py --config {config_file.relative_to(root)}",
        "working_directory": str(root), "git_revision": revision, "implementation_files": implementation,
        "preprocessing_fit_scope": "CV scaler: active original Batch-1 training fold only; final scaler: all original Batch 1 only.",
        "augmentation_scope": "One fixed A1 scale-0.5 view per active original source-training sample; no validation/target augmentation.",
        "latent_generation_scope": "One generated latent branch view per training input per optimization encounter; source only.",
        "selection_scope": "Five Batch-1 folds only; targets loaded after final checkpoint freeze.",
        "target_evaluation": "One scored evaluation after freeze; mandatory replay reused in-memory raw arrays without reloading files.",
        "sample_policy": "All valid records retained; no exclusion, deduplication, clipping, or imputation.",
        "exclusions": [], "raw_data_modified": False, "a3_implemented_or_run": False,
        "summary": summary,
    })
    write_report(out, cfg, summary, preflight, sanity, cv_df, metric_df, proportions_df,
                 confusion_df, comparison_df, audit, inputs, report_inventory)
    report_text = (out / "report.md").read_text(encoding="utf-8")
    consistency_checks = {
        "report_contains_cv_mean": f"{summary['source_cv_mean_accuracy']:.6f}" in report_text,
        "report_contains_cv_std": f"{summary['source_cv_std_accuracy']:.6f}" in report_text,
        "report_contains_target_mean": f"{summary['target_unweighted_mean_accuracy']:.6f}" in report_text,
        "report_contains_pooled_accuracy": f"{summary['target_pooled_accuracy']:.6f}" in report_text,
        "batch_metrics_equal_recomputation": (
            pd.read_csv(out / "batch_metrics.csv")[["batch", "samples", "correct"]].equals(
                metric_df[["batch", "samples", "correct"]])
            and np.allclose(pd.read_csv(out / "batch_metrics.csv").accuracy,
                            metric_df.accuracy, rtol=0.0, atol=1e-15)),
        "confusions_equal_recomputation": pd.read_csv(out / "confusion_matrices.csv").equals(confusion_df),
        "proportions_equal_recomputation": np.allclose(
            pd.read_csv(out / "class_prediction_proportions.csv").proportion,
            proportions_df.proportion, rtol=0.0, atol=1e-15),
    }
    audit["report_csv_summary_consistency"] = consistency_checks
    audit["checks"]["report_csv_summary_consistency"] = all(consistency_checks.values())
    audit["all_checks_passed"] = all(audit["checks"].values())
    audit["status"] = "passed" if audit["all_checks_passed"] else "failed"
    save_json(out / "post_run_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Post-run audit failed: {[k for k, v in audit['checks'].items() if not v]}")

    actual_before_inventory = sorted(path.name for path in out.iterdir())
    expected_before_inventory = [name for name in report_inventory if name != "output_inventory.json"]
    if actual_before_inventory != expected_before_inventory:
        raise RuntimeError({"output_inventory_expected": expected_before_inventory,
                            "output_inventory_actual": actual_before_inventory})
    inventory = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(out.iterdir())]
    inventory.append({"path": "output_inventory.json", "bytes": None, "sha256": None,
                      "note": "Self-entry; size and digest omitted to avoid recursive self-hashing."})
    save_json(out / "output_inventory.json", inventory)
    if sorted(path.name for path in out.iterdir()) != report_inventory:
        raise RuntimeError("Final output inventory does not match the complete actual output set")


def main(config_path: str = "configs/a2_formal.json") -> Path:
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_formal_a2"
    out = root / "runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    try:
        _execute(root, config_file, cfg, out, started)
    except Exception as exc:
        _failure_finalize(out, started, exc)
    finally:
        for path in out.iterdir():
            path.chmod(0o444)
        out.chmod(0o555)
    return out


if __name__ == "__main__":
    print(main())
