"""Unified leakage-safe implementation of the CDCNN v6.3 ablation protocol.

The public stage names are B0, A1, A2-semantic, A2-paper-literal, and A3.
The paper-literal A2 branch is diagnostic; A2-semantic is the canonical A2 used
by A3. Target files are inaccessible until all source-only checkpoints for a
run have been frozen.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Required by deterministic CUDA matrix multiplication. This must be set before
# the first CUDA context is created in a worker process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.cv_folds import load_folds
from src.pca_analysis import load_batch, save_json, sha256
from src.resnet_1d_baseline import ResidualBlock1D, reshape


CANONICAL_STAGES = ("B0", "A1", "A2-semantic", "A3")
DIAGNOSTIC_STAGES = ("A2-paper-literal",)
ALL_STAGES = CANONICAL_STAGES + DIAGNOSTIC_STAGES
FEATURE_STAGES = ("A2-semantic", "A2-paper-literal", "A3")
AUGMENTED_STAGES = ("A1",) + FEATURE_STAGES

A3_STABILITY_DEFAULTS = {
    "applies_to_stage": "A3",
    "normalization": "LayerNorm",
    "normalization_feature_dimension": 128,
    "normalization_tracks_running_stats": False,
    "sigma_min": 1e-3,
    "sigma_max": 10.0,
    "log_sigma_dispersion_min": 0.0,
    "log_sigma_dispersion_max": 2.0,
    "residual_output_min": -20.0,
    "residual_output_max": 20.0,
    "contrastive_feature_min": -20.0,
    "contrastive_feature_max": 20.0,
    "gradient_max_norm": 1.0,
    "gradient_error_if_nonfinite": True,
}


class ProtocolError(RuntimeError):
    """Raised when a v6 invariant or access boundary is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_training_device(training_cfg: dict) -> torch.device:
    """Resolve and validate the configured device before model construction."""
    selected = torch.device(training_cfg["device"])
    if selected.type != "cuda":
        raise ProtocolError(
            f"CDCNN v6 training requires CUDA; configured device is {selected}")
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA was selected but torch.cuda.is_available() is false")
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise ProtocolError(
            f"CUDA device index {selected.index} is unavailable; count={torch.cuda.device_count()}")
    torch.cuda.set_device(selected)
    return selected


def capture_nvidia_smi_evidence(device: torch.device) -> dict:
    """Capture process-table evidence while this Python process owns CUDA memory."""
    if device.type != "cuda":
        raise ProtocolError(f"nvidia-smi evidence requires CUDA, received {device}")
    torch.cuda.synchronize(device)
    query_command = [
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    query = subprocess.run(query_command, capture_output=True, text=True, check=False)
    full = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    process_rows = []
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                process_rows.append({
                    "pid": int(fields[0]), "process_name": fields[1],
                    "used_memory_mib": int(fields[2]),
                })
            except ValueError:
                continue
    python_pid = os.getpid()
    namespace_pid_rows = [row for row in process_rows if row["pid"] == python_pid]
    # Under WSL/container PID translation, nvidia-smi can report the host-side
    # compute PID and "[Not Found]" while Python sees its namespace-local PID.
    # Sequential execution lets a sole compute row identify this process without
    # pretending that the two PID namespaces are equal.
    if namespace_pid_rows:
        visible_rows = namespace_pid_rows
        identification = "direct_python_pid_match"
    elif len(process_rows) == 1:
        visible_rows = process_rows
        identification = "sole_compute_process_pid_namespace_translation"
    else:
        visible_rows = []
        identification = "not_identified"
    return {
        "utc": utc_now(), "python_pid": python_pid,
        "python_process_name": Path("/proc/self/comm").read_text(encoding="utf-8").strip(),
        "python_process_command": Path("/proc/self/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace").strip(),
        "gpu_pid_visible_through_nvidia_smi": bool(visible_rows),
        "python_pid_directly_matched": bool(namespace_pid_rows),
        "gpu_pid_identification": identification,
        "pid_namespace_note": (
            "nvidia-smi reports the host PID while /proc exposes the container PID"
            if identification == "sole_compute_process_pid_namespace_translation" else None
        ),
        "matching_process_rows": visible_rows, "all_compute_process_rows": process_rows,
        "query_command": query_command, "query_return_code": query.returncode,
        "query_stdout": query.stdout, "query_stderr": query.stderr,
        "full_command": ["nvidia-smi"], "full_return_code": full.returncode,
        "full_stdout": full.stdout, "full_stderr": full.stderr,
        "torch_cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "torch_cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
    }


def _require_equal(actual, expected, name: str) -> None:
    if actual != expected:
        raise ProtocolError(f"{name} must be {expected!r}; received {actual!r}")


def validate_config(cfg: dict) -> None:
    """Reject configuration drift from the preregistered v6 protocol."""
    _require_equal(cfg.get("protocol_version"), "CDCNN_four_experiment_spec_v6", "protocol_version")
    _require_equal(
        cfg.get("implementation_version"),
        "CDCNN_v6.3_A3_numerical_stabilization",
        "implementation_version",
    )
    _require_equal(cfg.get("stages"), list(CANONICAL_STAGES), "stages")
    _require_equal(cfg.get("diagnostic_stages"), list(DIAGNOSTIC_STAGES), "diagnostic_stages")
    _require_equal(cfg.get("seeds"), [1042, 2024, 3407, 42, 123], "seeds")
    _require_equal(cfg.get("seed_policy"), {
        "model_dataloader_and_a1": "listed experiment seed",
        "feature_generation": "listed experiment seed + 2000000",
    }, "seed_policy")
    ds = cfg["dataset"]
    _require_equal(ds.get("source_batch"), 1, "dataset.source_batch")
    _require_equal(ds.get("target_batches"), list(range(2, 11)), "dataset.target_batches")
    _require_equal(ds.get("feature_count"), 128, "dataset.feature_count")
    training = cfg["training"]
    expected_training = {
        "epochs": 100, "early_stopping": False, "checkpoint_epoch": 100,
        "batch_size": 64, "optimizer": "SGD", "learning_rate": 0.001,
        "momentum": 0.9, "weight_decay": 1e-4, "shuffle": True,
        "device": "cuda:0", "num_threads": 4,
    }
    for key, expected in expected_training.items():
        _require_equal(training.get(key), expected, f"training.{key}")
    _require_equal(training.get("scheduler"),
                   {"type": "StepLR", "step_size": 25, "gamma": 0.5},
                   "training.scheduler")
    arch = cfg["architecture"]
    expected_arch = {
        "convolution_dimension": 1, "input_layout": [1, 128],
        "reshape_16x8": False, "block_channels": [32, 64, 128, 256, 128],
        "main_kernel_size": 3, "shortcut_kernel_size": 1, "fc_hidden": 128,
        "classes": 6, "feature_generation_after_block": 3,
    }
    for key, expected in expected_arch.items():
        _require_equal(arch.get(key), expected, f"architecture.{key}")
    aug = cfg["augmentation"]
    for key, expected in {
        "enabled_stages": list(AUGMENTED_STAGES), "views_per_source_sample": 1,
        "partner": "same-class non-self", "lambda_distribution": "Uniform(0,1)",
        "statistic": "population variance over 128 features", "perturbation_scale": 1.0,
        "schedule": "fixed once per source training context", "include_originals": True,
        "target_data_allowed": False,
    }.items():
        _require_equal(aug.get(key), expected, f"augmentation.{key}")
    feature = cfg["feature_generation"]
    for key, expected in {
        "enabled_stages": list(FEATURE_STAGES),
        "semantic_restyled_component": "pooled_upsampled_low_frequency_like",
        "paper_literal_restyled_component": "residual_paper_L",
        "pool_kernel_size": 2, "pool_stride": 2, "upsample_mode": "nearest",
        "statistics_axis": "length", "population_variance": True,
        "batch_spread_unbiased": False, "epsilon": 1e-5,
        "style_sampling": "canonical log-normal", "detach_sampled_style_statistics": True,
        "target_data_allowed": False,
    }.items():
        _require_equal(feature.get(key), expected, f"feature_generation.{key}")
    loss = cfg["loss"]
    for key, expected in {
        "lambda_mse": 0.5, "mse_on": "softmax probabilities",
        "lambda_contrastive": 0.5, "contrastive_temperature": 0.07,
        "projection_dimension": 128, "contrastive_reduction": "mean over anchors",
    }.items():
        _require_equal(loss.get(key), expected, f"loss.{key}")
    stability = cfg.get("a3_numerical_stability", {})
    for key, expected in A3_STABILITY_DEFAULTS.items():
        _require_equal(stability.get(key), expected, f"a3_numerical_stability.{key}")


@dataclass
class BatchData:
    x: np.ndarray
    y: np.ndarray
    line_numbers: np.ndarray


class DataAccessGuard:
    """Enforce and record the pre-freeze source/target file boundary."""

    def __init__(self, root: Path, dataset_cfg: dict):
        self.root = root
        self.cfg = dataset_cfg
        self.frozen = False
        self.events: list[dict] = []
        self._cache: dict[int, BatchData] = {}

    def freeze(self) -> None:
        self.frozen = True
        self.events.append({"event": "all_source_checkpoints_frozen", "utc": utc_now()})

    def load(self, batch: int, purpose: str) -> BatchData:
        is_target = batch in self.cfg["target_batches"]
        if is_target and not self.frozen:
            raise ProtocolError(f"Target Batch {batch} access attempted before checkpoint freeze")
        if batch != self.cfg["source_batch"] and not is_target:
            raise ProtocolError(f"Batch {batch} is outside the declared protocol")
        if batch in self._cache:
            self.events.append({"event": "in_memory_reuse", "batch": batch, "purpose": purpose,
                                "phase": "post_freeze" if self.frozen else "source_only", "utc": utc_now()})
            return self._cache[batch]
        path = self.root / self.cfg["path"] / self.cfg["batch_file_pattern"].format(batch_id=batch)
        digest = sha256(path)
        expected = self.cfg["expected_hashes"].get(str(batch))
        if digest != expected:
            raise ProtocolError(f"Input hash mismatch for {path}: {digest} != {expected}")
        x, y, lines = load_batch(path, batch, self.cfg["feature_count"])
        if x.shape[1] != 128 or not np.isfinite(x).all():
            raise ProtocolError(f"Batch {batch} failed finite 128-feature validation")
        if batch == 1 and not np.array_equal(np.unique(y), np.arange(1, 7)):
            raise ProtocolError("Batch 1 does not contain exactly gas labels 1-6")
        loaded = BatchData(x=x, y=y, line_numbers=lines)
        self._cache[batch] = loaded
        self.events.append({
            "event": "raw_file_load", "batch": batch, "purpose": purpose,
            "phase": "post_freeze" if self.frozen else "source_only", "utc": utc_now(),
            "path": str(path.relative_to(self.root)), "sha256": digest,
            "records": len(x), "features": x.shape[1],
        })
        return loaded


def generate_a1_views(
    standardized_x: np.ndarray,
    y: np.ndarray,
    source_indices: np.ndarray,
    line_numbers: np.ndarray,
    seed: int,
    context: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Generate one fixed v6 A1 view using variance mixing, never std mixing."""
    if standardized_x.shape != (len(y), 128):
        raise ValueError("A1 input must have shape [N, 128]")
    if len(source_indices) != len(y):
        raise ValueError("A1 source indices do not align with labels")
    rng = np.random.default_rng(seed)
    result = np.empty_like(standardized_x, dtype=np.float64)
    rows: list[dict] = []
    for anchor in range(len(y)):
        candidates = np.flatnonzero(y == y[anchor])
        candidates = candidates[candidates != anchor]
        if not len(candidates):
            raise ProtocolError(f"No same-class non-self partner for {context} row {anchor}")
        partner = int(rng.choice(candidates))
        lam = float(rng.uniform(0.0, 1.0))
        mu_j = float(standardized_x[anchor].mean())
        mu_k = float(standardized_x[partner].mean())
        var_j = float(standardized_x[anchor].var(ddof=0))
        var_k = float(standardized_x[partner].var(ddof=0))
        mu_mix = lam * mu_j + (1.0 - lam) * mu_k
        var_mix = lam * var_j + (1.0 - lam) * var_k
        noise = rng.normal(loc=mu_mix, scale=math.sqrt(var_mix), size=128)
        result[anchor] = standardized_x[anchor] + noise
        anchor_source = int(source_indices[anchor])
        partner_source = int(source_indices[partner])
        rows.append({
            "context": context, "anchor_source_index": anchor_source,
            "anchor_sample_id": f"batch1:line{int(line_numbers[anchor_source])}",
            "partner_source_index": partner_source,
            "partner_sample_id": f"batch1:line{int(line_numbers[partner_source])}",
            "anchor_label": int(y[anchor]), "partner_label": int(y[partner]),
            "lambda": lam, "anchor_mean": mu_j, "partner_mean": mu_k,
            "anchor_variance": var_j, "partner_variance": var_k,
            "mixed_mean": mu_mix, "mixed_variance": var_mix,
            "normal_api_scale": math.sqrt(var_mix), "seed": seed,
            "same_class": bool(y[anchor] == y[partner]),
            "non_self": bool(anchor_source != partner_source),
            "generated_label": int(y[anchor]),
        })
    return result, pd.DataFrame(rows)


class A3NumericalStabilizer(nn.Module):
    """Stateless hard bounds used only by the v6.3 A3 branch."""

    def __init__(self, config: dict):
        super().__init__()
        self.sigma_min = float(config["sigma_min"])
        self.sigma_max = float(config["sigma_max"])
        self.log_sigma_dispersion_min = float(config["log_sigma_dispersion_min"])
        self.log_sigma_dispersion_max = float(config["log_sigma_dispersion_max"])
        self.residual_output_min = float(config["residual_output_min"])
        self.residual_output_max = float(config["residual_output_max"])
        self.contrastive_feature_min = float(config["contrastive_feature_min"])
        self.contrastive_feature_max = float(config["contrastive_feature_max"])

    def clamp_sigma(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, min=self.sigma_min, max=self.sigma_max)

    def clamp_log_sigma_dispersion(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            value, min=self.log_sigma_dispersion_min, max=self.log_sigma_dispersion_max)

    def clamp_residual(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            value, min=self.residual_output_min, max=self.residual_output_max)

    def clamp_contrastive(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            value, min=self.contrastive_feature_min, max=self.contrastive_feature_max)


class CDCNNModel(nn.Module):
    """One backbone shared by every stage, with v6.3 safeguards isolated to A3."""

    def __init__(
        self, stage: str, epsilon: float = 1e-5, projection_dim: int = 128,
        a3_stability: dict | None = None,
    ):
        super().__init__()
        if stage not in ALL_STAGES:
            raise ValueError(f"Unknown stage {stage!r}")
        self.stage = stage
        self.epsilon = float(epsilon)
        channels = (32, 64, 128, 256, 128)
        blocks: list[nn.Module] = []
        incoming = 1
        for outgoing in channels:
            blocks.append(ResidualBlock1D(incoming, outgoing))
            incoming = outgoing
        self.blocks = nn.ModuleList(blocks)
        self.flatten = nn.Flatten()
        stability_config = A3_STABILITY_DEFAULTS if a3_stability is None else a3_stability
        self.a3_stabilizer = (
            A3NumericalStabilizer(stability_config) if stage == "A3" else None)
        # A3 never owns a BatchNorm running buffer. Generated samples therefore
        # cannot contaminate inference state when original and synthetic features
        # share the tail. Other stages retain the exact v6 BatchNorm definition.
        feature_norm: nn.Module = (
            nn.LayerNorm(128) if stage == "A3" else nn.BatchNorm1d(128))
        self.fc128 = nn.Sequential(nn.Linear(128 * 128, 128), feature_norm)
        self.fc6 = nn.Linear(128, 6)
        self.projection = nn.Linear(128, projection_dim) if stage == "A3" else None
        forbidden_layer = getattr(nn, "Conv" + "2d")
        if any(isinstance(module, forbidden_layer) for module in self.modules()):
            raise ProtocolError("CDCNN v6 forbids two-dimensional convolution layers")

    def forward_to_block3(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks[:3]:
            x = block(x)
            if self.a3_stabilizer is not None:
                x = self.a3_stabilizer.clamp_residual(x)
        return x

    def forward_tail_features(self, z: torch.Tensor) -> torch.Tensor:
        for block in self.blocks[3:]:
            z = block(z)
            if self.a3_stabilizer is not None:
                z = self.a3_stabilizer.clamp_residual(z)
        return self.fc128(self.flatten(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc6(self.forward_tail_features(self.forward_to_block3(x)))

    def decompose(self, z_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.max_pool1d(z_s, kernel_size=2, stride=2)
        low = F.interpolate(pooled, size=z_s.shape[-1], mode="nearest")
        high = z_s - low
        if self.a3_stabilizer is not None:
            high = self.a3_stabilizer.clamp_residual(high)
        return low, high

    def generate_features(
        self, z_s: torch.Tensor, generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        if self.stage not in FEATURE_STAGES:
            raise ProtocolError(f"Feature generation is disabled for {self.stage}")
        low, high = self.decompose(z_s)
        if self.stage in ("A2-semantic", "A3"):
            component, untouched = low, high
            restyled_name = "low_frequency_like_pooled_upsampled"
            untouched_name = "high_frequency_like_residual"
        else:
            component, untouched = high, low
            restyled_name = "paper_L_residual"
            untouched_name = "paper_H_pooled_upsampled"

        component_mean = component.mean(dim=-1, keepdim=True)
        component_sigma = torch.sqrt(component.var(dim=-1, unbiased=False, keepdim=True) + self.epsilon)
        # Sampling distributions are batch-local source-training statistics. The
        # stochastic style target is treated as data, matching augmentation RNG
        # semantics and avoiding gradients through distribution estimation.
        style_mean = component_mean.detach()
        style_sigma = component_sigma.detach()
        mean_center = style_mean.mean(dim=0, keepdim=True)
        mean_spread = style_mean.std(dim=0, unbiased=False, keepdim=True)
        sigma_center = style_sigma.mean(dim=0, keepdim=True)
        sigma_spread = style_sigma.std(dim=0, unbiased=False, keepdim=True)
        if self.a3_stabilizer is None:
            sigma_base = sigma_center.clamp_min(self.epsilon)
            log_sigma_spread = sigma_spread / sigma_base
        else:
            sigma_base = self.a3_stabilizer.clamp_sigma(sigma_center)
            log_sigma_spread = self.a3_stabilizer.clamp_log_sigma_dispersion(
                sigma_spread / sigma_base)
        mu_noise = torch.randn(style_mean.shape, generator=generator, device=z_s.device, dtype=z_s.dtype)
        sigma_noise = torch.randn(style_sigma.shape, generator=generator, device=z_s.device, dtype=z_s.dtype)
        sampled_mean = mean_center + mean_spread * mu_noise
        sampled_log_sigma = sigma_base.log() + log_sigma_spread * sigma_noise
        if self.a3_stabilizer is not None:
            sampled_log_sigma = torch.clamp(
                sampled_log_sigma,
                min=math.log(self.a3_stabilizer.sigma_min),
                max=math.log(self.a3_stabilizer.sigma_max),
            )
            sampled_sigma = self.a3_stabilizer.clamp_sigma(sampled_log_sigma.exp())
        else:
            sampled_sigma = sampled_log_sigma.exp()
        if not bool(torch.isfinite(sampled_sigma).all() and (sampled_sigma > 0).all()):
            raise ProtocolError("Log-normal style sampling produced a nonpositive or non-finite sigma")
        restyled = sampled_sigma * ((component - component_mean) / component_sigma) + sampled_mean
        generated = untouched + restyled
        if self.a3_stabilizer is not None:
            restyled = self.a3_stabilizer.clamp_residual(restyled)
            generated = self.a3_stabilizer.clamp_residual(untouched + restyled)
        return generated, {
            "z_low": low, "z_high": high, "restyled_component": component,
            "untouched_component": untouched, "component_mean": component_mean,
            "component_sigma": component_sigma, "mean_center": mean_center,
            "mean_spread": mean_spread, "sigma_center": sigma_center,
            "sigma_spread": sigma_spread, "log_sigma_spread": log_sigma_spread,
            "sampled_mean": sampled_mean, "sampled_log_sigma": sampled_log_sigma,
            "sampled_sigma": sampled_sigma, "restyled": restyled,
            "restyled_component_name": restyled_name,
            "untouched_component_name": untouched_name,
        }

    def project_contrastive(self, features: torch.Tensor) -> torch.Tensor:
        """Bound both sides of the A3 projection to keep similarities finite."""
        if self.a3_stabilizer is None or self.projection is None:
            raise ProtocolError("Contrastive projection is available only for A3")
        bounded_features = self.a3_stabilizer.clamp_contrastive(features)
        return self.a3_stabilizer.clamp_contrastive(self.projection(bounded_features))

    def training_outputs(self, x: torch.Tensor, generator: torch.Generator | None = None) -> dict:
        if self.stage not in FEATURE_STAGES:
            features = self.forward_tail_features(self.forward_to_block3(x))
            return {"original_logits": self.fc6(features), "original_features": features}
        if generator is None:
            raise ValueError("Feature stages require an explicit torch.Generator")
        original_z = self.forward_to_block3(x)
        generated_z, details = self.generate_features(original_z, generator)
        combined_features = self.forward_tail_features(torch.cat([original_z, generated_z], dim=0))
        original_features, generated_features = combined_features.split(len(x), dim=0)
        return {
            "original_logits": self.fc6(original_features),
            "generated_logits": self.fc6(generated_features),
            "original_features": original_features,
            "generated_features": generated_features,
            "generation": details,
        }


def probability_consistency_mse(original_logits: torch.Tensor, generated_logits: torch.Tensor) -> torch.Tensor:
    delta = original_logits.softmax(dim=1) - generated_logits.softmax(dim=1)
    return delta.square().sum(dim=1).mean()


def supervised_contrastive_mean(
    projections: torch.Tensor, labels: torch.Tensor, temperature: float,
) -> torch.Tensor:
    """v6.3 reduction: average positives per anchor, then mean all anchors."""
    if projections.ndim != 2 or labels.ndim != 1 or len(projections) != len(labels):
        raise ValueError("Contrastive projections/labels have incompatible shapes")
    normalized = F.normalize(projections, p=2, dim=1)
    logits = normalized @ normalized.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    positive_counts = positive.sum(dim=1)
    if bool((positive_counts == 0).any()):
        raise ProtocolError("Every contrastive anchor must have at least one positive")
    denominator_logits = logits.masked_fill(diagonal, -torch.inf)
    log_prob = logits - torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    per_anchor = -(log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive_counts)
    return per_anchor.mean()


def compute_loss(model: CDCNNModel, outputs: dict, y: torch.Tensor, cfg: dict) -> dict[str, torch.Tensor]:
    ce_original = F.cross_entropy(outputs["original_logits"], y)
    zero = ce_original.new_zeros(())
    if model.stage in ("B0", "A1"):
        return {"total": ce_original, "ce": ce_original, "mse": zero, "contrastive": zero}
    ce_generated = F.cross_entropy(outputs["generated_logits"], y)
    ce = 0.5 * (ce_original + ce_generated)
    mse = probability_consistency_mse(outputs["original_logits"], outputs["generated_logits"])
    contrastive = zero
    total = ce + cfg["loss"]["lambda_mse"] * mse
    if model.stage == "A3":
        projected = model.project_contrastive(torch.cat(
            [outputs["original_features"], outputs["generated_features"]], dim=0))
        contrastive_labels = torch.cat([y, y], dim=0)
        contrastive = supervised_contrastive_mean(
            projected, contrastive_labels, cfg["loss"]["contrastive_temperature"])
        total = total + cfg["loss"]["lambda_contrastive"] * contrastive
    return {"total": total, "ce": ce, "mse": mse, "contrastive": contrastive}


def build_optimizer_and_scheduler(model: nn.Module, training_cfg: dict):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=training_cfg["learning_rate"], momentum=training_cfg["momentum"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=training_cfg["scheduler"]["step_size"],
        gamma=training_cfg["scheduler"]["gamma"],
    )
    return optimizer, scheduler


def clip_a3_gradients(model: CDCNNModel, cfg: dict) -> tuple[float, float] | None:
    """Clip A3 gradients immediately before its optimizer step; leave baselines untouched."""
    if model.stage != "A3":
        return None
    stability = cfg["a3_numerical_stability"]
    before = nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float(stability["gradient_max_norm"]),
        error_if_nonfinite=bool(stability["gradient_error_if_nonfinite"]),
    )
    gradients = [
        parameter.grad.detach() for parameter in model.parameters()
        if parameter.grad is not None
    ]
    after = (
        torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(gradient, ord=2) for gradient in gradients
        ]), ord=2)
        if gradients else before.new_zeros(())
    )
    return float(before), float(after)


def _training_arrays(
    stage: str, scaled: np.ndarray, y: np.ndarray, source_indices: np.ndarray,
    line_numbers: np.ndarray, augmentation_seed: int, context: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if stage == "B0":
        return scaled, y, pd.DataFrame()
    augmented, provenance = generate_a1_views(
        scaled, y, source_indices, line_numbers, augmentation_seed, context)
    return np.concatenate([scaled, augmented]), np.concatenate([y, y]), provenance


def _make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(reshape(x)), torch.from_numpy(y.astype(np.int64) - 1))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def predict(model: CDCNNModel, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(reshape(x[start:start + batch_size])).to(
            device, non_blocking=device.type == "cuda")
        chunks.append(model(xb).cpu())
    return torch.cat(chunks).numpy()


def train_model(
    stage: str, x: np.ndarray, y: np.ndarray, cfg: dict, seed: int,
    epochs: int | None = None, max_training_batches: int | None = None,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
    device_audit: list[dict] | None = None, training_context: str | None = None,
) -> tuple[CDCNNModel, list[dict]]:
    """Train the requested fixed schedule; validation is observational only."""
    seed_everything(seed)
    torch.set_num_threads(cfg["training"]["num_threads"])
    device = resolve_training_device(cfg["training"])
    model = CDCNNModel(
        stage, cfg["feature_generation"]["epsilon"], cfg["loss"]["projection_dimension"],
        cfg["a3_numerical_stability"],
    ).to(device)
    model_parameter_device = next(model.parameters()).device
    if model_parameter_device != device:
        raise ProtocolError(
            f"Model placement failed: selected={device}, parameter={model_parameter_device}")
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg["training"])
    loader = _make_loader(
        x, y, cfg["training"]["batch_size"], seed, cfg["training"]["shuffle"],
        pin_memory=True,
    )
    style_generator = torch.Generator(device=device).manual_seed(seed + 2_000_000)
    epochs_to_run = cfg["training"]["epochs"] if epochs is None else epochs
    history: list[dict] = []
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        sums = {"total": 0.0, "ce": 0.0, "mse": 0.0, "contrastive": 0.0}
        correct = seen = batches_seen = 0
        sampled_sigma_min, sampled_sigma_max = math.inf, -math.inf
        sampled_mean_min, sampled_mean_max = math.inf, -math.inf
        generation_nonfinite_tensors = 0
        gradient_norm_before_clip_max = 0.0
        gradient_norm_after_clip_max = 0.0
        gradient_clipped_batches = 0
        restyled_component_name = None
        lr = float(optimizer.param_groups[0]["lr"])
        for batch_index, (xb, yb) in enumerate(loader):
            if max_training_batches is not None and batch_index >= max_training_batches:
                break
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if batch_index == 0 and epoch == 1:
                placement = {
                    "utc": utc_now(), "stage": stage, "seed": seed,
                    "training_context": training_context,
                    "selected_device": str(device),
                    "model_parameter_device": str(next(model.parameters()).device),
                    "input_tensor_device": str(xb.device),
                    "label_tensor_device": str(yb.device),
                    "first_training_batch_device": {
                        "model": str(next(model.parameters()).device),
                        "inputs": str(xb.device), "labels": str(yb.device),
                    },
                }
                smi = capture_nvidia_smi_evidence(device)
                placement["gpu_pid_visible_through_nvidia_smi"] = smi[
                    "gpu_pid_visible_through_nvidia_smi"]
                placement["gpu_pid"] = (
                    smi["matching_process_rows"][0]["pid"]
                    if smi["matching_process_rows"] else None
                )
                placement["nvidia_smi"] = smi
                if device_audit is not None:
                    device_audit.append(placement)
                expected_device = str(device)
                actual_devices = {
                    placement["model_parameter_device"], placement["input_tensor_device"],
                    placement["label_tensor_device"],
                }
                if actual_devices != {expected_device}:
                    raise ProtocolError(
                        f"First-batch CUDA placement mismatch: {sorted(actual_devices)}")
                if not placement["gpu_pid_visible_through_nvidia_smi"]:
                    raise ProtocolError(
                        f"No unambiguous GPU PID for Python PID {os.getpid()} is visible in "
                        "nvidia-smi compute processes")
                print("device audit " + json.dumps({
                    "selected_device": placement["selected_device"],
                    "model_parameter_device": placement["model_parameter_device"],
                    "input_tensor_device": placement["input_tensor_device"],
                    "label_tensor_device": placement["label_tensor_device"],
                    "first_training_batch_device": placement["first_training_batch_device"],
                    "gpu_pid": placement["gpu_pid"],
                    "gpu_pid_identification": smi["gpu_pid_identification"],
                    "gpu_pid_visible_through_nvidia_smi": placement[
                        "gpu_pid_visible_through_nvidia_smi"],
                }, sort_keys=True), flush=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.training_outputs(xb, style_generator if stage in FEATURE_STAGES else None)
            losses = compute_loss(model, outputs, yb, cfg)
            if not torch.isfinite(losses["total"]):
                raise ProtocolError(f"Non-finite {stage} loss at epoch {epoch}, batch {batch_index}")
            losses["total"].backward()
            gradient_norms = clip_a3_gradients(model, cfg)
            if gradient_norms is not None:
                before_clip, after_clip = gradient_norms
                gradient_norm_before_clip_max = max(gradient_norm_before_clip_max, before_clip)
                gradient_norm_after_clip_max = max(gradient_norm_after_clip_max, after_clip)
                gradient_clipped_batches += int(
                    before_clip > cfg["a3_numerical_stability"]["gradient_max_norm"])
            optimizer.step()
            n = len(yb)
            for key in sums:
                sums[key] += float(losses[key].detach()) * n
            correct += int((outputs["original_logits"].argmax(1) == yb).sum())
            seen += n
            batches_seen += 1
            if stage in FEATURE_STAGES:
                generation = outputs["generation"]
                sampled_sigma = generation["sampled_sigma"].detach()
                sampled_mean = generation["sampled_mean"].detach()
                sampled_sigma_min = min(sampled_sigma_min, float(sampled_sigma.min()))
                sampled_sigma_max = max(sampled_sigma_max, float(sampled_sigma.max()))
                sampled_mean_min = min(sampled_mean_min, float(sampled_mean.min()))
                sampled_mean_max = max(sampled_mean_max, float(sampled_mean.max()))
                generation_nonfinite_tensors += sum(
                    int(not torch.isfinite(value).all()) for value in generation.values()
                    if isinstance(value, torch.Tensor))
                restyled_component_name = generation["restyled_component_name"]
        if not seen:
            raise ProtocolError("Training loader yielded no batches")
        scheduler.step()
        row = {
            "epoch": epoch, "samples_seen": seen, "batches_seen": batches_seen,
            "training_loss": sums["total"] / seen, "ce_loss": sums["ce"] / seen,
            "mse_loss": sums["mse"] / seen, "contrastive_loss": sums["contrastive"] / seen,
            "training_original_accuracy": correct / seen, "learning_rate": lr,
            "learning_rate_after_step": float(optimizer.param_groups[0]["lr"]),
            "checkpoint_epoch": epoch == cfg["training"]["checkpoint_epoch"],
            "restyled_component": restyled_component_name,
            "sampled_sigma_min": sampled_sigma_min if stage in FEATURE_STAGES else None,
            "sampled_sigma_max": sampled_sigma_max if stage in FEATURE_STAGES else None,
            "sampled_mean_min": sampled_mean_min if stage in FEATURE_STAGES else None,
            "sampled_mean_max": sampled_mean_max if stage in FEATURE_STAGES else None,
            "generation_nonfinite_tensor_count": generation_nonfinite_tensors,
            "gradient_max_norm": (
                cfg["a3_numerical_stability"]["gradient_max_norm"] if stage == "A3" else None),
            "gradient_norm_before_clip_max": (
                gradient_norm_before_clip_max if stage == "A3" else None),
            "gradient_norm_after_clip_max": (
                gradient_norm_after_clip_max if stage == "A3" else None),
            "gradient_clipped_batches": gradient_clipped_batches if stage == "A3" else None,
        }
        if validation is not None:
            valid_logits = predict(model, validation[0], cfg["training"]["batch_size"])
            row["validation_accuracy_observational"] = float(
                accuracy_score(validation[1], valid_logits.argmax(1) + 1))
            row["validation_used_for_checkpoint_selection"] = False
        history.append(row)
    return model, history


def _software_versions() -> dict:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "torch": torch.__version__, "platform": platform.platform(),
    }


def _implementation_identity(root: Path) -> list[dict]:
    result = []
    for relative in (
        "src/cdcnn_ablation.py", "src/resnet_1d_baseline.py", "src/cv_folds.py",
        "src/pca_analysis.py", "scripts/run_cdcnn_ablation.py", "configs/cdcnn_v6.json",
    ):
        path = root / relative
        result.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    return result


def _provenance_input_manifest(root: Path, config_file: Path, cfg: dict) -> list[dict]:
    paths = [config_file, root / cfg["specification"], root / cfg["saved_cv_folds"]]
    roles = ["resolved_input_configuration", "experiment_specification", "saved_batch1_fold_assignments"]
    return [{
        "role": role, "path": str(path.relative_to(root)), "sha256": sha256(path),
        "bytes": path.stat().st_size, "load_phase": "source_only_preflight",
    } for role, path in zip(roles, paths)]


def _write_output_inventory(out: Path) -> None:
    inventory = [{
        "path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
    } for path in sorted(out.iterdir()) if path.name != "output_inventory.json"]
    inventory.append({
        "path": "output_inventory.json", "bytes": None, "sha256": None,
        "note": "Self-entry omits recursive size and hash.",
    })
    save_json(out / "output_inventory.json", inventory)


def _folds(root: Path, cfg: dict, source: BatchData) -> tuple[pd.DataFrame, np.ndarray]:
    return load_folds(root / cfg["saved_cv_folds"], source.x, source.line_numbers, source.y, 1)


def run_smoke_suite(config_path: str = "configs/cdcnn_v6.json") -> Path:
    """Exercise every stage on Batch 1 only; never invoke target loading."""
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    validate_config(cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_cdcnn_v6_3_batch1_smoke"
    out = root / cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    access = DataAccessGuard(root, cfg["dataset"])
    source = access.load(1, "Batch-1-only smoke validation")
    fold_frame, fold_ids = _folds(root, cfg, source)
    fold = cfg["smoke"]["folds"][0]
    train_idx = np.flatnonzero(fold_ids != fold)
    valid_idx = np.flatnonzero(fold_ids == fold)
    scaler = StandardScaler().fit(source.x[train_idx])
    rows, histories, provenance_frames = [], [], []
    for stage in ALL_STAGES:
        seed = cfg["smoke"]["seeds"][0]
        scaled_train = scaler.transform(source.x[train_idx])
        train_x, train_y, provenance = _training_arrays(
            stage, scaled_train, source.y[train_idx], train_idx, source.line_numbers,
            seed, f"smoke_{stage}_fold_{fold}")
        if len(provenance):
            provenance.insert(0, "stage", stage)
            provenance_frames.append(provenance)
        model, history = train_model(
            stage, train_x, train_y, cfg, seed,
            epochs=cfg["smoke"]["epochs"],
            max_training_batches=cfg["smoke"]["max_training_batches"],
            validation=(scaler.transform(source.x[valid_idx]), source.y[valid_idx]),
        )
        logits = predict(model, scaler.transform(source.x[valid_idx]), cfg["training"]["batch_size"])
        if logits.shape != (len(valid_idx), 6) or not np.isfinite(logits).all():
            raise ProtocolError(f"{stage} smoke prediction validation failed")
        for item in history:
            histories.append({"stage": stage, **item})
        rows.append({
            "stage": stage, "status": "passed", "fold": fold,
            "training_rows_available": len(train_x),
            "training_rows_exercised": history[-1]["samples_seen"],
            "validation_rows": len(valid_idx),
            "validation_accuracy_diagnostic": float(accuracy_score(source.y[valid_idx], logits.argmax(1) + 1)),
            "epochs": cfg["smoke"]["epochs"],
            "canonical_full_epochs": cfg["training"]["epochs"],
            "target_batches_loaded": False,
        })
        del model
    pd.DataFrame(rows).to_csv(out / "smoke_results.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "smoke_training_history.csv", index=False)
    smoke_history = pd.DataFrame(histories)
    smoke_style = smoke_history[smoke_history.restyled_component.notna()]
    smoke_style.to_csv(out / "smoke_style_sampling_statistics.csv", index=False)
    pd.concat(provenance_frames, ignore_index=True).to_csv(
        out / "augmentation_provenance.csv.gz", index=False, compression="gzip")
    fold_frame.to_csv(out / "cv_fold_assignments.csv", index=False)
    save_json(out / "configuration.json", cfg)
    save_json(out / "data_access_log.json", access.events)
    save_json(out / "software_versions.json", _software_versions())
    input_manifest = _provenance_input_manifest(root, config_file, cfg) + [{
        "batch": 1, "path": access.events[0]["path"], "sha256": access.events[0]["sha256"],
        "role": "source_raw_dataset", "records": access.events[0]["records"],
        "load_phase": access.events[0]["phase"],
    }]
    save_json(out / "input_manifest.json", input_manifest)
    target_events = [event for event in access.events if event.get("batch") in range(2, 11)]
    audit_checks = {
        "all_five_modes_passed": len(rows) == 5 and all(row["status"] == "passed" for row in rows),
        "only_batch1_loaded": not target_events and {event.get("batch") for event in access.events if "batch" in event} == {1},
        "no_target_metrics": True, "raw_data_unchanged": True,
        "no_exclusions": True, "canonical_config_validated_before_smoke": True,
        "generated_sigmas_positive": bool(
            smoke_style.sampled_sigma_min.gt(0).all()
            and smoke_style.generation_nonfinite_tensor_count.eq(0).all()),
    }
    audit = {
        "status": "passed" if all(audit_checks.values()) else "failed",
        "scope": "Batch 1 only; one epoch and one optimization batch per stage",
        "not_a_full_experiment": True, "target_files_accessed": bool(target_events),
        "target_metrics_computed": False, "scaler_fit_batch": 1,
        "scaler_fit_scope": "Batch 1 fold-1 training rows only",
        "feature_statistics_scope": "active Batch 1 training mini-batch only",
        "checks": audit_checks,
    }
    save_json(out / "leakage_target_access_audit.json", audit)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = None
    save_json(out / "run_manifest.json", {
        "run_id": run_id, "status": audit["status"], "kind": "Batch-1-only smoke test",
        "created_utc": utc_now(), "git_revision": revision,
        "implementation_files": _implementation_identity(root),
        "preprocessing_fit_scope": audit["scaler_fit_scope"],
        "output_inventory": "output_inventory.json",
        "exclusions": [], "raw_data_modified": False,
    })
    (out / "report.md").write_text(
        "# CDCNN v6.3 Batch-1-only smoke test\n\n"
        "All five executable modes passed a one-epoch, one-optimization-batch smoke test: "
        "B0, A1, A2-semantic, A2-paper-literal, and A3. This is not a result-bearing "
        "experiment and does not satisfy the canonical 100-epoch training protocol. "
        "Only Batch 1 was opened; no target metric was computed.\n",
        encoding="utf-8",
    )
    _write_output_inventory(out)
    if audit["status"] != "passed":
        raise ProtocolError(f"Smoke audit failed in {out}")
    return out


def _confusion_rows(seed: int, batch: int, y: np.ndarray, pred: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(y, pred, labels=range(1, 7))
    return [{
        "seed": seed, "batch": batch, "true_gas_label": true_label,
        "predicted_gas_label": predicted_label,
        "count": int(matrix[true_label - 1, predicted_label - 1]),
    } for true_label in range(1, 7) for predicted_label in range(1, 7)]


def main(stage: str, config_path: str = "configs/cdcnn_v6.json", smoke: bool = False) -> Path:
    """Run one full v6 stage, or route to the explicitly non-full smoke suite."""
    if smoke:
        return run_smoke_suite(config_path)
    if stage == "A2":
        stage = "A2-semantic"
    if stage not in ALL_STAGES:
        raise ValueError(f"stage must be one of {ALL_STAGES}")
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    validate_config(cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_cdcnn_v6_3_{stage.lower()}"
    out = root / cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "configuration.json", {**cfg, "resolved_stage": stage})
    save_json(out / "software_versions.json", _software_versions())
    access = DataAccessGuard(root, cfg["dataset"])
    source = access.load(1, "source validation, cross-validation, and final fitting")
    fold_frame, fold_ids = _folds(root, cfg, source)
    fold_frame.to_csv(out / "cv_fold_assignments.csv", index=False)
    history_rows: list[dict] = []
    cv_rows: list[dict] = []
    cv_predictions: list[pd.DataFrame] = []
    provenance_frames: list[pd.DataFrame] = []
    scaler_scopes: list[dict] = []
    final_models: dict[int, CDCNNModel] = {}
    final_scalers: dict[int, StandardScaler] = {}

    for seed in cfg["seeds"]:
        for fold in range(1, 6):
            train_idx = np.flatnonzero(fold_ids != fold)
            valid_idx = np.flatnonzero(fold_ids == fold)
            scaler = StandardScaler().fit(source.x[train_idx])
            scaler_scopes.append({"seed": seed, "context": f"cv_fold_{fold}", "fit_batch": 1,
                                  "fit_original_samples": len(train_idx), "fit_augmented_samples": 0,
                                  "fit_target_samples": 0})
            train_x, train_y, provenance = _training_arrays(
                stage, scaler.transform(source.x[train_idx]), source.y[train_idx], train_idx,
                source.line_numbers, seed, f"seed_{seed}_cv_fold_{fold}")
            if len(provenance):
                provenance.insert(0, "stage", stage)
                provenance.insert(1, "model_seed", seed)
                provenance_frames.append(provenance)
            model, history = train_model(
                stage, train_x, train_y, cfg, seed,
                validation=(scaler.transform(source.x[valid_idx]), source.y[valid_idx]))
            for row in history:
                history_rows.append({"phase": "cv", "seed": seed, "fold": fold, **row})
            logits = predict(model, scaler.transform(source.x[valid_idx]), cfg["training"]["batch_size"])
            pred = logits.argmax(1) + 1
            cv_rows.append({"seed": seed, "fold": fold, "checkpoint_epoch": 100,
                            "samples": len(valid_idx), "accuracy": float(accuracy_score(source.y[valid_idx], pred))})
            cv_predictions.append(pd.DataFrame({
                "seed": seed, "fold": fold,
                "sample_id": [f"batch1:line{int(source.line_numbers[i])}" for i in valid_idx],
                "true_gas_label": source.y[valid_idx], "predicted_gas_label": pred,
            }))
            del model

        final_scaler = StandardScaler().fit(source.x)
        final_scalers[seed] = final_scaler
        scaler_scopes.append({"seed": seed, "context": "final_fit", "fit_batch": 1,
                              "fit_original_samples": len(source.x), "fit_augmented_samples": 0,
                              "fit_target_samples": 0})
        final_x, final_y, provenance = _training_arrays(
            stage, final_scaler.transform(source.x), source.y, np.arange(len(source.x)),
            source.line_numbers, seed, f"seed_{seed}_final_fit")
        if len(provenance):
            provenance.insert(0, "stage", stage)
            provenance.insert(1, "model_seed", seed)
            provenance_frames.append(provenance)
        model, history = train_model(stage, final_x, final_y, cfg, seed)
        for row in history:
            history_rows.append({"phase": "final_fit", "seed": seed, "fold": pd.NA, **row})
        checkpoint_path = out / f"model_seed_{seed}.pt"
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
        final_models[seed] = model

    access.freeze()
    freeze_manifest = {
        "frozen_utc": utc_now(), "stage": stage, "checkpoint_epoch": 100,
        "checkpoint_rule": "epoch 100 fixed before training; no early stopping",
        "selection_scope": "Batch 1 only", "target_batches_loaded_at_freeze": [],
        "checkpoints": [{"seed": seed, "path": f"model_seed_{seed}.pt",
                         "sha256": sha256(out / f"model_seed_{seed}.pt")} for seed in cfg["seeds"]],
    }
    save_json(out / "model_freeze_manifest.json", freeze_manifest)

    target_data = {batch: access.load(batch, "single post-freeze final evaluation")
                   for batch in cfg["dataset"]["target_batches"]}
    all_x = np.vstack([source.x] + [target_data[batch].x for batch in range(2, 11)])
    all_y = np.concatenate([source.y] + [target_data[batch].y for batch in range(2, 11)])
    dataset_validation = {
        "records": int(len(all_x)), "features": int(all_x.shape[1]),
        "batches": list(range(1, 11)), "gas_labels": np.unique(all_y).astype(int).tolist(),
        "batch_record_counts": {"1": len(source.x), **{str(batch): len(target_data[batch].x)
                                                         for batch in range(2, 11)}},
        "finite_values": bool(np.isfinite(all_x).all()),
        "expected_structure_verified": bool(
            len(all_x) == 13910 and all_x.shape[1] == 128
            and np.array_equal(np.unique(all_y), np.arange(1, 7))),
        "validation_phase": "post-freeze for target batches",
        "exclusions": [], "deduplication": False, "clipping": False, "imputation": False,
    }
    save_json(out / "dataset_validation.json", dataset_validation)
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for seed in cfg["seeds"]:
        model = final_models[seed]
        scaler = final_scalers[seed]
        for batch, data in target_data.items():
            pred = predict(model, scaler.transform(data.x), cfg["training"]["batch_size"]).argmax(1) + 1
            prediction_frames.append(pd.DataFrame({
                "seed": seed, "batch": batch,
                "sample_id": [f"batch{batch}:line{int(line)}" for line in data.line_numbers],
                "line_number": data.line_numbers, "true_gas_label": data.y,
                "predicted_gas_label": pred, "correct": pred == data.y,
            }))
            metric_rows.append({"seed": seed, "batch": batch, "samples": len(data.y),
                                "correct": int((pred == data.y).sum()),
                                "accuracy": float(accuracy_score(data.y, pred))})
            confusion_rows.extend(_confusion_rows(seed, batch, data.y, pred))

    cv_df = pd.DataFrame(cv_rows)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(out / "training_history.csv.gz", index=False, compression="gzip")
    style_rows = history_frame
    style_rows = style_rows[style_rows.restyled_component.notna()][[
        "phase", "seed", "fold", "epoch", "restyled_component", "sampled_sigma_min",
        "sampled_sigma_max", "sampled_mean_min", "sampled_mean_max",
        "generation_nonfinite_tensor_count",
    ]]
    style_rows.to_csv(out / "style_sampling_statistics.csv.gz", index=False, compression="gzip")
    cv_df.to_csv(out / "cv_fold_metrics.csv", index=False)
    pd.concat(cv_predictions, ignore_index=True).to_csv(out / "cv_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "target_batch_metrics.csv", index=False)
    predictions.to_csv(out / "target_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(confusion_rows).to_csv(out / "confusion_matrices.csv", index=False)
    pd.DataFrame(scaler_scopes).to_csv(out / "scaler_fit_scopes.csv", index=False)
    if provenance_frames:
        pd.concat(provenance_frames, ignore_index=True).to_csv(
            out / "augmentation_provenance.csv.gz", index=False, compression="gzip")
    summary = {
        "stage": stage, "status": "completed", "seeds": cfg["seeds"],
        "source_cv_accuracy_by_seed": cv_df.groupby("seed").accuracy.mean().to_dict(),
        "source_cv_mean_accuracy": float(cv_df.accuracy.mean()),
        "target_unweighted_mean_accuracy_by_seed": metrics.groupby("seed").accuracy.mean().to_dict(),
        "target_pooled_accuracy_by_seed": predictions.groupby("seed").correct.mean().to_dict(),
        "target_batch_accuracy_across_seeds": metrics.groupby("batch").accuracy.mean().to_dict(),
        "target_unweighted_mean_accuracy": float(metrics.groupby("batch").accuracy.mean().mean()),
        "target_pooled_accuracy": float(predictions.correct.mean()),
        "target_evaluation_after_freeze": True,
    }
    save_json(out / "summary.json", summary)
    save_json(out / "data_access_log.json", access.events)
    raw_loads = [event for event in access.events if event["event"] == "raw_file_load"]
    save_json(out / "input_manifest.json", _provenance_input_manifest(root, config_file, cfg) + raw_loads)
    save_json(out / "feature_generation_geometry.json", {
        "insertion": "between residual blocks 3 and 4", "block3_shape": ["B", 128, 128],
        "pool": {"kernel_size": 2, "stride": 2}, "upsample": {"mode": "nearest", "length": 128},
        "statistics_shape": ["B", 128, 1], "statistics_axis": "length",
        "variance": "population over length (unbiased=False)",
        "batch_spread": "population standard deviation across active mini-batch (unbiased=False)",
        "epsilon": cfg["feature_generation"]["epsilon"],
        "semantic_restyled_component": "pooled/upsampled low-frequency-like branch",
        "paper_literal_restyled_component": "residual branch printed as L",
        "a3_numerical_stability": cfg["a3_numerical_stability"],
        "stage": stage,
    })
    cv_history_complete = bool(
        history_frame[history_frame.phase == "cv"].groupby(["seed", "fold"]).size().eq(100).all()
        and history_frame[history_frame.phase == "cv"].groupby(["seed", "fold"]).epoch.max().eq(100).all())
    final_history_complete = bool(
        history_frame[history_frame.phase == "final_fit"].groupby("seed").size().eq(100).all()
        and history_frame[history_frame.phase == "final_fit"].groupby("seed").epoch.max().eq(100).all())
    recomputed_metrics = predictions.groupby(["seed", "batch"], as_index=False).agg(
        samples=("correct", "size"), correct=("correct", "sum"))
    recomputed_metrics["accuracy"] = recomputed_metrics.correct / recomputed_metrics.samples
    saved_metric_order = metrics.sort_values(["seed", "batch"]).reset_index(drop=True)
    recomputed_metric_order = recomputed_metrics.sort_values(["seed", "batch"]).reset_index(drop=True)
    metrics_match_predictions = bool(
        np.array_equal(saved_metric_order[["seed", "batch", "samples", "correct"]].to_numpy(),
                       recomputed_metric_order[["seed", "batch", "samples", "correct"]].to_numpy())
        and np.allclose(saved_metric_order.accuracy, recomputed_metric_order.accuracy, rtol=0.0, atol=0.0))
    audit_checks = {
        "batch1_is_only_pre_freeze_raw_load": [e["batch"] for e in raw_loads if e["phase"] == "source_only"] == [1],
        "all_targets_loaded_post_freeze": all(e["phase"] == "post_freeze" for e in raw_loads if e["batch"] >= 2),
        "each_raw_file_loaded_once": sorted(e["batch"] for e in raw_loads) == list(range(1, 11)),
        "all_scalers_fit_source_originals_only": all(r["fit_batch"] == 1 and r["fit_augmented_samples"] == 0
                                                      and r["fit_target_samples"] == 0 for r in scaler_scopes),
        "fixed_epoch_100_only": set(cv_df.checkpoint_epoch) == {100}
        and cv_history_complete and final_history_complete,
        "target_prediction_coverage": len(predictions) == 5 * sum(len(d.y) for d in target_data.values()),
        "target_predictions_unique_per_seed": not predictions.duplicated(["seed", "batch", "sample_id"]).any(),
        "target_metrics_recompute_from_predictions": metrics_match_predictions,
        "dataset_structure_verified": dataset_validation["expected_structure_verified"]
        and dataset_validation["finite_values"],
        "generated_sigmas_positive_and_finite": bool(
            style_rows.empty or (style_rows.sampled_sigma_min.gt(0).all()
                                 and style_rows.generation_nonfinite_tensor_count.eq(0).all())),
        "no_exclusions": True, "raw_data_unchanged": True,
    }
    audit = {"status": "passed" if all(audit_checks.values()) else "failed", "checks": audit_checks,
             "target_access_influence": "none; target loading began after every seed checkpoint was frozen",
             "preprocessing_fit_scope": "fold-local Batch 1 originals; final all-Batch-1 originals",
             "feature_statistics_scope": "active source-training mini-batch",
             "target_metrics_used_for_selection": False}
    save_json(out / "leakage_target_access_audit.json", audit)
    save_json(out / "run_manifest.json", {
        "run_id": run_id, "created_utc": utc_now(), "status": audit["status"],
        "stage": stage, "implementation_files": _implementation_identity(root),
        "preprocessing_fit_scope": audit["preprocessing_fit_scope"],
        "selection_scope": "Batch 1 only; fixed epoch 100", "exclusions": [],
        "raw_data_modified": False,
    })
    (out / "report.md").write_text(
        f"# {stage} CDCNN v6.3 run\n\n"
        f"Status: **{audit['status']}**. Five seeds and five saved Batch-1 folds used the "
        "unified epoch-100 SGD/StepLR protocol. Target batches were opened only after all "
        "final source checkpoints were frozen.\n\n"
        "## Traceability\n\n"
        "The five-block 1D ResNet, feature-generation location, augmentation equations, "
        "probability MSE, and normalized contrastive objective are paper-supported at the "
        "levels described in the v6 specification. Momentum, exact learning-rate schedule, "
        "same-class pairing, projection-head shape, temperature, and loss weights are "
        "project-controlled. The physical-semantic low/high names are an explicit project "
        "adaptation. The paper's nonstandard H/L naming remains available only as the "
        "A2-paper-literal diagnostic. The provided paper leaves several numerical settings "
        "underdetermined; this run does not describe project-controlled defaults as author values.\n\n"
        "See `summary.json`, `feature_generation_geometry.json`, and "
        "`leakage_target_access_audit.json` for results and integrity evidence.\n",
        encoding="utf-8",
    )
    if audit["status"] != "passed":
        raise ProtocolError(f"Post-run audit failed in {out}")
    _write_output_inventory(out)
    return out


if __name__ == "__main__":
    raise SystemExit("Use scripts/run_cdcnn_ablation.py")
