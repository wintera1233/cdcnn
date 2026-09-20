"""A3 stage definition: v6.3 hard bounds, contrastive objective, and clipping.

Everything specific to A3 lives here so the stage can be read and changed on its
own. The shared backbone, augmentation, feature generation, cross-entropy, and
probability MSE remain in `src/cdcnn_ablation.py`, which imports this module.

Nothing here imports the pipeline module: `contrastive_term` and
`clip_a3_gradients` take the model as an argument, so the dependency runs in one
direction only.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.protocol import ProtocolError


# Stages that borrow A3's normalization and, where noted, its hard bounds and
# gradient clipping. The confound-ablation stages exist to separate those
# effects from the contrastive loss; see docs/a3-confound-ablation.md.
LAYERNORM_STAGES = ("A3", "B0-LN", "B0-stab", "A2-stab")
STABILIZED_STAGES = ("A3", "B0-stab", "A2-stab")

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


def contrastive_term(model, outputs: dict, y: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Project both branches into one 2B set and return the A3 contrastive loss.

    The generated view carries the anchor's ground-truth label; no pseudo-label
    is ever produced.
    """
    projected = model.project_contrastive(torch.cat(
        [outputs["original_features"], outputs["generated_features"]], dim=0))
    contrastive_labels = torch.cat([y, y], dim=0)
    return supervised_contrastive_mean(
        projected, contrastive_labels, cfg["loss"]["contrastive_temperature"])


def clip_a3_gradients(model, cfg: dict) -> tuple[float, float] | None:
    """Clip stabilized-stage gradients before the optimizer step; leave baselines untouched."""
    if model.stage not in STABILIZED_STAGES:
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
