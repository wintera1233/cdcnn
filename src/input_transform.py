"""Input transforms applied after the source-fitted scaler.

Every transform here is parameter-free: it fits nothing, so it cannot carry
information between samples or batches and adds no leakage surface. The scaler
that runs before it is still fitted on Batch 1 training rows only.

Motivation: under the Batch-1 `StandardScaler`, drifted target rows reach extreme
values (Batch 2 contains |z| near 1.5e4), and the A3 confound ablation showed
that per-sample normalization inside the network (LayerNorm) is worth +0.0845
target mean on its own. These transforms test the same idea at the input.
"""

from __future__ import annotations

import numpy as np

from src.protocol import ProtocolError


# Fixed before training; never tuned against any target batch.
CLIP_LIMIT = 5.0
EPSILON = 1e-8

INPUT_TRANSFORM_KINDS = ("identity", "per_sample", "signed_log", "clip")

# Stage -> transform. Stages absent from this mapping use "identity", which is
# the exact pre-v6.4 behaviour.
STAGE_INPUT_TRANSFORMS = {
    "B0-PS": "per_sample",
    "B0-LN-PS": "per_sample",
    "B0-LN-LOG": "signed_log",
    "B0-LN-CLIP": "clip",
    # v6.5: the confound ladder rebuilt on per-sample-normalized inputs.
    "B0-stab-PS": "per_sample",
    "A2-stab-PS": "per_sample",
    "A3-PS": "per_sample",
    # v6.6: augmentation isolated, and the paper's literal restyled branch.
    "A1-stab-PS": "per_sample",
    "A2-lit-PS": "per_sample",
    "A3-lit-PS": "per_sample",
    # v6.7: A1 augmentation at reduced noise magnitude.
    "A1-PS-s50": "per_sample",
    "A1-PS-s20": "per_sample",
    "A1-PS-s05": "per_sample",
    # v6.8: duplication control, scale 0.0.
    "A1-PS-s00": "per_sample",
}


def input_transform_for(stage: str) -> str:
    return STAGE_INPUT_TRANSFORMS.get(stage, "identity")


def apply_input_transform(kind: str, x: np.ndarray, clip_limit: float = CLIP_LIMIT) -> np.ndarray:
    """Apply one parameter-free transform to already-scaled rows."""
    if kind == "identity":
        return x
    if kind == "per_sample":
        # Standardize each row against its own 128 features, so a batch-wide
        # offset or gain cannot move the representation.
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, ddof=0, keepdims=True)
        return (x - mean) / (std + EPSILON)
    if kind == "signed_log":
        # Compress extreme magnitudes while preserving sign and ordering.
        return np.sign(x) * np.log1p(np.abs(x))
    if kind == "clip":
        return np.clip(x, -clip_limit, clip_limit)
    raise ProtocolError(f"Unknown input transform {kind!r}; expected one of {INPUT_TRANSFORM_KINDS}")


def prepare_inputs(stage: str, scaler, x: np.ndarray) -> np.ndarray:
    """Scale with the source-fitted scaler, then apply the stage's transform."""
    return apply_input_transform(input_transform_for(stage), scaler.transform(x))


def prepare_inputs_from_params(kind: str, mean: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Frozen-checkpoint equivalent of `prepare_inputs` for target evaluation."""
    return apply_input_transform(kind, (x - mean) / scale)
