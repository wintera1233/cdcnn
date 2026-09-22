"""Target evaluation, run once after every checkpoint is frozen.

The target mean is the unweighted mean of the nine per-batch accuracies, the
statistic the paper's Table 3 reports; see `CLAUDE.md` section 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src import normalize
from src.data import (GAS_LABELS, N_CLASSES, TARGET_BATCHES, load_source,
                      load_target)
from src.model import build, to_input
from src.protocol import ProtocolError, TargetAccessLog


def load_checkpoint(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build(payload["variant"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


@torch.no_grad()
def predict(model, normalizer: dict, x: np.ndarray, device: str,
            batch_size: int = 512) -> np.ndarray:
    inputs = to_input(normalize.apply(normalizer, x))
    predictions = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start:start + batch_size].to(device)
        predictions.append(model(chunk).argmax(dim=1).cpu().numpy())
    # Back to the dataset's 1..6 labels.
    return np.concatenate(predictions) + 1


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for true, predicted in zip(y_true, y_pred):
        matrix[true - 1, predicted - 1] += 1
    return matrix.tolist()


def evaluate_checkpoint(path: Path, access_log: TargetAccessLog, device: str) -> dict:
    """Score one frozen checkpoint on Batch 1 and on every target batch."""
    if not isinstance(access_log, TargetAccessLog):
        raise ProtocolError("target evaluation requires a TargetAccessLog")
    model, payload = load_checkpoint(path, device)
    normalizer = payload["normalizer"]

    source_x, source_y = load_source()
    source_accuracy = float(
        (predict(model, normalizer, source_x, device) == source_y).mean())

    per_batch: dict[str, float] = {}
    matrices: dict[str, list[list[int]]] = {}
    pooled_correct = pooled_total = 0
    for index in TARGET_BATCHES:
        x, y = load_target(index, access_log)
        predicted = predict(model, normalizer, x, device)
        per_batch[str(index)] = float((predicted == y).mean())
        matrices[str(index)] = confusion(y, predicted)
        pooled_correct += int((predicted == y).sum())
        pooled_total += len(y)

    return {"checkpoint": str(path), "variant": payload["variant"],
            "seed": payload["seed"], "epoch": payload["epoch"],
            "learning_rate": payload.get("learning_rate"),
            "cell": cell_name(payload["variant"], payload.get("learning_rate")),
            "source_accuracy": source_accuracy,
            "per_batch_accuracy": per_batch,
            "target_mean": float(np.mean(list(per_batch.values()))),
            "pooled_accuracy": pooled_correct / pooled_total,
            "confusion": matrices}


def cell_name(variant: str, learning_rate) -> str:
    """One design cell: a variant at a learning rate."""
    return variant if learning_rate is None else f"{variant}@lr{learning_rate:g}"


def aggregate(results: list[dict]) -> dict:
    """Mean and standard deviation of the target mean across seeds, per cell."""
    by_variant: dict[str, list[dict]] = {}
    for result in results:
        by_variant.setdefault(result.get("cell") or result["variant"], []).append(result)
    summary = {}
    for variant, entries in by_variant.items():
        means = np.array([entry["target_mean"] for entry in entries], dtype=np.float64)
        batches = {key: float(np.mean([entry["per_batch_accuracy"][key]
                                       for entry in entries]))
                   for key in entries[0]["per_batch_accuracy"]}
        summary[variant] = {
            "variant": entries[0]["variant"],
            "learning_rate": entries[0].get("learning_rate"),
            "seeds": sorted(entry["seed"] for entry in entries),
            "target_mean": float(means.mean()),
            # Sample standard deviation: three seeds, so ddof=1.
            "target_mean_sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "target_mean_per_seed": {str(entry["seed"]): entry["target_mean"]
                                     for entry in sorted(entries, key=lambda e: e["seed"])},
            "source_accuracy": float(np.mean([entry["source_accuracy"]
                                              for entry in entries])),
            "pooled_accuracy": float(np.mean([entry["pooled_accuracy"]
                                              for entry in entries])),
            "per_batch_accuracy": batches}
    return summary


def epoch_curves(run_dir: Path, access_log: TargetAccessLog, device: str) -> dict:
    """Batch 1 and target accuracy at every saved epoch, for Fig. S1.

    The target batches are parsed once and held, because re-parsing nine files
    for each of four hundred checkpoints would dominate the runtime. Every
    checkpoint must already be frozen: `access_log` carries the freeze events, so
    `src.audit.leakage_audit` still sees the freezes preceding these accesses.
    """
    if not access_log.freezes:
        raise ProtocolError("epoch curves require frozen checkpoints")
    source_x, source_y = load_source()
    targets = {index: load_target(index, access_log) for index in TARGET_BATCHES}

    curves: dict[str, list[dict]] = {}
    for directory in sorted((run_dir / "checkpoints").iterdir()):
        epochs = sorted(directory.glob("epoch_*.pt"))
        if not epochs:
            continue
        points = []
        for path in epochs:
            model, payload = load_checkpoint(path, device)
            normalizer = payload["normalizer"]
            per_batch = {}
            for index, (x, y) in targets.items():
                per_batch[str(index)] = float(
                    (predict(model, normalizer, x, device) == y).mean())
            points.append({
                "epoch": payload["epoch"],
                "source_accuracy": float(
                    (predict(model, normalizer, source_x, device) == source_y).mean()),
                "target_mean": float(np.mean(list(per_batch.values()))),
                "per_batch_accuracy": per_batch})
        curves[directory.name] = points
    return curves


def class_recall(matrix: list[list[int]]) -> dict[str, float]:
    array = np.asarray(matrix, dtype=np.float64)
    totals = array.sum(axis=1)
    return {GAS_LABELS[index + 1]: (float(array[index, index] / totals[index])
                                    if totals[index] else float("nan"))
            for index in range(N_CLASSES)}
