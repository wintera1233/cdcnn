#!/usr/bin/env python3
"""Run frozen CDCNN v6.3 A3 inference on Batch-2 GAS4 samples only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cdcnn_ablation import CDCNNModel, predict
from src.pca_analysis import load_batch


IMPLEMENTATION_VERSION = "CDCNN_v6.3_A3_numerical_stabilization"
STAGE = "A3"
TARGET_BATCH = 2
TARGET_LABEL = 4
GAS_NAMES = {
    # Verified against the paper's Table 2 counts in all ten batches;
    # see docs/per-class-failure.md. The earlier mapping here was wrong.
    1: "Acetone",
    2: "Acetaldehyde",
    3: "Ethanol",
    4: "Ethylene",
    5: "Ammonia",
    6: "Toluene",
}


class InferenceError(RuntimeError):
    """Raised when a frozen-input or inference invariant is violated."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-run",
        default="runs/20260911T092326995583Z_cdcnn_v6_3_full",
        help="Completed v6.3 full-run directory containing frozen source checkpoints.",
    )
    parser.add_argument("--output-root", default="runs")
    parser.add_argument(
        "--alias",
        default="runs/GAS4_predict",
        help="Non-overwritten directory symlink created after successful inference.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InferenceError(message)


def checkpoint_rows(root: Path, full_run: Path, cfg: dict) -> list[dict]:
    freeze_path = full_run / "global_freeze_manifest.json"
    freeze = read_json(freeze_path)
    require(freeze["status"] == "all_source_runs_frozen", "Global freeze manifest did not pass")
    require(freeze["target_batches_loaded_at_freeze"] == [], "Target data was loaded before freeze")
    rows = [row for row in freeze["checkpoints"] if row["stage"] == STAGE]
    by_seed = {int(row["seed"]): row for row in rows}
    seeds = [int(seed) for seed in cfg["seeds"]]
    require(set(by_seed) == set(seeds), f"Frozen A3 seed coverage mismatch: {sorted(by_seed)}")
    result = []
    for seed in seeds:
        row = dict(by_seed[seed])
        path = root / row["checkpoint"]
        require(path.is_file(), f"Missing checkpoint: {path}")
        actual_hash = sha256(path)
        require(actual_hash == row["sha256"], f"Checkpoint hash mismatch: {path}")

        source_run = root / row["source_run"]
        source_manifest = read_json(source_run / "run_manifest.json")
        implementation_rows = {
            item["path"]: item["sha256"] for item in source_manifest["implementation_files"]
        }
        current_implementation = root / "src/cdcnn_ablation.py"
        require(
            sha256(current_implementation) == implementation_rows["src/cdcnn_ablation.py"],
            "Current CDCNN implementation differs from the code recorded with the checkpoint",
        )
        row["resolved_checkpoint"] = str(path)
        result.append(row)
    return result


def load_frozen_model(checkpoint_path: Path, expected_seed: int) -> tuple[CDCNNModel, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(payload.get("stage") == STAGE, f"Checkpoint is not {STAGE}: {checkpoint_path}")
    require(payload.get("seed") == expected_seed, f"Checkpoint seed mismatch: {checkpoint_path}")
    require(payload.get("epoch") == 100, f"Checkpoint is not frozen at epoch 100: {checkpoint_path}")
    require(
        payload.get("implementation_version") == IMPLEMENTATION_VERSION,
        f"Checkpoint is not {IMPLEMENTATION_VERSION}: {checkpoint_path}",
    )
    require(payload.get("source_batch") == 1, "Checkpoint source batch is not Batch 1")
    require(payload.get("target_batches_loaded") == [], "Checkpoint records pre-freeze target access")
    mean = np.asarray(payload["scaler_mean"], dtype=np.float64)
    scale = np.asarray(payload["scaler_scale"], dtype=np.float64)
    require(mean.shape == (128,) and scale.shape == (128,), "Frozen scaler has the wrong shape")
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), "Frozen scaler is non-finite")
    require(bool((scale > 0).all()), "Frozen scaler contains a non-positive scale")

    model = CDCNNModel(
        STAGE,
        payload["feature_generation"]["epsilon"],
        payload["loss"]["projection_dimension"],
        payload["a3_numerical_stability"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def probability_columns() -> list[str]:
    return [f"prob_gas{label}_{GAS_NAMES[label].lower()}" for label in GAS_NAMES]


def write_inventory(out: Path) -> None:
    rows = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(out.iterdir())
        if path.name != "output_inventory.json"
    ]
    rows.append(
        {
            "path": "output_inventory.json",
            "bytes": None,
            "sha256": None,
            "note": "Self-entry omits recursive size and hash.",
        }
    )
    write_json(out / "output_inventory.json", rows)


def main() -> Path:
    args = parse_args()
    require(args.batch_size > 0, "--batch-size must be positive")
    root = Path.cwd().resolve()
    full_run = (root / args.full_run).resolve()
    output_root = (root / args.output_root).resolve()
    alias = (root / args.alias).absolute()
    require(full_run.is_dir(), f"Full run does not exist: {full_run}")
    require(not alias.exists() and not alias.is_symlink(), f"Refusing to overwrite alias: {alias}")

    controller = read_json(full_run / "controller_result.json")
    summary = read_json(full_run / "experiment_summary.json")
    require(controller["status"] == "completed", "Full v6.3 run is not completed")
    require(summary["status"] == "passed", "Full v6.3 run did not pass")
    require(summary["implementation_version"] == IMPLEMENTATION_VERSION, "Full run is not v6.3")
    orchestration = read_json(full_run / "orchestration_config.json")
    config_path = root / orchestration["config_path"]
    require(
        sha256(config_path) == orchestration["config_sha256"],
        "Current protocol configuration differs from the full-run configuration",
    )
    cfg = read_json(config_path)
    seeds = [int(seed) for seed in orchestration["seeds"]]
    require(seeds == [int(seed) for seed in cfg["seeds"]], "Seed order mismatch")
    checkpoints = checkpoint_rows(root, full_run, cfg)

    dataset_cfg = cfg["dataset"]
    batch_path = root / dataset_cfg["path"] / dataset_cfg["batch_file_pattern"].format(
        batch_id=TARGET_BATCH
    )
    batch_hash = sha256(batch_path)
    require(
        batch_hash == dataset_cfg["expected_hashes"][str(TARGET_BATCH)],
        f"Batch {TARGET_BATCH} hash mismatch",
    )
    x, y, line_numbers = load_batch(batch_path, TARGET_BATCH, dataset_cfg["feature_count"])
    selected = y == TARGET_LABEL
    gas4_x = x[selected]
    gas4_y = y[selected]
    gas4_lines = line_numbers[selected]
    require(len(gas4_x) > 0, "Batch 2 contains no GAS4 samples")
    require(gas4_x.shape[1] == 128, "Selected data does not contain 128 features")
    require(np.isfinite(gas4_x).all(), "Selected GAS4 features contain non-finite values")
    require(np.array_equal(np.unique(gas4_y), np.array([TARGET_LABEL])), "Non-GAS4 row selected")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_GAS4_predict"
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=False)

    per_seed_arrays: list[np.ndarray] = []
    per_seed_frames: list[pd.DataFrame] = []
    checkpoint_inputs: list[dict] = []
    prob_columns = probability_columns()
    sample_ids = np.asarray([f"batch{TARGET_BATCH}:line{int(line)}" for line in gas4_lines])

    for row in checkpoints:
        seed = int(row["seed"])
        checkpoint_path = Path(row["resolved_checkpoint"])
        model, payload = load_frozen_model(checkpoint_path, seed)
        scaled = (gas4_x - payload["scaler_mean"]) / payload["scaler_scale"]
        require(np.isfinite(scaled).all(), f"Seed {seed} standardized inputs are non-finite")
        logits = predict(model, scaled, args.batch_size)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        require(probabilities.shape == (len(gas4_x), 6), f"Seed {seed} output shape is invalid")
        require(np.isfinite(probabilities).all(), f"Seed {seed} probabilities are non-finite")
        require(bool((probabilities >= 0).all()), f"Seed {seed} probabilities are negative")
        require(
            np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6),
            f"Seed {seed} probabilities do not sum to one",
        )
        predicted = probabilities.argmax(axis=1) + 1
        frame = pd.DataFrame(
            {
                "seed": seed,
                "batch": TARGET_BATCH,
                "sample_id": sample_ids,
                "line_number": gas4_lines,
                "true_gas_label": gas4_y,
                "true_gas_name": GAS_NAMES[TARGET_LABEL],
                "predicted_gas_label": predicted,
                "predicted_gas_name": [GAS_NAMES[int(label)] for label in predicted],
            }
        )
        frame[prob_columns] = probabilities
        per_seed_arrays.append(probabilities)
        per_seed_frames.append(frame)
        checkpoint_inputs.append(
            {
                "role": "frozen_A3_v6.3_checkpoint",
                "seed": seed,
                "path": str(checkpoint_path.relative_to(root)),
                "sha256": row["sha256"],
                "frozen_utc": row["frozen_utc"],
                "epoch": int(payload["epoch"]),
                "scaler_fit_scope": "Batch 1 originals only; loaded from checkpoint without fitting",
            }
        )
        del model, payload, logits

    per_seed = np.stack(per_seed_arrays, axis=0).astype(np.float32, copy=False)
    ensemble = per_seed.mean(axis=0, dtype=np.float64)
    ensemble_std = per_seed.std(axis=0, dtype=np.float64)
    ensemble_predicted = ensemble.argmax(axis=1) + 1
    require(np.allclose(ensemble.sum(axis=1), 1.0, rtol=0.0, atol=1e-6), "Mean probabilities do not sum to one")

    prior_path = full_run / "target_predictions.csv.gz"
    prior = pd.read_csv(prior_path)
    prior = prior[
        (prior["stage"] == STAGE)
        & (prior["batch"] == TARGET_BATCH)
        & (prior["true_gas_label"] == TARGET_LABEL)
    ][["seed", "sample_id", "predicted_gas_label"]]
    replay = pd.concat(per_seed_frames, ignore_index=True)[
        ["seed", "sample_id", "predicted_gas_label"]
    ]
    comparison = replay.merge(
        prior,
        on=["seed", "sample_id"],
        suffixes=("_replay", "_original_run"),
        validate="one_to_one",
    )
    require(len(comparison) == len(seeds) * len(gas4_x), "Prior replay coverage mismatch")
    require(
        bool(
            comparison["predicted_gas_label_replay"].eq(
                comparison["predicted_gas_label_original_run"]
            ).all()
        ),
        "Frozen checkpoint replay disagrees with original run predictions",
    )

    np.save(out / "softmax_probabilities_by_seed.npy", per_seed, allow_pickle=False)
    np.save(out / "softmax_probabilities_mean.npy", ensemble, allow_pickle=False)
    np.save(out / "softmax_probabilities_std.npy", ensemble_std, allow_pickle=False)
    pd.concat(per_seed_frames, ignore_index=True).to_csv(
        out / "softmax_probabilities_by_seed.csv.gz", index=False, compression="gzip"
    )
    ensemble_frame = pd.DataFrame(
        {
            "batch": TARGET_BATCH,
            "sample_id": sample_ids,
            "line_number": gas4_lines,
            "true_gas_label": gas4_y,
            "true_gas_name": GAS_NAMES[TARGET_LABEL],
            "predicted_gas_label": ensemble_predicted,
            "predicted_gas_name": [GAS_NAMES[int(label)] for label in ensemble_predicted],
        }
    )
    ensemble_frame[prob_columns] = ensemble
    ensemble_frame.to_csv(out / "softmax_probabilities_mean.csv", index=False)
    pd.DataFrame(
        {
            "array_row": np.arange(len(gas4_x)),
            "sample_id": sample_ids,
            "line_number": gas4_lines,
            "true_gas_label": gas4_y,
            "true_gas_name": GAS_NAMES[TARGET_LABEL],
        }
    ).to_csv(out / "sample_index.csv", index=False)

    per_seed_accuracy = {
        str(seed): float((per_seed[index].argmax(axis=1) + 1 == TARGET_LABEL).mean())
        for index, seed in enumerate(seeds)
    }
    predicted_counts = {
        str(label): int((ensemble_predicted == label).sum()) for label in GAS_NAMES
    }
    axes = {
        "softmax_probabilities_by_seed.npy": {
            "shape": list(per_seed.shape),
            "axes": ["seed", "sample", "gas_class"],
            "seed_order": seeds,
        },
        "softmax_probabilities_mean.npy": {
            "shape": list(ensemble.shape),
            "axes": ["sample", "gas_class"],
            "definition": "arithmetic mean of the five per-seed softmax probability vectors",
        },
        "softmax_probabilities_std.npy": {
            "shape": list(ensemble_std.shape),
            "axes": ["sample", "gas_class"],
            "definition": "population standard deviation across the five seed models",
        },
        "gas_class_order": [
            {"array_index": label - 1, "gas_label": label, "gas_name": GAS_NAMES[label]}
            for label in GAS_NAMES
        ],
        "sample_order": "sample_index.csv; original ascending Batch-2 line order",
        "dtype": {"by_seed": str(per_seed.dtype), "mean": str(ensemble.dtype), "std": str(ensemble_std.dtype)},
    }
    write_json(out / "array_axes.json", axes)
    write_json(
        out / "summary.json",
        {
            "stage": STAGE,
            "implementation_version": IMPLEMENTATION_VERSION,
            "batch": TARGET_BATCH,
            "selected_true_gas_label": TARGET_LABEL,
            "selected_true_gas_name": GAS_NAMES[TARGET_LABEL],
            "batch_total_rows_loaded_for_label_selection": int(len(x)),
            "model_input_rows": int(len(gas4_x)),
            "non_gas4_model_input_rows": 0,
            "seeds": seeds,
            "per_seed_accuracy": per_seed_accuracy,
            "mean_probability_ensemble_accuracy": float((ensemble_predicted == TARGET_LABEL).mean()),
            "mean_probability_ensemble_predicted_counts": predicted_counts,
            "probability_rows_finite_nonnegative_sum_to_one": True,
            "original_run_argmax_replay_exact": True,
            "training_performed": False,
            "scaler_fitting_performed": False,
            "parameter_updates_performed": False,
        },
    )
    write_json(
        out / "input_manifest.json",
        [
            {
                "role": "Batch-2 raw dataset loaded only to select GAS4 rows",
                "path": str(batch_path.relative_to(root)),
                "sha256": batch_hash,
                "records": int(len(x)),
                "selected_records": int(len(gas4_x)),
            },
            {
                "role": "completed v6.3 full-run manifest",
                "path": str((full_run / "global_freeze_manifest.json").relative_to(root)),
                "sha256": sha256(full_run / "global_freeze_manifest.json"),
            },
            {
                "role": "original-run prediction replay reference",
                "path": str(prior_path.relative_to(root)),
                "sha256": sha256(prior_path),
            },
            *checkpoint_inputs,
        ],
    )
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        git_revision = None
    write_json(
        out / "run_manifest.json",
        {
            "run_id": run_id,
            "created_utc": utc_now(),
            "status": "completed",
            "kind": "inference_only",
            "command": " ".join([sys.executable, *sys.argv]),
            "git_revision": git_revision,
            "upstream_full_run": str(full_run.relative_to(root)),
            "requested_alias": str(alias.relative_to(root)),
            "model_mode": "eval",
            "autograd": "disabled by inference function",
            "execution_device": "cpu",
            "training_performed": False,
            "scaler_fitting_performed": False,
            "raw_data_modified": False,
            "exclusions": "Only non-GAS4 rows were excluded from model input by explicit request.",
        },
    )
    (out / "report.md").write_text(
        "# A3 v6.3 Batch-2 Acetaldehyde inference\n\n"
        f"All {len(gas4_x)} GAS4 (Acetaldehyde) records in Batch 2 were evaluated by the "
        f"five frozen A3 v6.3 epoch-100 checkpoints (seeds {seeds}). No training, scaler "
        "fitting, or parameter update occurred. The primary probability array is "
        "`softmax_probabilities_mean.npy`; its rows map through `sample_index.csv`, and its "
        "six columns follow `array_axes.json`. Per-seed probabilities are retained separately.\n\n"
        f"The mean-probability ensemble classified {(ensemble_predicted == TARGET_LABEL).sum()} of "
        f"{len(gas4_x)} records as Acetaldehyde "
        f"({(ensemble_predicted == TARGET_LABEL).mean():.6f} accuracy). Every per-seed argmax "
        "exactly replayed the corresponding prediction stored by the original completed full run.\n",
        encoding="utf-8",
    )
    write_inventory(out)

    alias.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(out, start=alias.parent)
    alias.symlink_to(relative_target, target_is_directory=True)
    print(out)
    return out


if __name__ == "__main__":
    main()
