#!/usr/bin/env python3
"""Plot one count/row-normalized confusion matrix per evaluated batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream_run", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    predictions_path = args.upstream_run / "predictions.csv.gz"
    configuration_path = args.upstream_run / "configuration.json"
    if not predictions_path.is_file() or not configuration_path.is_file():
        raise FileNotFoundError("The upstream run must contain predictions.csv.gz and configuration.json")

    predictions = pd.read_csv(predictions_path)
    required = {"batch", "true_gas_label", "predicted_gas_label"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    upstream_config = json.loads(configuration_path.read_text(encoding="utf-8"))
    gas_mapping = upstream_config["resolved_dataset_configuration"]["dataset"]["gas_mapping"]
    optimizer_name = upstream_config.get("training", {}).get("optimizer", "unspecified optimizer")
    labels = sorted(int(label) for label in gas_mapping)
    display_names = [gas_mapping[str(label)] for label in labels]
    batches = sorted(int(batch) for batch in predictions["batch"].unique())

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / f"{stamp}_1d_resnet_confusion_matrices"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    matrix_rows = []
    output_files = []
    for batch in batches:
        frame = predictions[predictions["batch"] == batch]
        matrix = pd.crosstab(frame["true_gas_label"], frame["predicted_gas_label"]).reindex(
            index=labels, columns=labels, fill_value=0
        ).to_numpy(dtype=int)
        row_totals = matrix.sum(axis=1, keepdims=True)
        percentages = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals != 0)
        annotations = np.array(
            [[f"{matrix[i, j]}\n{percentages[i, j]:.1%}" for j in range(len(labels))]
             for i in range(len(labels))]
        )

        accuracy = float(np.trace(matrix) / matrix.sum())
        fig, ax = plt.subplots(figsize=(9.5, 7.5))
        sns.heatmap(percentages, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=display_names, yticklabels=display_names, square=True,
                    linewidths=.5, linecolor="white", cbar_kws={"label": "Within-true-class proportion"}, ax=ax)
        ax.set_title(
            f"Canonical 1D ResNet B0 ({optimizer_name}) — Batch {batch}\n"
            f"N={len(frame):,}; accuracy={accuracy:.1%}"
        )
        ax.set_xlabel("Predicted gas")
        ax.set_ylabel("True gas")
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        name = f"batch_{batch:02d}_confusion_matrix.png"
        fig.savefig(output_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        output_files.append(name)

        for i, true_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                matrix_rows.append({"batch": batch, "true_gas_label": true_label,
                                    "predicted_gas_label": predicted_label,
                                    "count": int(matrix[i, j]), "row_proportion": percentages[i, j]})

    matrix_name = "confusion_matrices_with_row_proportions.csv"
    pd.DataFrame(matrix_rows).to_csv(output_dir / matrix_name, index=False)
    output_files.append(matrix_name)

    configuration = {
        "purpose": "Per-batch confusion-matrix visualization of an existing model run",
        "upstream_run": str(args.upstream_run),
        "evaluated_batches": batches,
        "class_order": labels,
        "gas_mapping": gas_mapping,
        "optimizer": optimizer_name,
        "annotations": "count and percentage normalized within each true class",
        "color_scale": "row proportion, fixed from 0 to 1 for cross-batch comparison",
        "preprocessing_fit_scope": "No fitting; plots use saved upstream predictions only",
        "exclusions": [],
    }
    write_json(output_dir / "configuration.json", configuration)
    write_json(output_dir / "input_manifest.json", {
        "inputs": [
            {"path": str(predictions_path), "sha256": sha256(predictions_path), "rows": len(predictions)},
            {"path": str(configuration_path), "sha256": sha256(configuration_path)},
        ]
    })
    write_json(output_dir / "software_versions.json", {
        "python": platform.python_version(), "pandas": pd.__version__,
        "numpy": np.__version__, "matplotlib": matplotlib.__version__, "seaborn": sns.__version__,
    })
    output_files += ["configuration.json", "input_manifest.json", "software_versions.json"]
    write_json(output_dir / "output_inventory.json", {
        "outputs": [{"path": name, "sha256": sha256(output_dir / name)} for name in output_files]
    })
    write_json(output_dir / "run_manifest.json", {
        "run_id": output_dir.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete", "upstream_run": str(args.upstream_run), "outputs": len(output_files) + 1,
    })
    print(output_dir)


if __name__ == "__main__":
    main()
