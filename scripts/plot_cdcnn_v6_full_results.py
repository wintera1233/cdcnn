#!/usr/bin/env python3
"""Create multi-batch confusion matrices and an accuracy-by-batch plot.

The script is deliberately read-only with respect to the upstream run. It writes
all plots and provenance records to a new, timestamped run directory.
"""

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
from matplotlib.ticker import PercentFormatter


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def load_stage_order(run_dir: Path, confusion: pd.DataFrame) -> list[str]:
    config_path = run_dir / "orchestration_config.json"
    if config_path.is_file():
        configured = json.loads(config_path.read_text(encoding="utf-8")).get("stages", [])
        available = set(confusion["stage"].astype(str))
        ordered = [str(stage) for stage in configured if str(stage) in available]
        if set(ordered) == available:
            return ordered
    return list(dict.fromkeys(confusion["stage"].astype(str)))


def confusion_figure(
    confusion: pd.DataFrame,
    stage: str,
    batches: list[int],
    labels: list[int],
    output_path: Path,
) -> None:
    stage_frame = confusion[confusion["stage"] == stage]
    seeds = sorted(int(seed) for seed in stage_frame["seed"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(22, 21), constrained_layout=True)
    display_labels = [GAS_NAMES.get(label, f"Class {label}") for label in labels]
    image = None
    color_map = matplotlib.colormaps["Blues"].copy()
    color_map.set_bad(color="#eeeeee")

    for ax, batch in zip(axes.flat, batches, strict=True):
        batch_frame = stage_frame[stage_frame["batch"] == batch]
        matrix = (
            batch_frame.groupby(["true_gas_label", "predicted_gas_label"], sort=False)["count"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=labels, columns=labels, fill_value=0)
            .to_numpy(dtype=int)
        )
        row_totals = matrix.sum(axis=1, keepdims=True)
        proportions = np.divide(
            matrix,
            row_totals,
            out=np.full_like(matrix, np.nan, dtype=float),
            where=row_totals != 0,
        )
        annotations = np.empty(matrix.shape, dtype=object)
        for row in range(len(labels)):
            for column in range(len(labels)):
                annotations[row, column] = (
                    "—"
                    if row_totals[row, 0] == 0
                    else f"{proportions[row, column]:.0%}\n(n={matrix[row, column]:,})"
                )

        image = ax.imshow(
            proportions,
            cmap=color_map,
            vmin=0,
            vmax=1,
            aspect="equal",
        )
        ax.set_xticks(np.arange(len(labels)), labels=display_labels)
        ax.set_yticks(np.arange(len(labels)), labels=display_labels)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = proportions[row, column]
                text_color = "white" if np.isfinite(value) and value >= 0.55 else "#202020"
                ax.text(
                    column,
                    row,
                    annotations[row, column],
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )
        total = int(matrix.sum())
        accuracy = float(np.trace(matrix) / total) if total else float("nan")
        ax.set_title(f"Batch {batch}  |  accuracy {accuracy:.1%}", fontsize=13, weight="bold")
        ax.set_xlabel("Predicted gas")
        ax.set_ylabel("True gas")
        ax.tick_params(axis="x", rotation=38, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    if image is None:
        raise ValueError(f"No confusion matrices found for stage {stage}")
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        fraction=0.025,
        pad=0.035,
        aspect=60,
    )
    colorbar.set_label("Share of each true class (row-normalized)")
    colorbar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.suptitle(
        f"{stage}: confusion matrices across target batches\n"
        f"Aggregated over {len(seeds)} seeds; each cell shows row percentage and total predictions",
        fontsize=19,
        weight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def accuracy_figure(summary: pd.DataFrame, stages: list[str], output_path: Path) -> None:
    color_map = matplotlib.colormaps["tab10"]
    palette = {stage: color_map(index) for index, stage in enumerate(stages)}
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    plot_rows: list[pd.DataFrame] = []

    for stage in stages:
        stage_frame = summary[summary["stage"] == stage].sort_values("batch").copy()
        x = stage_frame["batch"].to_numpy(dtype=int)
        mean = stage_frame["mean"].to_numpy(dtype=float)
        std = stage_frame["std"].to_numpy(dtype=float)
        color = palette[stage]
        ax.plot(x, mean, marker="o", markersize=6.5, linewidth=2.3, label=stage, color=color)
        ax.fill_between(
            x,
            np.clip(mean - std, 0, 1),
            np.clip(mean + std, 0, 1),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        plot_rows.append(stage_frame)

    all_batches = sorted(int(batch) for batch in summary["batch"].unique())
    ax.set_title("Target accuracy by batch and experiment phase", fontsize=17, weight="bold")
    ax.set_xlabel("Target batch")
    ax.set_ylabel("Accuracy (five-seed mean ± 1 SD)")
    ax.set_xticks(all_batches)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_ylim(0, 1)
    ax.grid(True, axis="both", color="#d8d8d8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(title="Phase", ncol=2, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_run", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    upstream_run = args.upstream_run.resolve()
    confusion_path = upstream_run / "confusion_matrices.csv"
    summary_path = upstream_run / "per_batch_summary.csv"
    for path in (confusion_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    confusion = pd.read_csv(confusion_path)
    summary = pd.read_csv(summary_path)
    require_columns(
        confusion,
        {"stage", "seed", "batch", "true_gas_label", "predicted_gas_label", "count"},
        confusion_path,
    )
    require_columns(summary, {"stage", "batch", "mean", "std"}, summary_path)

    stages = load_stage_order(upstream_run, confusion)
    batches = sorted(int(batch) for batch in confusion["batch"].unique())
    labels = sorted(
        set(confusion["true_gas_label"].astype(int)).union(confusion["predicted_gas_label"].astype(int))
    )
    if len(batches) != 9:
        raise ValueError(f"Expected exactly 9 target batches for a 3x3 figure; found {batches}")
    if set(summary["stage"].astype(str)) != set(stages):
        raise ValueError("Stage mismatch between confusion matrices and per-batch summary")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_root.resolve() / f"{stamp}_cdcnn_v6_3_result_plots"
    output_dir.mkdir(parents=True, exist_ok=False)

    output_files: list[Path] = []
    for stage in stages:
        safe_stage = stage.lower().replace("-", "_")
        output_path = output_dir / f"{safe_stage}_all_batches_confusion_matrices.png"
        confusion_figure(confusion, stage, batches, labels, output_path)
        output_files.append(output_path)

    accuracy_path = output_dir / "all_phases_accuracy_by_batch.png"
    accuracy_figure(summary, stages, accuracy_path)
    output_files.append(accuracy_path)

    plot_data_path = output_dir / "accuracy_plot_data.csv"
    summary[summary["stage"].isin(stages)].sort_values(["stage", "batch"]).to_csv(
        plot_data_path, index=False
    )
    output_files.append(plot_data_path)

    input_paths = [confusion_path, summary_path]
    config_path = upstream_run / "orchestration_config.json"
    if config_path.is_file():
        input_paths.append(config_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Combined target-batch confusion matrices and accuracy comparison",
        "upstream_run": str(upstream_run),
        "stages": stages,
        "batches": batches,
        "class_order": labels,
        "class_names": {str(label): GAS_NAMES.get(label, f"Class {label}") for label in labels},
        "confusion_aggregation": "Counts summed over all five seeds, then normalized within true-class rows",
        "accuracy_summary": "Arithmetic mean with +/-1 sample standard deviation over five seeds",
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
        "software_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    manifest_path = output_dir / "plot_manifest.json"
    write_json(manifest_path, manifest)
    output_files.append(manifest_path)

    inventory_path = output_dir / "output_inventory.json"
    write_json(
        inventory_path,
        {"outputs": [{"path": path.name, "sha256": sha256(path)} for path in output_files]},
    )
    print(output_dir)


if __name__ == "__main__":
    main()
