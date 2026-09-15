#!/usr/bin/env python3
"""Plot v6.3 target accuracy as five-seed mean +/- sample SD bars."""

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


STAGES = ["B0", "A1", "A2-semantic", "A3"]
DISPLAY_NAMES = {"B0": "B0", "A1": "A1", "A2-semantic": "A2", "A3": "A3"}
SEEDS = [1042, 2024, 3407, 42, 123]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#8F63B8"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_and_verify(upstream_run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed_path = upstream_run / "per_seed_summary.csv"
    stage_path = upstream_run / "stage_summary.csv"
    for path in (per_seed_path, stage_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    per_seed = pd.read_csv(per_seed_path)
    stage_summary = pd.read_csv(stage_path)
    required_seed_columns = {"stage", "seed", "target_mean_accuracy"}
    required_stage_columns = {"stage", "target_mean", "target_mean_std"}
    if missing := required_seed_columns.difference(per_seed.columns):
        raise ValueError(f"{per_seed_path} is missing columns: {sorted(missing)}")
    if missing := required_stage_columns.difference(stage_summary.columns):
        raise ValueError(f"{stage_path} is missing columns: {sorted(missing)}")

    selected = per_seed[per_seed["stage"].isin(STAGES)].copy()
    if set(selected["stage"]) != set(STAGES):
        raise ValueError(f"Expected stages {STAGES}; found {sorted(selected['stage'].unique())}")
    for stage in STAGES:
        actual_seeds = selected.loc[selected["stage"] == stage, "seed"].astype(int).tolist()
        if len(actual_seeds) != 5 or set(actual_seeds) != set(SEEDS):
            raise ValueError(f"{stage} does not contain exactly the prescribed seeds {SEEDS}")

    computed = (
        selected.groupby("stage", sort=False)["target_mean_accuracy"]
        .agg(mean="mean", std="std", count="count")
        .reindex(STAGES)
        .reset_index()
    )
    reported = stage_summary.set_index("stage").reindex(STAGES)
    if not np.allclose(computed["mean"], reported["target_mean"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed target means do not match stage_summary.csv")
    if not np.allclose(computed["std"], reported["target_mean_std"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed sample SDs do not match stage_summary.csv")

    computed["display_stage"] = computed["stage"].map(DISPLAY_NAMES)
    return selected, computed


def plot_accuracy(per_seed: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#262626",
            "xtick.color": "#404040",
            "ytick.color": "#404040",
        }
    )
    fig, ax = plt.subplots(figsize=(10.2, 7.0))
    x = np.arange(len(STAGES), dtype=float)
    means = summary["mean"].to_numpy(dtype=float)
    stds = summary["std"].to_numpy(dtype=float)

    bars = ax.bar(
        x,
        means,
        width=0.64,
        color=COLORS,
        edgecolor="white",
        linewidth=1.2,
        zorder=2,
    )
    ax.errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#252525",
        elinewidth=1.7,
        capsize=6,
        capthick=1.7,
        zorder=4,
    )

    offsets = np.linspace(-0.18, 0.18, len(SEEDS))
    for stage_index, stage in enumerate(STAGES):
        values = (
            per_seed[per_seed["stage"] == stage]
            .set_index("seed")
            .reindex(SEEDS)["target_mean_accuracy"]
            .to_numpy(dtype=float)
        )
        ax.scatter(
            stage_index + offsets,
            values,
            s=34,
            facecolor="white",
            edgecolor="#252525",
            linewidth=1.0,
            zorder=5,
        )

    for bar, mean, std in zip(bars, means, stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.016,
            f"{mean:.1%} ± {std:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color="#202020",
        )

    ax.set_title("CDCNN v6.3 Target Accuracy Across Five Seeds", fontsize=17, pad=18)
    ax.text(
        0.5,
        1.01,
        "Bars show mean; error bars show ±1 sample SD; circles show individual seeds",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#555555",
    )
    ax.set_ylabel("Target mean accuracy (Batches 2–10)")
    ax.set_xlabel("Experiment")
    ax.set_xticks(x, summary["display_stage"])
    ax.set_ylim(0, 0.55)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")
    fig.text(
        0.99,
        0.01,
        "A2 = A2-semantic  •  target mean is unweighted across the nine target batches",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#666666",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_run", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    upstream_run = args.upstream_run.resolve()
    per_seed, summary = load_and_verify(upstream_run)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_root.resolve() / f"{stamp}_cdcnn_v6_3_accuracy_bars"
    output_dir.mkdir(parents=True, exist_ok=False)

    chart_path = output_dir / "b0_a1_a2_a3_target_accuracy_mean_sd.png"
    plot_accuracy(per_seed, summary, chart_path)

    data_path = output_dir / "accuracy_bar_data.csv"
    summary[["display_stage", "stage", "count", "mean", "std"]].to_csv(data_path, index=False)

    input_paths = [upstream_run / "per_seed_summary.csv", upstream_run / "stage_summary.csv"]
    manifest_path = output_dir / "plot_manifest.json"
    write_json(
        manifest_path,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "B0/A1/A2/A3 target-accuracy bar chart for the CDCNN v6.3 five-seed run",
            "upstream_run": str(upstream_run),
            "metric": "Per-seed unweighted mean accuracy across target Batches 2-10",
            "variation": "Sample standard deviation across the five seeds (ddof=1)",
            "seeds": SEEDS,
            "inputs": [{"path": str(path), "sha256": sha256(path)} for path in input_paths],
            "software_versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
            },
        },
    )

    inventory_path = output_dir / "output_inventory.json"
    write_json(
        inventory_path,
        {
            "outputs": [
                {"path": path.name, "sha256": sha256(path)}
                for path in (chart_path, data_path, manifest_path)
            ]
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
