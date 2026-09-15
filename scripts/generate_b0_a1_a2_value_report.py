#!/usr/bin/env python3
"""Create a provenance-preserving B0/A1/A2 cross-batch value report."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "B0": ROOT / "runs/20260908T163610Z_canonical_1d_resnet_b0_sgd",
    "A1": ROOT / "runs/20260908T180205Z_formal_a1_scale_0p5",
    "A2": ROOT / "runs/20260908T184845032203Z_formal_a2",
}
ORDER = ["B0", "A1", "A2"]
TARGET_BATCHES = list(range(2, 11))
GAS_CONFIG = ROOT / "configs/pca.json"
COMPARISON = RUNS["A2"] / "b0_a1_a2_comparison.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def best_label(values: dict[str, float], tolerance: float = 1e-12) -> str:
    maximum = max(values.values())
    winners = [name for name in ORDER if abs(values[name] - maximum) <= tolerance]
    return " = ".join(winners)


def main() -> None:
    created = datetime.now(timezone.utc)
    run_id = created.strftime("%Y%m%dT%H%M%S%fZ") + "_b0_a1_a2_value_report"
    output_dir = ROOT / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    gas_document = json.loads(GAS_CONFIG.read_text(encoding="utf-8"))
    gas_names = {int(key): value for key, value in gas_document["dataset"]["gas_mapping"].items()}
    if sorted(gas_names) != list(range(1, 7)):
        raise ValueError(f"Expected gas labels 1-6, found {sorted(gas_names)}")

    input_paths = [GAS_CONFIG, COMPARISON]
    metrics_by_experiment: dict[str, pd.DataFrame] = {}
    confusions_by_experiment: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    configurations: dict[str, dict] = {}
    upstream_manifests: dict[str, dict] = {}

    for experiment, run_dir in RUNS.items():
        required = [
            run_dir / "batch_metrics.csv",
            run_dir / "confusion_matrices.csv",
            run_dir / "summary.json",
            run_dir / "configuration.json",
            run_dir / "run_manifest.json",
        ]
        missing = [relative(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {experiment} inputs: {missing}")
        input_paths.extend(required)
        metrics_by_experiment[experiment] = pd.read_csv(required[0])
        confusions_by_experiment[experiment] = pd.read_csv(required[1])
        summaries[experiment] = json.loads(required[2].read_text(encoding="utf-8"))
        configurations[experiment] = json.loads(required[3].read_text(encoding="utf-8"))
        upstream_manifests[experiment] = json.loads(required[4].read_text(encoding="utf-8"))

    comparison = pd.read_csv(COMPARISON)
    accuracy_rows: list[dict] = []
    most_confused_rows: list[dict] = []
    largest_pair_rows: list[dict] = []
    pooled_gas_rows: list[dict] = []
    validation_checks: dict[str, bool] = {}

    for experiment in ORDER:
        metrics = metrics_by_experiment[experiment].copy()
        confusion = confusions_by_experiment[experiment].copy()
        expected_metric_columns = {"batch", "samples", "correct", "accuracy"}
        expected_confusion_columns = {"batch", "true_gas_label", "predicted_gas_label", "count"}
        if not expected_metric_columns.issubset(metrics.columns):
            raise ValueError(f"{experiment} batch metrics have unexpected columns")
        if not expected_confusion_columns.issubset(confusion.columns):
            raise ValueError(f"{experiment} confusion matrix has unexpected columns")
        metrics["batch"] = metrics["batch"].astype(int)
        confusion[["batch", "true_gas_label", "predicted_gas_label", "count"]] = confusion[
            ["batch", "true_gas_label", "predicted_gas_label", "count"]
        ].astype(int)
        if metrics["batch"].tolist() != TARGET_BATCHES:
            raise ValueError(f"{experiment} target batches are not exactly 2-10")
        expected_cells = {(batch, true, pred) for batch in TARGET_BATCHES for true in gas_names for pred in gas_names}
        actual_cells = set(confusion[["batch", "true_gas_label", "predicted_gas_label"]].itertuples(index=False, name=None))
        if actual_cells != expected_cells or len(confusion) != len(expected_cells):
            raise ValueError(f"{experiment} confusion matrix is not a complete 9 x 6 x 6 grid")

        for metric in metrics.itertuples(index=False):
            matrix = confusion[confusion["batch"] == metric.batch]
            samples = int(matrix["count"].sum())
            correct = int(matrix.loc[matrix["true_gas_label"] == matrix["predicted_gas_label"], "count"].sum())
            if samples != int(metric.samples) or correct != int(metric.correct):
                raise ValueError(f"{experiment} Batch {metric.batch} metric/confusion count mismatch")
            if abs(correct / samples - float(metric.accuracy)) > 1e-12:
                raise ValueError(f"{experiment} Batch {metric.batch} accuracy mismatch")
            accuracy_rows.append(
                {
                    "experiment": experiment,
                    "batch": int(metric.batch),
                    "samples": samples,
                    "correct": correct,
                    "accuracy": float(metric.accuracy),
                }
            )

            gas_stats = []
            for true_label in gas_names:
                true_rows = matrix[matrix["true_gas_label"] == true_label]
                support = int(true_rows["count"].sum())
                if support == 0:
                    continue
                true_correct = int(true_rows.loc[true_rows["predicted_gas_label"] == true_label, "count"].sum())
                wrong_rows = true_rows[true_rows["predicted_gas_label"] != true_label]
                dominant = wrong_rows.sort_values(["count", "predicted_gas_label"], ascending=[False, True]).iloc[0]
                gas_stats.append(
                    {
                        "true_label": true_label,
                        "support": support,
                        "correct": true_correct,
                        "misclassified": support - true_correct,
                        "recall": true_correct / support,
                        "dominant_predicted_label": int(dominant.predicted_gas_label),
                        "dominant_count": int(dominant["count"]),
                    }
                )
            worst = sorted(gas_stats, key=lambda row: (row["recall"], -row["misclassified"], row["true_label"]))[0]
            batch_errors = samples - correct
            most_confused_rows.append(
                {
                    "experiment": experiment,
                    "batch": int(metric.batch),
                    "true_gas_label": worst["true_label"],
                    "true_gas": gas_names[worst["true_label"]],
                    "support": worst["support"],
                    "correct": worst["correct"],
                    "misclassified": worst["misclassified"],
                    "recall": worst["recall"],
                    "dominant_wrong_label": worst["dominant_predicted_label"],
                    "dominant_wrong_gas": gas_names[worst["dominant_predicted_label"]],
                    "dominant_wrong_count": worst["dominant_count"],
                    "dominant_wrong_share_of_true_gas": worst["dominant_count"] / worst["support"],
                    "share_of_batch_errors": worst["misclassified"] / batch_errors if batch_errors else 0.0,
                }
            )

            off_diagonal = matrix[matrix["true_gas_label"] != matrix["predicted_gas_label"]].copy()
            off_diagonal["true_support"] = off_diagonal["true_gas_label"].map(
                {row["true_label"]: row["support"] for row in gas_stats}
            )
            off_diagonal["within_true_share"] = off_diagonal["count"] / off_diagonal["true_support"]
            largest = off_diagonal.sort_values(
                ["count", "within_true_share", "true_gas_label", "predicted_gas_label"],
                ascending=[False, False, True, True],
            ).iloc[0]
            largest_pair_rows.append(
                {
                    "experiment": experiment,
                    "batch": int(metric.batch),
                    "true_gas_label": int(largest.true_gas_label),
                    "true_gas": gas_names[int(largest.true_gas_label)],
                    "predicted_gas_label": int(largest.predicted_gas_label),
                    "predicted_gas": gas_names[int(largest.predicted_gas_label)],
                    "count": int(largest["count"]),
                    "true_support": int(largest.true_support),
                    "within_true_share": float(largest.within_true_share),
                    "share_of_batch_errors": int(largest["count"]) / batch_errors if batch_errors else 0.0,
                }
            )

        pooled = confusion.groupby(["true_gas_label", "predicted_gas_label"], as_index=False)["count"].sum()
        for true_label in gas_names:
            true_rows = pooled[pooled["true_gas_label"] == true_label]
            support = int(true_rows["count"].sum())
            true_correct = int(true_rows.loc[true_rows["predicted_gas_label"] == true_label, "count"].sum())
            dominant = true_rows[true_rows["predicted_gas_label"] != true_label].sort_values(
                ["count", "predicted_gas_label"], ascending=[False, True]
            ).iloc[0]
            pooled_gas_rows.append(
                {
                    "experiment": experiment,
                    "true_gas_label": true_label,
                    "true_gas": gas_names[true_label],
                    "support": support,
                    "correct": true_correct,
                    "misclassified": support - true_correct,
                    "recall": true_correct / support,
                    "dominant_wrong_label": int(dominant.predicted_gas_label),
                    "dominant_wrong_gas": gas_names[int(dominant.predicted_gas_label)],
                    "dominant_wrong_count": int(dominant["count"]),
                    "dominant_wrong_share_of_true_gas": int(dominant["count"]) / support,
                }
            )

        derived = pd.DataFrame([row for row in accuracy_rows if row["experiment"] == experiment])
        derived_unweighted = float(derived["accuracy"].mean())
        derived_pooled = int(derived["correct"].sum()) / int(derived["samples"].sum())
        validation_checks[f"{experiment.lower()}_unweighted_matches_summary"] = (
            abs(derived_unweighted - float(summaries[experiment]["target_unweighted_mean_accuracy"])) <= 1e-12
        )
        validation_checks[f"{experiment.lower()}_pooled_matches_summary"] = (
            abs(derived_pooled - float(summaries[experiment]["target_pooled_accuracy"])) <= 1e-12
        )

    accuracy = pd.DataFrame(accuracy_rows)
    b0_lookup = accuracy[accuracy["experiment"] == "B0"].set_index("batch")["accuracy"]
    a1_lookup = accuracy[accuracy["experiment"] == "A1"].set_index("batch")["accuracy"]
    accuracy["delta_vs_b0"] = accuracy.apply(lambda row: row.accuracy - b0_lookup.loc[row.batch], axis=1)
    accuracy["delta_vs_previous"] = accuracy.apply(
        lambda row: 0.0
        if row.experiment == "B0"
        else row.accuracy - (b0_lookup if row.experiment == "A1" else a1_lookup).loc[row.batch],
        axis=1,
    )

    summary_rows = []
    for experiment in ORDER:
        subset = accuracy[accuracy["experiment"] == experiment]
        summary_rows.append(
            {
                "experiment": experiment,
                "source_cv_mean_accuracy": float(summaries[experiment]["source_cv_mean_accuracy"]),
                "source_cv_std_accuracy": float(summaries[experiment]["source_cv_std_accuracy"]),
                "target_unweighted_mean_accuracy": float(subset["accuracy"].mean()),
                "target_pooled_correct": int(subset["correct"].sum()),
                "target_pooled_samples": int(subset["samples"].sum()),
                "target_pooled_accuracy": int(subset["correct"].sum()) / int(subset["samples"].sum()),
            }
        )
    summary_metrics = pd.DataFrame(summary_rows)
    summary_metrics["unweighted_delta_vs_b0"] = (
        summary_metrics["target_unweighted_mean_accuracy"] - summary_metrics.iloc[0]["target_unweighted_mean_accuracy"]
    )
    summary_metrics["pooled_delta_vs_b0"] = (
        summary_metrics["target_pooled_accuracy"] - summary_metrics.iloc[0]["target_pooled_accuracy"]
    )

    comparison_batch = comparison[comparison["scope"] == "target_batch"].copy()
    comparison_batch["batch"] = comparison_batch["batch"].astype(int)
    comparison_values = comparison_batch.set_index("batch")[["b0_accuracy", "a1_accuracy", "a2_accuracy"]]
    derived_values = accuracy.pivot(index="batch", columns="experiment", values="accuracy")[["B0", "A1", "A2"]]
    validation_checks["batch_accuracies_match_a2_comparison"] = bool(
        (derived_values.to_numpy() - comparison_values.to_numpy()).max() <= 1e-12
        and (comparison_values.to_numpy() - derived_values.to_numpy()).max() <= 1e-12
    )
    validation_checks["all_validation_checks_passed"] = all(validation_checks.values())
    if not validation_checks["all_validation_checks_passed"]:
        raise ValueError(f"Validation failed: {validation_checks}")

    most_confused = pd.DataFrame(most_confused_rows)
    largest_pairs = pd.DataFrame(largest_pair_rows)
    pooled_gas = pd.DataFrame(pooled_gas_rows)

    accuracy.to_csv(output_dir / "cross_batch_accuracy.csv", index=False)
    summary_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)
    most_confused.to_csv(output_dir / "most_confused_gas_by_batch.csv", index=False)
    largest_pairs.to_csv(output_dir / "largest_confusion_pair_by_batch.csv", index=False)
    pooled_gas.to_csv(output_dir / "pooled_gas_confusions.csv", index=False)
    write_json(output_dir / "validation.json", validation_checks)

    configuration = {
        "analysis_id": "b0_a1_a2_value_report",
        "experiments": {name: relative(path) for name, path in RUNS.items()},
        "gas_mapping": {str(key): value for key, value in gas_names.items()},
        "metrics": {
            "cross_batch_accuracy": "correct predictions divided by samples, separately for target batches 2-10",
            "target_unweighted_mean_accuracy": "arithmetic mean of the nine target-batch accuracies",
            "target_pooled_accuracy": "total correct divided by 13,465 target samples",
            "most_confused_gas": "present true gas with the lowest recall in a batch; ties use more errors, then lower label",
            "dominant_wrong_gas": "largest off-diagonal prediction count within the selected true-gas row; ties use lower label",
            "largest_confusion_pair": "largest directed off-diagonal cell by count; ties use within-true share, then labels",
        },
        "target_batches": TARGET_BATCHES,
        "sample_policy": "No exclusions; no raw records read or changed; no deduplication, clipping, or imputation.",
        "new_preprocessing_fit_scope": "None. This report analyzes frozen upstream result artifacts only.",
        "upstream_preprocessing_fit_scope": {
            name: upstream_manifests[name].get("preprocessing_fit_scope") for name in ORDER
        },
    }
    write_json(output_dir / "configuration.json", configuration)

    unique_inputs = list(dict.fromkeys(input_paths))
    input_manifest = [
        {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in unique_inputs
    ]
    write_json(output_dir / "input_manifest.json", input_manifest)

    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_revision = None
    software_versions = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }
    write_json(output_dir / "software_versions.json", software_versions)

    accuracy_pivot = accuracy.pivot(index="batch", columns="experiment", values="accuracy")[["B0", "A1", "A2"]]
    accuracy_table_rows = []
    for batch, row in accuracy_pivot.iterrows():
        values = {name: float(row[name]) for name in ORDER}
        accuracy_table_rows.append(
            [
                str(batch),
                f"{values['B0']:.2%}",
                f"{values['A1']:.2%}",
                f"{values['A2']:.2%}",
                best_label(values),
            ]
        )

    summary_lookup = summary_metrics.set_index("experiment")
    summary_table_rows = []
    for experiment in ORDER:
        row = summary_lookup.loc[experiment]
        summary_table_rows.append(
            [
                experiment,
                f"{row.source_cv_mean_accuracy:.2%} ± {row.source_cv_std_accuracy:.2%}",
                f"{row.target_unweighted_mean_accuracy:.2%}",
                f"{row.unweighted_delta_vs_b0 * 100:+.2f} pp",
                f"{row.target_pooled_accuracy:.2%}",
                f"{row.pooled_delta_vs_b0 * 100:+.2f} pp",
            ]
        )

    worst_pooled_rows = []
    for experiment in ORDER:
        subset = pooled_gas[pooled_gas["experiment"] == experiment]
        row = subset.sort_values(["recall", "misclassified", "true_gas_label"], ascending=[True, False, True]).iloc[0]
        worst_pooled_rows.append(
            [
                experiment,
                row.true_gas,
                f"{row.correct}/{row.support} ({row.recall:.2%})",
                f"{row.dominant_wrong_gas}: {row.dominant_wrong_count} ({row.dominant_wrong_share_of_true_gas:.2%})",
            ]
        )

    confused_pivot_rows = []
    for batch in TARGET_BATCHES:
        cells = [str(batch)]
        for experiment in ORDER:
            row = most_confused[(most_confused["experiment"] == experiment) & (most_confused["batch"] == batch)].iloc[0]
            cells.append(
                f"{row.true_gas} → {row.dominant_wrong_gas}; recall {row.recall:.1%} "
                f"({row.dominant_wrong_count}/{row.support} in that wrong cell)"
            )
        confused_pivot_rows.append(cells)

    winner_counts = {name: 0 for name in ORDER}
    ties = 0
    for _, row in accuracy_pivot.iterrows():
        maximum = row.max()
        winners = [name for name in ORDER if abs(float(row[name]) - float(maximum)) <= 1e-12]
        if len(winners) == 1:
            winner_counts[winners[0]] += 1
        else:
            ties += 1

    b0_summary = summary_lookup.loc["B0"]
    a1_summary = summary_lookup.loc["A1"]
    a2_summary = summary_lookup.loc["A2"]
    report = f"""# B0, A1, and A2 cross-batch value report

## Outcome

A2 has the highest overall target performance: **{a2_summary.target_unweighted_mean_accuracy:.2%}** unweighted mean accuracy and **{a2_summary.target_pooled_accuracy:.2%}** pooled accuracy. Relative to B0, these are changes of **{a2_summary.unweighted_delta_vs_b0 * 100:+.2f} percentage points** and **{a2_summary.pooled_delta_vs_b0 * 100:+.2f} percentage points**, respectively.

A1 changes the unweighted target mean by **{a1_summary.unweighted_delta_vs_b0 * 100:+.2f} percentage points** and pooled accuracy by **{a1_summary.pooled_delta_vs_b0 * 100:+.2f} percentage points** versus B0. Across the nine target batches, A2 is the sole best result in {winner_counts['A2']} batches, A1 in {winner_counts['A1']}, B0 in {winner_counts['B0']}, with {ties} tied batch. Batch 8 is the lowest-accuracy batch for all three configurations.

## Accuracy summary

{markdown_table(['Configuration', 'Batch-1 CV mean ± SD', 'Target mean', 'Mean Δ vs B0', 'Target pooled', 'Pooled Δ vs B0'], summary_table_rows)}

The target mean gives every batch equal weight. The pooled result weights batches by sample count, so the two summaries answer different questions.

## Cross-batch accuracy

{markdown_table(['Batch', 'B0', 'A1', 'A2', 'Best'], accuracy_table_rows)}

## Most confused gas

“Most confused gas” is defined here as the present true gas with the lowest recall. The arrow shows its most common wrong prediction. Classes absent from a batch are not eligible. Because this recall-based definition can differ from the largest raw off-diagonal count, both measures are saved as separate CSV files.

{markdown_table(['Batch', 'B0', 'A1', 'A2'], confused_pivot_rows)}

Across all target batches pooled, **Acetaldehyde** is the lowest-recall gas in every configuration and is mostly predicted as **Ethanol**:

{markdown_table(['Configuration', 'Lowest-recall gas', 'Correct / support (recall)', 'Dominant wrong prediction'], worst_pooled_rows)}

This is a measured classification-confusion pattern. It does not by itself establish sensor drift as its physical cause. Notably, A2 improves aggregate target accuracy while its pooled Acetaldehyde recall remains extremely low; aggregate gains therefore should not be read as uniform improvement across gases.

## Provenance and validation

- B0: `{relative(RUNS['B0'])}` (canonical SGD B0 used by the A1/A2 chain)
- A1: `{relative(RUNS['A1'])}` (input augmentation, perturbation scale 0.5)
- A2: `{relative(RUNS['A2'])}` (A1 plus latent feature generation; completed corrected run)
- New preprocessing: none. This report reads frozen metrics and confusion matrices only.
- Upstream scaler scope: Batch-1 training fold for CV and all Batch 1 for final fitting; no target-batch scaler was fit.
- Samples excluded: none. Raw dataset files were not accessed or modified by this reporting run.
- Validation: confusion totals, diagonal counts, saved batch metrics, saved summaries, and the A2 comparison artifact agree exactly.

## Machine-readable outputs

- `cross_batch_accuracy.csv`: every batch accuracy and deltas.
- `summary_metrics.csv`: source CV, unweighted target, and pooled target summaries.
- `most_confused_gas_by_batch.csv`: lowest-recall gas and its dominant wrong destination.
- `largest_confusion_pair_by_batch.csv`: largest off-diagonal flow by raw count.
- `pooled_gas_confusions.csv`: pooled support, recall, and dominant wrong destination for all six gases.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    completed = datetime.now(timezone.utc)
    run_manifest = {
        "run_id": run_id,
        "analysis_id": "b0_a1_a2_value_report",
        "status": "completed",
        "created_utc": created.isoformat(),
        "completed_utc": completed.isoformat(),
        "command": "python scripts/generate_b0_a1_a2_value_report.py",
        "working_directory": str(ROOT),
        "git_revision": git_revision,
        "raw_data_accessed": False,
        "raw_data_modified": False,
        "exclusions": [],
        "new_preprocessing_fit_scope": "None; frozen upstream result artifacts only.",
        "upstream_preprocessing_fit_scope": configuration["upstream_preprocessing_fit_scope"],
        "input_manifest": "input_manifest.json",
        "software_versions": "software_versions.json",
        "configuration": "configuration.json",
        "validation": validation_checks,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    output_names = [
        "configuration.json",
        "cross_batch_accuracy.csv",
        "input_manifest.json",
        "largest_confusion_pair_by_batch.csv",
        "most_confused_gas_by_batch.csv",
        "pooled_gas_confusions.csv",
        "report.md",
        "run_manifest.json",
        "software_versions.json",
        "summary_metrics.csv",
        "validation.json",
    ]
    inventory = [
        {"path": name, "bytes": (output_dir / name).stat().st_size, "sha256": sha256(output_dir / name)}
        for name in output_names
    ]
    inventory.append(
        {"path": "output_inventory.json", "bytes": None, "sha256": None, "note": "Self-entry omitted to avoid recursive hashing."}
    )
    write_json(output_dir / "output_inventory.json", inventory)
    print(relative(output_dir))


if __name__ == "__main__":
    main()
