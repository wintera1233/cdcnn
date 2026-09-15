"""Validated exploratory PCA workflow for the gas sensor drift dataset."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_batch(path: Path, batch_id: int, feature_count: int):
    rows, labels, line_numbers = [], [], []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            tokens = line.split()
            if len(tokens) != feature_count + 1:
                raise ValueError(f"{path}:{line_number}: expected {feature_count + 1} tokens")
            try:
                label = int(tokens[0])
                values = np.empty(feature_count, dtype=np.float64)
                observed = []
                for token in tokens[1:]:
                    index_text, value_text = token.split(":", 1)
                    index = int(index_text)
                    if not 1 <= index <= feature_count:
                        raise ValueError(f"feature index {index} is out of range")
                    observed.append(index)
                    values[index - 1] = float(value_text)
                if observed != list(range(1, feature_count + 1)):
                    raise ValueError("feature indices are missing, duplicated, or out of order")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(values)
            labels.append(label)
            line_numbers.append(line_number)
    return np.vstack(rows), np.asarray(labels), np.asarray(line_numbers)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def versions() -> dict:
    import scipy
    import seaborn
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        "seaborn": seaborn.__version__, "joblib": joblib.__version__,
        "platform": platform.platform(),
    }


def duplicate_table(x: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    _, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
    records = []
    for group, count in enumerate(counts):
        if count > 1:
            members = meta.loc[inverse == group, ["sample_id", "batch", "line_number", "gas_label"]]
            for row in members.itertuples(index=False):
                records.append({"duplicate_group": int(group), "group_size": int(count),
                                "sample_id": row.sample_id, "batch": row.batch,
                                "line_number": row.line_number, "gas_label": row.gas_label})
    return pd.DataFrame(records, columns=["duplicate_group", "group_size", "sample_id", "batch", "line_number", "gas_label"])


def plot_scatter(scores, meta, color_col, path, title, label_names=None):
    fig, ax = plt.subplots(figsize=(10, 7))
    values = sorted(meta[color_col].unique())
    cmap = plt.get_cmap("tab10")
    for i, value in enumerate(values):
        mask = meta[color_col].to_numpy() == value
        name = label_names.get(str(value), str(value)) if label_names else f"Batch {value}"
        ax.scatter(scores[mask, 0], scores[mask, 1], s=9, alpha=.45,
                   color=cmap(i % 10), label=name, rasterized=True)
    ax.set(xlabel="PC1 score", ylabel="PC2 score", title=title)
    ax.legend(markerscale=2, fontsize=8, frameon=False, ncol=2)
    ax.axhline(0, color="0.8", lw=.6); ax.axvline(0, color="0.8", lw=.6)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def run_pca(name, fit_mask, x, meta, out, cfg):
    scaler = StandardScaler().fit(x[fit_mask])
    pca = PCA(svd_solver=cfg["pca"]["svd_solver"], whiten=cfg["pca"]["whiten"])
    pca.fit(scaler.transform(x[fit_mask]))
    scores = pca.transform(scaler.transform(x))
    joblib.dump(scaler, out / f"{name}_scaler.joblib")
    joblib.dump(pca, out / f"{name}_pca.joblib")

    ev = pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                       "explained_variance": pca.explained_variance_,
                       "explained_variance_ratio": pca.explained_variance_ratio_,
                       "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_)})
    ev.to_csv(out / f"{name}_explained_variance.csv", index=False)
    pd.DataFrame(pca.components_.T, index=np.arange(1, x.shape[1] + 1),
                 columns=[f"PC{i}" for i in range(1, pca.n_components_ + 1)]).rename_axis("feature_index").to_csv(out / f"{name}_loadings.csv")
    score_df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(scores, columns=[f"PC{i}" for i in range(1, scores.shape[1] + 1)])], axis=1)
    score_df.to_csv(out / f"{name}_scores.csv.gz", index=False, compression="gzip")

    first = [f"PC{i}" for i in range(1, min(10, scores.shape[1]) + 1)]
    summary = score_df.groupby("batch")[first].agg(["mean", "std", "median"])
    summary.columns = [f"{pc}_{stat}" for pc, stat in summary.columns]
    summary.reset_index().to_csv(out / f"{name}_batch_score_summary.csv", index=False)
    gas_summary = score_df.groupby(["batch", "gas_label"])[first].agg(["count", "mean", "std"])
    gas_summary.columns = [f"{pc}_{stat}" for pc, stat in gas_summary.columns]
    gas_summary.reset_index().to_csv(out / f"{name}_batch_gas_score_summary.csv", index=False)

    plot_scatter(scores, meta, "batch", out / f"{name}_pc1_pc2_by_batch.png", f"{name.replace('_', ' ').title()}: PC1–PC2 by batch")
    plot_scatter(scores, meta, "gas_label", out / f"{name}_pc1_pc2_by_gas.png", f"{name.replace('_', ' ').title()}: PC1–PC2 by gas", cfg["dataset"]["gas_mapping"])
    fig, ax = plt.subplots(figsize=(9, 5)); limit = min(40, len(ev))
    ax.plot(ev.component[:limit], ev.explained_variance_ratio[:limit] * 100, marker="o", ms=3)
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)", title=f"{name.replace('_', ' ').title()}: scree plot (first {limit} PCs)")
    fig.tight_layout(); fig.savefig(out / f"{name}_scree.png", dpi=180); plt.close(fig)
    thresholds = {str(t): int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), t) + 1) for t in (.8, .9, .95, .99)}
    return {"fit_sample_count": int(fit_mask.sum()), "components": int(pca.n_components_),
            "pc1_ratio": float(ev.iloc[0].explained_variance_ratio),
            "pc2_ratio": float(ev.iloc[1].explained_variance_ratio),
            "pc1_pc2_ratio": float(ev.iloc[:2].explained_variance_ratio.sum()),
            "components_for_cumulative_variance": thresholds}


def main(config_path: str = "configs/pca.json") -> Path:
    root = Path.cwd().resolve()
    config_file = (root / config_path).resolve()
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_pca"
    out = root / cfg["output_root"] / run_id
    out.mkdir(parents=True, exist_ok=False)
    inputs, arrays, labels, batches, lines = [], [], [], [], []
    for batch in cfg["dataset"]["expected_batch_ids"]:
        path = root / cfg["dataset"]["path"] / cfg["dataset"]["batch_file_pattern"].format(batch_id=batch)
        arr, lab, line = load_batch(path, batch, cfg["dataset"]["expected_feature_count"])
        inputs.append({"batch": batch, "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                       "sha256": sha256(path), "records": len(arr)})
        arrays.append(arr); labels.append(lab); batches.extend([batch] * len(arr)); lines.append(line)
    x = np.vstack(arrays); y = np.concatenate(labels); line_numbers = np.concatenate(lines)
    meta = pd.DataFrame({"sample_id": [f"batch{b}:line{line}" for b, line in zip(batches, line_numbers)],
                         "batch": batches, "line_number": line_numbers, "gas_label": y})
    meta["gas_name"] = meta.gas_label.astype(str).map(cfg["dataset"]["gas_mapping"])

    finite = np.isfinite(x)
    variances = np.var(x, axis=0)
    duplicates = duplicate_table(x, meta)
    duplicates.to_csv(out / "duplicate_records.csv", index=False)
    feature_stats = pd.DataFrame({"feature_index": np.arange(1, x.shape[1] + 1), "minimum": x.min(0),
                                  "maximum": x.max(0), "mean": x.mean(0), "std": x.std(0),
                                  "variance": variances, "median": np.median(x, 0),
                                  "q01": np.quantile(x, .01, axis=0), "q99": np.quantile(x, .99, axis=0)})
    feature_stats.to_csv(out / "feature_numerical_summary.csv", index=False)
    abs_values = np.abs(x)
    validation = {
        "records": int(len(x)), "features": int(x.shape[1]), "batches": sorted(set(batches)),
        "gas_labels": sorted(map(int, np.unique(y))), "finite_value_count": int(finite.sum()),
        "nonfinite_value_count": int((~finite).sum()),
        "zero_variance_feature_indices": (np.flatnonzero(variances == 0) + 1).tolist(),
        "exact_duplicate_groups": int(duplicates.duplicate_group.nunique()) if len(duplicates) else 0,
        "records_in_exact_duplicate_groups": int(len(duplicates)),
        "absolute_value_quantiles": {str(q): float(np.quantile(abs_values, q)) for q in (0, .5, .9, .99, .999, 1)},
        "extreme_value_policy": "Reported only; no clipping, removal, or imputation. Extremes are descriptive, not automatically invalid.",
        "exclusions": [], "concentration_analysis": "unavailable: no concentration field or sidecar metadata in supplied files",
    }
    save_json(out / "numerical_validation.json", validation)
    if validation["nonfinite_value_count"] or validation["zero_variance_feature_indices"]:
        raise ValueError("PCA stopped: non-finite values or zero-variance features found; see validation output")

    global_result = run_pca("global", np.ones(len(x), dtype=bool), x, meta, out, cfg)
    source_result = run_pca("source_only", meta.batch.to_numpy() == cfg["source_batch"], x, meta, out, cfg)
    config_copy = json.loads(json.dumps(cfg)); config_copy["resolved_dataset_path"] = str((root / cfg["dataset"]["path"]).resolve())
    save_json(out / "configuration.json", config_copy)
    save_json(out / "input_manifest.json", inputs)
    save_json(out / "software_versions.json", versions())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except Exception:
        revision = None
    manifest = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": f"{sys.executable} scripts/run_pca.py --config {config_path}",
                "git_revision": revision, "random_seed": cfg["random_seed"],
                "preprocessing": {"global": "StandardScaler fit on batches 1-10; PCA fit on batches 1-10",
                                  "source_only": "StandardScaler fit on Batch 1 only; PCA fit on Batch 1 only; all batches transformed with those objects"},
                "sample_policy": "All structurally and numerically valid samples retained; no resampling or batch weighting.",
                "results": {"global": global_result, "source_only": source_result}}
    save_json(out / "run_manifest.json", manifest)
    write_report(out, inputs, validation, global_result, source_result)
    output_files = sorted(p.name for p in out.iterdir() if p.name != "output_inventory.json")
    save_json(out / "output_inventory.json", output_files)
    return out


def write_report(out, inputs, validation, global_result, source_result):
    rows = "\n".join(f"| {i['batch']} | {i['records']:,} |" for i in inputs)
    dup = f"{validation['exact_duplicate_groups']} groups involving {validation['records_in_exact_duplicate_groups']} records"
    gb = pd.read_csv(out / "global_batch_score_summary.csv")
    sb = pd.read_csv(out / "source_only_batch_score_summary.csv")
    gmin, gmax = gb.loc[gb.PC1_mean.idxmin()], gb.loc[gb.PC1_mean.idxmax()]
    sb["distance"] = np.hypot(sb.PC1_mean - sb.loc[sb.batch == 1, "PC1_mean"].iloc[0],
                              sb.PC2_mean - sb.loc[sb.batch == 1, "PC2_mean"].iloc[0])
    far = sb.loc[sb.distance.idxmax()]
    b1_sd = sb.loc[sb.batch == 1, "PC1_std"].iloc[0]
    b2_sd = sb.loc[sb.batch == 2, "PC1_std"].iloc[0]
    text = f"""# Exploratory PCA report

## Scope and data validity

All 13,910 supplied records from `Dataset/batch1.dat` through `batch10.dat` were parsed as a gas label plus 128 indexed floating-point features. The feature positions are treated as extracted features (16 sensors × 8 features), not raw time samples. Labels and provenance fields were excluded from the PCA matrices.

| Batch | Records |
|---:|---:|
{rows}

Numerical validation found {validation['nonfinite_value_count']} non-finite values, {len(validation['zero_variance_feature_indices'])} zero-variance features, and {dup}. Exact duplicates were retained. No records were excluded, deduplicated, clipped, imputed, resampled, or batch-weighted. Feature-level ranges and tail quantiles are in `feature_numerical_summary.csv`; these describe extreme values without declaring them erroneous.

Concentration analysis and concentration-matched comparisons are **unavailable** because the supplied files contain no concentration field and no sidecar metadata. No concentrations were inferred.

## Global PCA

One StandardScaler and one PCA were fit on all batches (all {global_result['fit_sample_count']:,} samples). PC1 explains {global_result['pc1_ratio']:.2%}, PC2 explains {global_result['pc2_ratio']:.2%}, and together they explain {global_result['pc1_pc2_ratio']:.2%}. Components needed for 80%, 90%, 95%, and 99% cumulative variance are {global_result['components_for_cumulative_variance']}.

Because batch sizes range from 161 to 3,613, this ordinary sample-level fit gives larger batches more influence on the global mean, scale, and covariance. The plots should therefore not be interpreted as a batch-balanced estimate. All valid samples were intentionally retained.

The global PC1 batch centroids span from {gmin.PC1_mean:.2f} (Batch {int(gmin.batch)}) to {gmax.PC1_mean:.2f} (Batch {int(gmax.batch)}), while PC1 within-batch standard deviations range from {gb.PC1_std.min():.2f} to {gb.PC1_std.max():.2f}. Location and spread therefore vary by batch in the leading global subspace, but the score clouds overlap; this is not complete batch separation.

## Source-only PCA

The StandardScaler and PCA were both fit exclusively on the {source_result['fit_sample_count']:,} Batch 1 samples. The same fitted transformations were then applied to every batch. Within the Batch 1 fit, PC1 explains {source_result['pc1_ratio']:.2%}, PC2 explains {source_result['pc2_ratio']:.2%}, and together they explain {source_result['pc1_pc2_ratio']:.2%}. Components needed for 80%, 90%, 95%, and 99% of Batch 1 standardized variance are {source_result['components_for_cumulative_variance']}. Target batches were never used to refit scaling or PCA.

Relative to the Batch 1 centroid at approximately zero by construction, Batch {int(far.batch)} has the largest PC1–PC2 centroid displacement ({far.distance:.2f}). Batch 2 has an especially large PC1 spread (standard deviation {b2_sd:.2f}, versus {b1_sd:.2f} in Batch 1) and extreme projected scores visible in the full-range plot. These observations were retained. This shows that Batch-1 scaling maps some later observations far outside the source score range.

## Interpretation and limits

The batch-colored score plots and batch score summaries quantify changes in the measured multivariate feature distributions. Separation, centroid displacement, or changing spread across batches is consistent with distribution shift, but PCA alone cannot identify its cause or prove sensor drift. Gas composition can also affect visible structure; the gas-colored plots and batch-by-gas summaries expose that confounding without concentration matching. Two-dimensional plots omit variance in later components, and visual separation does not establish predictive performance. No classifier was trained.

Review the compressed score tables for sample-level provenance, the loading tables for feature contributions, the explained-variance tables and scree plots for dimensionality, and the batch/batch-gas summaries for quantitative comparisons. Fitted scaler and PCA objects are preserved for exact reuse.
"""
    (out / "report.md").write_text(text, encoding="utf-8")
