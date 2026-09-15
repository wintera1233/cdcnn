#!/usr/bin/env python3
"""Create gas-by-batch PCA diagnostics from an existing fitted PCA run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pca_analysis import load_batch


GAS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
BATCH_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"]
BATCH_COLORS = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 10))


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def limits(scores: np.ndarray, zoom: bool) -> dict[str, list[float]]:
    if zoom:
        bounds = np.quantile(scores[:, :2], [.005, .995], axis=0)
    else:
        bounds = np.vstack([scores[:, :2].min(axis=0), scores[:, :2].max(axis=0)])
    span = bounds[1] - bounds[0]
    margin = np.where(span > 0, .03 * span, 1.0)
    return {"x": [float(bounds[0, 0] - margin[0]), float(bounds[1, 0] + margin[0])],
            "y": [float(bounds[0, 1] - margin[1]), float(bounds[1, 1] + margin[1])],
            "quantiles": ([.005, .995] if zoom else None), "margin_fraction": .03}


def outside_mask(scores: np.ndarray, lim: dict) -> np.ndarray:
    return ((scores[:, 0] < lim["x"][0]) | (scores[:, 0] > lim["x"][1]) |
            (scores[:, 1] < lim["y"][0]) | (scores[:, 1] > lim["y"][1]))


def decorate(ax, title: str, ratios, source_only=False):
    qualifier = " (Batch 1 fit)" if source_only else ""
    ax.set(xlabel=f"PC1 ({ratios[0]:.2%}{qualifier})",
           ylabel=f"PC2 ({ratios[1]:.2%}{qualifier})", title=title)
    ax.axhline(0, color="0.82", lw=.6, zorder=0)
    ax.axvline(0, color="0.82", lw=.6, zorder=0)


def gas_legend(gases, gas_names):
    return [Line2D([], [], marker="o", ls="none", color=GAS_COLORS[i], label=gas_names[str(g)])
            for i, g in enumerate(gases)]


def batch_legend(batches):
    return [Line2D([], [], marker=BATCH_MARKERS[i], ls="none", color="0.25", label=f"Batch {b}")
            for i, b in enumerate(batches)]


def combined_plot(scores, meta, ratios, gas_names, lim, path, title, source_only):
    fig, ax = plt.subplots(figsize=(12, 8))
    gases, batches = sorted(meta.gas_label.unique()), sorted(meta.batch.unique())
    for gi, gas in enumerate(gases):
        for bi, batch in enumerate(batches):
            mask = (meta.gas_label.to_numpy() == gas) & (meta.batch.to_numpy() == batch)
            ax.scatter(scores[mask, 0], scores[mask, 1], s=13, alpha=.48,
                       color=GAS_COLORS[gi], marker=BATCH_MARKERS[bi], linewidths=0,
                       rasterized=True)
    decorate(ax, title, ratios, source_only)
    ax.set_xlim(lim["x"]); ax.set_ylim(lim["y"])
    first = ax.legend(handles=gas_legend(gases, gas_names), title="Gas (color)",
                      loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    ax.add_artist(first)
    ax.legend(handles=batch_legend(batches), title="Batch (marker)", loc="lower left",
              bbox_to_anchor=(1.01, 0), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def batch_panels(scores, meta, ratios, gas_names, lim, path, view):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    gases = sorted(meta.gas_label.unique()); b1 = meta.batch.to_numpy() == 1
    for batch, ax in zip(range(1, 11), axes.flat):
        ax.scatter(scores[b1, 0], scores[b1, 1], s=7, color="0.72", alpha=.16,
                   marker=".", linewidths=0, rasterized=True, zorder=1)
        for gi, gas in enumerate(gases):
            mask = (meta.batch.to_numpy() == batch) & (meta.gas_label.to_numpy() == gas)
            ax.scatter(scores[mask, 0], scores[mask, 1], s=13, color=GAS_COLORS[gi],
                       alpha=.64, marker="o", edgecolors="none", rasterized=True, zorder=2)
        ax.set_title(f"Batch {batch}")
        ax.axhline(0, color="0.84", lw=.5); ax.axvline(0, color="0.84", lw=.5)
        ax.set_xlim(lim["x"]); ax.set_ylim(lim["y"])
    for ax in axes[-1]: ax.set_xlabel(f"PC1 ({ratios[0]:.2%}, Batch 1 fit)")
    for ax in axes[:, 0]: ax.set_ylabel(f"PC2 ({ratios[1]:.2%}, Batch 1 fit)")
    handles = [Line2D([], [], marker=".", ls="none", color="0.65", markersize=10,
                      label="Batch 1 reference (background)")]
    handles += gas_legend(gases, gas_names)
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               title="Foreground: current batch, colored by gas")
    fig.suptitle(f"Source-only PCA: each batch against Batch 1 ({view})", fontsize=15)
    fig.tight_layout(rect=[0, .1, 1, .95]); fig.savefig(path, dpi=190); plt.close(fig)


def trajectory_plot(scores, meta, ratios, gas_names, lim, path, view):
    gases = sorted(meta.gas_label.unique())
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    centroids = []
    for gi, (gas, ax) in enumerate(zip(gases, axes.flat)):
        rows = []
        for bi, batch in enumerate(range(1, 11)):
            mask = (meta.gas_label.to_numpy() == gas) & (meta.batch.to_numpy() == batch)
            xy = scores[mask, :2]
            if len(xy) == 0:
                centroids.append({"gas_label": int(gas), "gas_name": gas_names[str(gas)],
                                  "batch": batch, "sample_count": 0,
                                  "PC1_centroid": np.nan, "PC2_centroid": np.nan})
                rows.append([np.nan, np.nan])
                continue
            ax.scatter(xy[:, 0], xy[:, 1], s=9, color=BATCH_COLORS[bi], alpha=.38,
                       edgecolors="none", rasterized=True)
            centroid = xy.mean(axis=0); rows.append(centroid)
            centroids.append({"gas_label": int(gas), "gas_name": gas_names[str(gas)],
                              "batch": batch, "sample_count": len(xy),
                              "PC1_centroid": centroid[0], "PC2_centroid": centroid[1]})
        rows = np.asarray(rows)
        ax.plot(rows[:, 0], rows[:, 1], color="0.2", lw=1.1, alpha=.8, zorder=3)
        for bi, centroid in enumerate(rows):
            if not np.all(np.isfinite(centroid)):
                continue
            ax.scatter(*centroid, s=(95 if bi == 0 else 48), color=BATCH_COLORS[bi],
                       edgecolor=("black" if bi == 0 else "white"), linewidth=.9, zorder=4)
            ax.annotate(f"B{bi + 1}", centroid, xytext=(4, 4), textcoords="offset points",
                        fontsize=7, weight=("bold" if bi == 0 else "normal"), clip_on=True)
        ax.set_title(gas_names[str(gas)])
        ax.axhline(0, color="0.84", lw=.5); ax.axvline(0, color="0.84", lw=.5)
        ax.set_xlim(lim["x"]); ax.set_ylim(lim["y"])
    for ax in axes[-1]: ax.set_xlabel(f"PC1 ({ratios[0]:.2%}, Batch 1 fit)")
    for ax in axes[:, 0]: ax.set_ylabel(f"PC2 ({ratios[1]:.2%}, Batch 1 fit)")
    handles = [Line2D([], [], marker="o", ls="none", color=BATCH_COLORS[i], label=f"B{i+1}") for i in range(10)]
    handles += [Line2D([], [], marker="o", ls="-", color="black", markerfacecolor=BATCH_COLORS[0],
                       markersize=9, label="B1 centroid (emphasized)")]
    fig.legend(handles=handles, loc="lower center", ncol=11, frameon=False, title="Batch chronology")
    fig.suptitle(f"Source-only PCA: within-gas batch trajectories ({view})", fontsize=15)
    fig.tight_layout(rect=[0, .09, 1, .95]); fig.savefig(path, dpi=190); plt.close(fig)
    return pd.DataFrame(centroids)


def summaries(scores, meta, lim):
    frame = meta[["batch", "gas_label", "gas_name"]].copy()
    frame[["PC1", "PC2"]] = scores[:, :2]
    records = []
    for (batch, gas, name), group in frame.groupby(["batch", "gas_label", "gas_name"], sort=True):
        cx, cy = group[["PC1", "PC2"]].mean()
        dispersion = np.sqrt(((group.PC1 - cx) ** 2 + (group.PC2 - cy) ** 2).mean())
        records.append({"batch": batch, "gas_label": gas, "gas_name": name, "n": len(group),
                        "PC1_centroid": cx, "PC2_centroid": cy, "rms_dispersion": dispersion})
    out = pd.DataFrame(records)
    ref = out[out.batch == 1].set_index("gas_label")
    out["centroid_displacement_from_B1"] = [np.hypot(r.PC1_centroid-ref.loc[r.gas_label, "PC1_centroid"],
                                                             r.PC2_centroid-ref.loc[r.gas_label, "PC2_centroid"])
                                                   for r in out.itertuples()]
    out["dispersion_ratio_to_B1"] = [r.rms_dispersion/ref.loc[r.gas_label, "rms_dispersion"] for r in out.itertuples()]
    return out


def extreme_diagnostics(root, cfg, upstream, scores, meta, scaler, pca, out):
    arrays = []
    for batch in cfg["dataset"]["expected_batch_ids"]:
        path = root / cfg["dataset"]["path"] / cfg["dataset"]["batch_file_pattern"].format(batch_id=batch)
        x, _, _ = load_batch(path, batch, cfg["dataset"]["expected_feature_count"])
        arrays.append(x)
    x = np.vstack(arrays)
    z = scaler.transform(x)
    recomputed = pca.transform(z)
    max_delta = float(np.max(np.abs(recomputed[:, :2] - scores[:, :2])))
    if max_delta > 1e-8:
        raise ValueError(f"Saved scores do not match saved source-only fitted objects: max delta={max_delta}")
    selected = []
    for pc in range(2):
        idx = int(np.argmax(np.abs(scores[:, pc])))
        if idx not in selected: selected.append(idx)
    detail_rows, sample_rows = [], []
    for idx in selected:
        row = meta.iloc[idx]
        centered = z[idx] - pca.mean_
        sums, residuals = [], []
        for pc in range(2):
            contributions = centered * pca.components_[pc]
            order = np.argsort(np.abs(contributions))[::-1][:12]
            sums.append(contributions.sum()); residuals.append(contributions.sum() - scores[idx, pc])
            for rank, j in enumerate(order, 1):
                detail_rows.append({"sample_id": row.sample_id, "explained_coordinate": f"PC{pc+1}",
                                    "rank_by_absolute_contribution": rank, "feature_index": int(j+1),
                                    "raw_value": x[idx, j], "batch1_feature_mean": scaler.mean_[j],
                                    "batch1_feature_scale": scaler.scale_[j], "standardized_value": z[idx, j],
                                    "pca_center": pca.mean_[j], "centered_standardized_value": centered[j],
                                    "loading": pca.components_[pc, j], "signed_contribution": contributions[j]})
        extreme_for = [f"largest_absolute_PC{pc+1}" for pc in range(2)
                       if idx == int(np.argmax(np.abs(scores[:, pc])))]
        sample_rows.append({"extreme_for": ";".join(extreme_for), "sample_id": row.sample_id,
                            "original_file": f"Dataset/batch{int(row.batch)}.dat",
                            "line_number": int(row.line_number), "batch": int(row.batch),
                            "gas_label": int(row.gas_label), "gas_name": row.gas_name,
                            "PC1": scores[idx, 0], "PC2": scores[idx, 1],
                            "PC1_contribution_sum": sums[0], "PC1_score_residual": residuals[0],
                            "PC2_contribution_sum": sums[1], "PC2_score_residual": residuals[1]})
    pd.DataFrame(sample_rows).to_csv(out / "source_only_extreme_samples.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(out / "source_only_extreme_feature_contributions.csv", index=False)
    return pd.DataFrame(sample_rows), max_delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/pca.json"))
    args = parser.parse_args()
    root = Path.cwd().resolve(); upstream = args.source_run.resolve()
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_pca_visual_revision"
    out = root / cfg["output_root"] / run_id; out.mkdir(parents=True, exist_ok=False)
    gas_names = cfg["dataset"]["gas_mapping"]
    records = {}; score_sets = {}; models = {}
    for name in ("global", "source_only"):
        score_df = pd.read_csv(upstream / f"{name}_scores.csv.gz")
        scores = score_df[["PC1", "PC2"]].to_numpy()
        pca = joblib.load(upstream / f"{name}_pca.joblib")
        scaler = joblib.load(upstream / f"{name}_scaler.joblib")
        ratios = pca.explained_variance_ratio_[:2]
        zoom_lim, full_lim = limits(scores, True), limits(scores, False)
        records[name] = {"zoom": zoom_lim, "full": full_lim,
                         "explained_variance_ratio": ratios.tolist()}
        for view, lim in (("full", full_lim), ("zoom", zoom_lim)):
            combined_plot(scores, score_df, ratios, gas_names, lim,
                          out / f"{name}_gas_batch_{view}.png",
                          f"{name.replace('_', ' ').title()} PCA: gas (color) and batch (marker) — {view}", name == "source_only")
        outside = outside_mask(scores, zoom_lim)
        counts = score_df.assign(outside_zoom=outside).groupby(["batch", "gas_label", "gas_name"], as_index=False).agg(
            total_points=("sample_id", "size"), points_outside_viewport=("outside_zoom", "sum"))
        counts.to_csv(out / f"{name}_zoom_outside_counts.csv", index=False)
        records[name]["points_outside_zoom"] = int(outside.sum())
        score_sets[name] = (scores, score_df); models[name] = (scaler, pca)
    save_json(out / "viewport_limits.json", records)

    scores, meta = score_sets["source_only"]; scaler, pca = models["source_only"]
    ratios = pca.explained_variance_ratio_[:2]
    for view in ("full", "zoom"):
        lim = records["source_only"][view]
        batch_panels(scores, meta, ratios, gas_names, lim, out / f"source_only_batch1_comparison_{view}.png", view)
        centroids = trajectory_plot(scores, meta, ratios, gas_names, lim,
                                    out / f"source_only_within_gas_trajectories_{view}.png", view)
    zoom_lim = records["source_only"]["zoom"]
    valid_centroid = centroids[["PC1_centroid", "PC2_centroid"]].notna().all(axis=1)
    centroids["outside_zoom"] = False
    centroids.loc[valid_centroid, "outside_zoom"] = outside_mask(
        centroids.loc[valid_centroid, ["PC1_centroid", "PC2_centroid"]].to_numpy(), zoom_lim)
    centroids.to_csv(out / "source_only_gas_batch_centroids.csv", index=False)
    summary = summaries(scores, meta, zoom_lim)
    summary.to_csv(out / "source_only_gas_batch_shift_summary.csv", index=False)
    extreme, verify_delta = extreme_diagnostics(root, cfg, upstream, scores, meta, scaler, pca, out)
    write_report(out, upstream, records, summary, centroids, extreme, verify_delta)
    revision_config = {"source_run": str(upstream), "config_source": str((root / args.config).resolve()),
                       "pca_configuration": cfg, "views": ["full", "zoom"],
                       "zoom_quantiles": [.005, .995], "zoom_margin_fraction": .03,
                       "axes": "linear", "extreme_feature_count_per_coordinate": 12}
    save_json(out / "configuration.json", revision_config)
    source_files = [upstream / f"{name}_{kind}" for name in ("global", "source_only")
                    for kind in ("scores.csv.gz", "scaler.joblib", "pca.joblib")]
    raw_files = [root / cfg["dataset"]["path"] / cfg["dataset"]["batch_file_pattern"].format(batch_id=batch)
                 for batch in cfg["dataset"]["expected_batch_ids"]]
    source_files += [root / args.config, upstream / "run_manifest.json"] + raw_files
    manifest = {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run": str(upstream), "purpose": "visualization-only derivative analysis",
                "preprocessing_fit_scope": {"global": "reused fit on batches 1-10",
                    "source_only": "reused StandardScaler and PCA fit on Batch 1 only"},
                "sample_policy": "All saved scores retained; no clipping, exclusion, imputation, deduplication, or refitting.",
                "visualization_zoom": "Pooled PC1/PC2 0.5th-99.5th percentiles plus 3% axis-span margin; linear axes only.",
                "exclusions": [], "score_transform_verification_max_abs_delta": verify_delta,
                "inputs": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in source_files],
                "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                    "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}}
    save_json(out / "run_manifest.json", manifest)
    save_json(out / "output_inventory.json", sorted(p.name for p in out.iterdir() if p.name != "output_inventory.json"))
    print(out)


def write_report(out, upstream, records, summary, centroids, extreme, verify_delta):
    far = summary.loc[summary.groupby("gas_label").centroid_displacement_from_B1.idxmax()].sort_values("gas_label")
    dispersion = summary.loc[summary.groupby("gas_label").dispersion_ratio_to_B1.idxmax()].sort_values("gas_label")
    names = far.gas_name.tolist()
    shift_lines = "\n".join(f"- {r.gas_name}: largest displacement {r.centroid_displacement_from_B1:.2f} in Batch {int(r.batch)}; "
                            f"largest dispersion ratio {d.dispersion_ratio_to_B1:.2f}× in Batch {int(d.batch)}."
                            for r, d in zip(far.itertuples(), dispersion.itertuples()))
    off = centroids[centroids.outside_zoom]
    off_text = (", ".join(f"{r.gas_name} B{int(r.batch)}" for r in off.itertuples()) if len(off) else "none")
    extreme_lines = "\n".join(f"- `{r.sample_id}` ({r.extreme_for}; `{r.original_file}` line {int(r.line_number)}), "
                               f"Batch {int(r.batch)}, {r.gas_name}: PC1={r.PC1:.6g}, PC2={r.PC2:.6g}."
                               for r in extreme.itertuples())
    missing = centroids[centroids.sample_count == 0]
    missing_text = (", ".join(f"{r.gas_name} B{int(r.batch)}" for r in missing.itertuples()) if len(missing) else "none")
    text = f"""# Revised PCA visualization report

## Scope and reproducibility

This is a visualization-only derivative of `{upstream}`. It reuses the saved global and source-only score tables, StandardScalers, and PCA models. No model or scaler was fit, and no samples were removed, clipped, imputed, deduplicated, or changed. The saved source-only scores reproduce transformation by the saved Batch 1 scaler and PCA to a maximum absolute difference of {verify_delta:.3g}. Axes are linear.

Zoom limits use the pooled score distribution's 0.5th and 99.5th percentiles independently on PC1 and PC2, followed by a 3% margin. Global limits are PC1 [{records['global']['zoom']['x'][0]:.6g}, {records['global']['zoom']['x'][1]:.6g}] and PC2 [{records['global']['zoom']['y'][0]:.6g}, {records['global']['zoom']['y'][1]:.6g}], with {records['global']['points_outside_zoom']:,} points outside. Source-only limits are PC1 [{records['source_only']['zoom']['x'][0]:.6g}, {records['source_only']['zoom']['x'][1]:.6g}] and PC2 [{records['source_only']['zoom']['y'][0]:.6g}, {records['source_only']['zoom']['y'][1]:.6g}], with {records['source_only']['points_outside_zoom']:,} points outside. Counts by batch and gas are in the corresponding `*_zoom_outside_counts.csv` files. These viewports do not alter scores or calculations.

## Same-gas changes across batches

The combined plots encode gas by six fixed colors and batch by ten marker shapes, so shifts can be followed within gas without confusing gas identity with batch. The source-only panels use the same Batch 1 scaler and PCA basis throughout; their axis percentages are explicitly Batch 1 explained-variance ratios.

{shift_lines}

Displacement is Euclidean distance between each batch×gas full-data PC1–PC2 centroid and that gas's Batch 1 centroid. Dispersion is root-mean-square PC1–PC2 distance from the full-data gas×batch centroid. Exact values and sample counts are in `source_only_gas_batch_shift_summary.csv`. These gas-specific comparisons are primary. Whole-batch centroids are deliberately secondary because changing gas composition can move them.

The zoomed views reveal the dense within-gas clouds and moderate centroid paths that are compressed in the full-range figures. The full-range figures retain every projected point and are essential for judging tail behavior. Conclusions about apparent overlap, compactness, and visual separation are sensitive to the extreme points in full-range scaling; the numerical centroid and dispersion calculations themselves always use all observations. Off-screen full-data centroids in the trajectory zoom: {off_text}.

Unavailable centroids because the supplied batch×gas group has zero samples: {missing_text}. They are recorded with `sample_count=0` and blank centroid coordinates; no centroid is inferred or substituted. Trajectory lines break across these gaps.

These are measured distribution shifts in the PCA coordinate systems. They do not by themselves establish the cause as sensor drift, especially without concentration or other acquisition metadata.

## Extreme source-only projections

{extreme_lines}

`source_only_extreme_feature_contributions.csv` reports the 12 largest absolute signed feature contributions for both PC1 and PC2 for each selected sample, including raw feature value, Batch 1 mean and scale, Batch-1-standardized value, PCA center, loading, and signed contribution. Contributions use `(standardized_value - pca.mean_) * component_loading`; their sums reproduce the saved coordinates (residuals are recorded in `source_only_extreme_samples.csv`). Extremes are inspected, not classified as errors or evidence of omitted scaling.

## Figure guide

- `global_gas_batch_full.png` and `global_gas_batch_zoom.png`: all batches, gas colors and batch markers.
- `source_only_gas_batch_full.png` and `source_only_gas_batch_zoom.png`: same encoding in the Batch 1 coordinate system.
- `source_only_batch1_comparison_full.png` and `source_only_batch1_comparison_zoom.png`: common-limit 2×5 comparisons against a faint Batch 1 reference.
- `source_only_within_gas_trajectories_full.png` and `source_only_within_gas_trajectories_zoom.png`: gas-specific full-data centroids for every observed batch×gas group and sample clouds.
"""
    (out / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
