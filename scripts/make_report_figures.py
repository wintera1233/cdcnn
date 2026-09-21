#!/usr/bin/env python3
"""Build the figures for the experiment review in reports/.

Reads completed run artifacts only; writes PNGs into reports/figures/. Colors are
the validated default palette from the dataviz skill, used unchanged: categorical
slots 1-3 (blue, orange, aqua), the blue sequential ramp for magnitude, and the
blue/red diverging pair for signed contributions. Figure text is English so the
PNGs render identically everywhere; the slide text around them is Chinese.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
POS, NEG = "#2a78d6", "#d03b3b"                      # diverging poles
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GAS = {1: "Acetone", 2: "Acetaldehyde", 3: "Ethanol", 4: "Ethylene", 5: "Ammonia", 6: "Toluene"}
PAPER = {"ResNet": 0.6344, "CDWC": 0.6705, "CDCNN": 0.7230}
PAPER_BATCH = [0.8271, 0.7683, 0.7244, 0.7891, 0.9318, 0.6182, 0.7152, 0.5664, 0.5663]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def tidy(ax, xgrid=False, ygrid=False):
    ax.grid(axis="y" if ygrid else "x", color="#e6e5e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if not xgrid and not ygrid:
        ax.grid(False)


def latest(pattern: str) -> Path:
    runs = sorted((ROOT / "runs").glob(pattern))
    if not runs:
        sys.exit(f"no run matching {pattern}")
    return runs[-1]


def stage_table() -> pd.DataFrame:
    """Every stage with a completed, audited target evaluation."""
    rows = {}
    for run in sorted((ROOT / "runs").iterdir()):
        ss, au = run / "stage_summary.csv", run / "leakage_target_access_audit.json"
        pb = run / "per_batch_summary.csv"
        if not (ss.is_file() and au.is_file() and pb.is_file()):
            continue
        if json.loads(au.read_text())["status"] != "passed":
            continue
        batch = pd.read_csv(pb).pivot(index="batch", columns="stage", values="mean")
        for s, r in pd.read_csv(ss).set_index("stage").iterrows():
            rows.setdefault(s, {"cv": r.source_cv_mean, "tm": r.target_mean,
                                **{b: batch.loc[b, s] for b in range(2, 11)}})
    return pd.DataFrame(rows).T


def fig_progress(stages: pd.DataFrame) -> None:
    order = ["A1", "A2-semantic", "B0", "A3", "B0-LN", "B0-LN-PS", "B0-stab-PS"]
    labels = {"A1": "A1  augmentation", "A2-semantic": "A2  feature generation",
              "B0": "B0  baseline", "A3": "A3  full CDCNN (v6.3)",
              "B0-LN": "+ LayerNorm", "B0-LN-PS": "+ per-sample inputs",
              "B0-stab-PS": "+ bounds (best)"}
    values = [stages.loc[s, "tm"] for s in order]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(order))
    ax.barh(y, values, color=SERIES[0], height=0.62, zorder=3)
    for yi, v in zip(y, values):
        ax.text(v + 0.006, yi, f"{v:.3f}", va="center", ha="left", fontsize=10, color=INK)
    for i, (name, value) in enumerate(PAPER.items()):
        ax.axvline(value, color=MUTED, linewidth=1, linestyle="--", zorder=2)
        ax.annotate(f"paper {name}\n{value:.3f}", xy=(value, len(order) - 0.45),
                    xytext=(0, 6 + 21 * i), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=MUTED,
                    annotation_clip=False)
    ax.set_yticks(y, [labels[s] for s in order])
    ax.set_xlim(0, 0.80)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("Target mean accuracy, Batches 2-10 (5 seeds)")
    ax.set_title("Where the accuracy came from", loc="left", fontsize=13, color=INK, pad=62)
    tidy(ax, ygrid=True)
    fig.tight_layout(); fig.savefig(OUT / "fig1_progress.png", dpi=200); plt.close(fig)


def fig_contribution() -> None:
    items = [("Per-sample input normalization", 0.1087), ("LayerNorm instead of BatchNorm", 0.0845),
             ("Epoch alignment", 0.0043), ("Feature generation (paper's branch)", 0.0044),
             ("Contrastive loss", 0.0025), ("Hard bounds + gradient clipping", 0.0044),
             ("Augmentation (any noise scale)", -0.0131)]
    items.sort(key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(items))
    values = [v for _, v in items]
    ax.barh(y, values, color=[POS if v > 0 else NEG for v in values], height=0.62, zorder=3)
    for yi, v in zip(y, values):
        ax.text(v + (0.002 if v > 0 else -0.002), yi, f"{v:+.4f}", va="center",
                ha="left" if v > 0 else "right", fontsize=10, color=INK)
    ax.axvline(0, color="#c9c8c3", linewidth=1)
    ax.set_yticks(y, [k for k, _ in items])
    ax.set_xlim(-0.03, 0.13)
    ax.set_xlabel("Change in target mean accuracy")
    ax.set_title("Contribution of each change", loc="left", fontsize=13, color=INK, pad=12)
    tidy(ax, ygrid=True)
    fig.tight_layout(); fig.savefig(OUT / "fig2_contribution.png", dpi=200); plt.close(fig)


def fig_per_batch(stages: pd.DataFrame) -> None:
    batches = list(range(2, 11))
    series = [("B0 baseline", [stages.loc["B0", b] for b in batches], SERIES[0]),
              ("This project, best", [stages.loc["B0-stab-PS", b] for b in batches], SERIES[1]),
              ("Paper CDCNN", PAPER_BATCH, SERIES[2])]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(batches)); width = 0.27
    for i, (name, values, color) in enumerate(series):
        ax.bar(x + (i - 1) * width, values, width * 0.92, label=name, color=color, zorder=3)
    ax.set_xticks(x, [f"B{b}" for b in batches])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-batch accuracy: the gap is Batches 5-10", loc="left",
                 fontsize=13, color=INK, pad=34)
    ax.legend(frameon=False, ncols=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=10)
    tidy(ax, xgrid=True)
    fig.tight_layout(); fig.savefig(OUT / "fig3_per_batch.png", dpi=200); plt.close(fig)


def fig_class_recall() -> None:
    run = latest("*_cdcnn_v6_9_epoch_aligned_full")
    c = pd.read_csv(run / "confusion_matrices.csv")
    c = c[c.stage == "B0-stab-PS"]
    grid = np.full((6, 9), np.nan)
    for gi, g in enumerate(range(1, 7)):
        for bi, b in enumerate(range(2, 11)):
            m = c[(c.batch == b) & (c.true_gas_label == g)]
            total = m["count"].sum()
            if total:
                grid[gi, bi] = m[m.predicted_gas_label == g]["count"].sum() / total
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for gi in range(6):
        for bi in range(9):
            v = grid[gi, bi]
            if np.isnan(v):
                ax.text(bi, gi, "-", ha="center", va="center", color=MUTED, fontsize=10)
            else:
                ax.text(bi, gi, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                        color="#ffffff" if v > 0.55 else INK)
    ax.set_xticks(range(9), [f"B{b}" for b in range(2, 11)])
    ax.set_yticks(range(6), [GAS[g] for g in range(1, 7)])
    ax.set_title("Per-class recall collapses for three gases", loc="left",
                 fontsize=13, color=INK, pad=12)
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig4_class_recall.png", dpi=200); plt.close(fig)


def fig_confusion() -> None:
    run = latest("*_cdcnn_v6_9_epoch_aligned_full")
    c = pd.read_csv(run / "confusion_matrices.csv")
    c = c[c.stage == "B0-stab-PS"]
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, batch in zip(axes, (6, 8)):
        m = (c[c.batch == batch].groupby(["true_gas_label", "predicted_gas_label"])["count"]
             .sum().unstack(fill_value=0).reindex(index=range(1, 7), columns=range(1, 7),
                                                  fill_value=0))
        rec = m.div(m.sum(axis=1).replace(0, 1), axis=0).to_numpy()
        ax.imshow(rec, cmap=cmap, vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                if rec[i, j] >= 0.01:
                    ax.text(j, i, f"{rec[i, j]:.2f}", ha="center", va="center", fontsize=9,
                            color="#ffffff" if rec[i, j] > 0.55 else INK)
        ax.set_xticks(range(6), [GAS[g][:9] for g in range(1, 7)], rotation=45, ha="right")
        ax.set_yticks(range(6), [GAS[g] for g in range(1, 7)])
        ax.set_title(f"Batch {batch}", loc="left", fontsize=12, color=INK)
        ax.tick_params(length=0); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("True gas")
    fig.suptitle("Where the errors go: Ammonia and Toluene become Acetone or Ethanol",
                 x=0.012, ha="left", fontsize=13, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "fig5_confusion.png", dpi=200); plt.close(fig)


def fig_balance() -> None:
    run = latest("*_cdcnn_v6_11_class_balance_one_seed_pilot")
    cv = pd.read_csv(run / "source_cv_predictions.csv.gz")
    tgt = pd.read_csv(run / "target_predictions.csv.gz")
    stages = ["B0-stab-PS", "B0-wce-PS", "B0-bal-PS"]
    names = ["none", "class weights", "balanced sampling"]
    source = [(cv[(cv.stage == s) & (cv.true_gas_label == 4)].predicted_gas_label == 4).mean()
              for s in stages]
    target = [(tgt[(tgt.stage == s) & (tgt.true_gas_label == 4)].predicted_gas_label == 4).mean()
              for s in stages]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(3); width = 0.34
    ax.bar(x - width / 2, source, width * 0.92, label="Batch 1 CV (source)", color=SERIES[0], zorder=3)
    ax.bar(x + width / 2, target, width * 0.92, label="Batches 2-10 (target)", color=SERIES[1], zorder=3)
    for xi, v in zip(x - width / 2, source):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    for xi, v in zip(x + width / 2, target):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Ethylene recall")
    ax.set_title("Class balancing fixes Ethylene in-domain and nothing on the targets",
                 loc="left", fontsize=13, color=INK, pad=12)
    ax.legend(frameon=False, fontsize=10)
    tidy(ax, xgrid=True)
    fig.tight_layout(); fig.savefig(OUT / "fig6_class_balance.png", dpi=200); plt.close(fig)


# Figures lifted straight from the paper, for the literature-review slides.
# PDF images are numbered in page order by pdfimages; the docx keeps its own
# media order. Both are verified by size and page in the mapping below.
PAPER_PDF_FIGURES = {
    "paper_fig1_concept.png": 0,      # Fig 1  deep learning vs domain generalization, contrastive sphere
    "paper_fig4_aug_pca.png": 3,      # Fig 4  PCA of raw / gauss / constant / CDCNN augmentation
    "paper_fig5_featgen.png": 4,      # Fig 5  original vs artificial feature space per gas
    "paper_fig6_comparison.png": 5,   # Fig 6  accuracy of every compared algorithm
}
PAPER_DOCX_FIGURES = {
    "paper_figS4_blocks.png": "image35.png",   # Fig S4  conv block and feature generation block
    "paper_figS5_augment.png": "image36.png",  # Fig S5  data augmentation block
    "paper_figS3_confusion.png": "image34.png",  # Fig S3  CDCNN and CDWC confusion matrices
}


def extract_paper_figures() -> None:
    """Pull the paper's own figures out of the PDF and the supplement."""
    import shutil
    import subprocess
    import tempfile
    import zipfile

    pdf = ROOT / "docs/paper/1-s2.0-S0924424724003078-main.pdf"
    docx = ROOT / "docs/paper/1-s2.0-S0924424724003078-mmc1.docx"
    if not (pdf.is_file() and docx.is_file()):
        print("  paper sources missing; skipping paper figures")
        return
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdfimages", "-png", "-f", "2", "-l", "8", str(pdf),
                        str(Path(tmp) / "fig")], check=True)
        for name, index in PAPER_PDF_FIGURES.items():
            src = Path(tmp) / f"fig-{index:03d}.png"
            if src.is_file():
                shutil.copyfile(src, OUT / name)
        with zipfile.ZipFile(docx) as archive:
            for name, member in PAPER_DOCX_FIGURES.items():
                (OUT / name).write_bytes(archive.read(f"word/media/{member}"))
    framework = ROOT / "docs/paper/fig2_cdcnn.png"
    if framework.is_file():
        shutil.copyfile(framework, OUT / "paper_fig2_framework.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stages = stage_table()
    fig_per_batch(stages)
    fig_class_recall()
    fig_confusion()
    extract_paper_figures()
    paper = ROOT / "docs/paper/figS3_confusion.png"
    if paper.is_file():
        (OUT / "fig7_paper_confusion.png").write_bytes(paper.read_bytes())
    print("figures written to", OUT)
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name:<28} {f.stat().st_size // 1024:4d} KB")


if __name__ == "__main__":
    main()
