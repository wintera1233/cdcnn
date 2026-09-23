#!/usr/bin/env python
"""Does the block's artificial cloud cover where the target batches actually are?

The paper's Fig. 5 draws the block's output "in one of the test domains
(batch2)": it applies the block to batch 2 and plots the result beside batch 2's
own features, then argues the artificial features are "more dispersed, covering a
wider spatial range, which helps to represent the drift effect better". That
comparison is circular. Artificial features generated *from* batch 2 sit near
batch 2 by construction, whatever the block does, so the picture cannot be
evidence that the block reaches an unseen domain.

`scripts/make_feature_generation_figure.py` reproduces Fig. 5 as printed, for
comparison with the paper. This draws the non-circular version instead.

During training the block only ever sees Batch 1, so the artificial features the
network is actually shown are generated from Batch 1. Those are the orange cloud
here, and they are **identical in every row** - one fixed reference region per
gas. Each row's blue cloud is a different target batch's real features. The
question the figure asks is the one the method needs answered: does the orange
region contain the blue?

The panels of a column share one PCA basis, fitted on everything drawn in that
column, so the rows are comparable and the orange really is the same shape.

Post-hoc diagnostic. The frozen checkpoint computes features; the target batches
are read after every checkpoint was frozen and evaluated.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

matplotlib.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import normalize  # noqa: E402
from src.data import GAS_LABELS, load_source, load_target  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.generate import FeatureGeneration  # noqa: E402
from src.model import to_input  # noqa: E402
from src.protocol import TargetAccessLog  # noqa: E402

TARGET_C, ARTIFICIAL = "#2a78d6", "#eb6834"
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
ORDER = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def trim(points: np.ndarray, keep: float) -> np.ndarray:
    """The points in dense regions, by distance to the k-th nearest neighbour.

    Decides which points the hull encloses; every point is still drawn. A hull is
    dragged out of shape by one stray, and these clouds are thin curved arcs, so
    distance from the centroid would cut the ends off the arc while keeping a
    stray near the middle. Local density is what "outlier" means here.

    Drawing only. The coverage percentages below are computed on the full set.
    """
    if len(points) < 8 or keep >= 1.0:
        return points
    k = max(3, int(round(0.05 * len(points))))
    distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    distances.sort(axis=1)
    return points[distances[:, k] <= np.quantile(distances[:, k], keep)]


def hull_vertices(points: np.ndarray):
    if len(points) < 3:
        return None
    try:
        return points[ConvexHull(points).vertices]
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="position",
                        choices=("scalar", "channel", "position"))
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--keep", type=float, default=0.88)
    parser.add_argument("--out",
                        default="reports/figures/feature_generation_coverage.png")
    args = parser.parse_args()

    checkpoint = glob.glob(
        "runs/*input_norm_5seed/checkpoints/R-fig-logps_lr0.0003_seed1042/final.pt")[0]
    model, payload = load_checkpoint(Path(checkpoint), "cpu")
    block = FeatureGeneration(style=args.style)
    log = TargetAccessLog()
    log.record_freeze(Path("post-hoc"), "diagnostic")

    @torch.no_grad()
    def features(rows: np.ndarray) -> torch.Tensor:
        inp = to_input(normalize.apply(payload["normalizer"], rows))
        out = []
        for start in range(0, len(inp), 256):
            h = inp[start:start + 256]
            for index, layer in enumerate(model.blocks):
                h = layer(h)
                if index == 2:
                    break
            out.append(h)
        return torch.cat(out)

    # The artificial cloud: generated from Batch 1, which is the only thing the
    # block ever sees in training. Applied to the whole source at once, mixed
    # gases, because Eqs. (12)-(13) estimate the style spread *across* the batch.
    source_x, source_y = load_source()
    artificial = block(features(source_x), torch.Generator().manual_seed(args.seed))
    domains = [(b, *load_target(b, log)) for b in args.batches]
    target_features = {b: features(x) for b, x, _ in domains}

    figure, axes = plt.subplots(len(args.batches), len(ORDER),
                                figsize=(3.2 * len(ORDER), 2.95 * len(args.batches)))
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor(SURFACE)

    for column, name in enumerate(ORDER):
        label = next(k for k, v in GAS_LABELS.items() if v == name)
        orange = artificial[torch.as_tensor(source_y == label)].flatten(1).numpy()
        blues = {b: target_features[b][torch.as_tensor(y == label)].flatten(1).numpy()
                 for b, _, y in domains}

        # One basis per column, fitted on everything the column draws.
        pooled = np.vstack([orange] + [v for v in blues.values() if len(v)])
        centre = pooled.mean(axis=0)
        _, _, components = np.linalg.svd(pooled - centre, full_matrices=False)
        basis = components[:2].T
        orange_2d = (orange - centre) @ basis
        blues_2d = {b: ((v - centre) @ basis if len(v) else None)
                    for b, v in blues.items()}

        dense_orange = trim(orange_2d, args.keep)
        vertices = hull_vertices(dense_orange)
        inside = MplPath(vertices) if vertices is not None else None

        spread = [dense_orange] + [trim(v, args.keep)
                                   for v in blues_2d.values() if v is not None
                                   and len(v) >= 3]
        dense = np.vstack(spread)
        limits = [(dense[:, i].min() - (np.ptp(dense[:, i]) * 0.20 or 1.0),
                   dense[:, i].max() + (np.ptp(dense[:, i]) * 0.20 or 1.0))
                  for i in (0, 1)]

        for row, (batch, _, _) in enumerate(domains):
            axis = axes[row, column]
            axis.set_facecolor(SURFACE)
            for spine in axis.spines.values():
                spine.set_color("#d9d8d4")
            axis.tick_params(labelsize=8, colors=INK_MUTED)
            if row == 0:
                axis.set_title(name, fontsize=11.5, color=INK)
            if column == 0:
                axis.set_ylabel(f"batch {batch}", fontsize=10.5, color=INK)

            blue = blues_2d[batch]
            if vertices is not None:
                axis.add_patch(Polygon(vertices, closed=True, facecolor=ARTIFICIAL,
                                       alpha=0.20, edgecolor=ARTIFICIAL,
                                       linewidth=1.3, zorder=1))
            axis.scatter(orange_2d[:, 0], orange_2d[:, 1], s=7, color=ARTIFICIAL,
                         alpha=0.55, linewidths=0, zorder=2,
                         label="Artificial feature (from Batch 1)")
            if blue is None or not len(blue):
                axis.set_xticks([]); axis.set_yticks([])
                axis.text(0.5, 0.06, "n = 0", transform=axis.transAxes,
                          ha="center", fontsize=9, color=INK_MUTED)
                continue
            axis.scatter(blue[:, 0], blue[:, 1], s=9, color=TARGET_C, alpha=0.85,
                         linewidths=0, zorder=4,
                         label=f"Original feature (target batch)")
            axis.set_xlim(*limits[0]); axis.set_ylim(*limits[1])
            if inside is not None:
                covered = float(inside.contains_points(blue).mean())
                axis.text(0.03, 0.94, f"{covered:.0%} inside", transform=axis.transAxes,
                          fontsize=9.5, va="top", color=INK,
                          bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                                    edgecolor="#d9d8d4", linewidth=0.8))

    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles[::-1], names[::-1], loc="upper center", ncol=2,
                  frameon=True, fontsize=10.5, markerscale=2.2,
                  bbox_to_anchor=(0.5, 0.995))
    top = 1 - 0.58 / figure.get_figheight()
    figure.suptitle(
        "Do the features generated from Batch 1 cover the real target domains? "
        "(percentages are of target points inside the orange hull, in this 2D projection)",
        fontsize=12.5, color=INK, y=top - 0.010)
    figure.tight_layout(rect=(0, 0, 1, top - 0.028))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
