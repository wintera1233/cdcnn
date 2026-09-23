#!/usr/bin/env python
"""Drift-aligned 2D scatter: Batch 1, the block's artificial cloud, the target.

A companion to `make_generation_drift_projection_figure.py`, which reduces each
(gas, batch) to one axis. This keeps that axis and adds a second one so the
clouds can be seen as clouds:

    axis 1  u = (mu_target - mu_source) / ||.||      the real drift direction
    axis 2  the first principal component of the residual after the component
            along u is removed from every point (Batch 1, artificial and target
            pooled) - the largest variance orthogonal to the drift

Unlike a plain 2D PCA the drift axis is fixed by construction, so a cloud that
does not travel along `u` cannot hide behind a basis chosen to spread the data.
Coordinates are in block-3 units, origin at the Batch 1 centroid, and both axes
share one scale within a panel so distances read honestly.

The artificial cloud is generated from Batch 1, the only thing training shows the
network, with the training minibatching. Post-hoc diagnostic on a frozen
checkpoint; the target batches were opened and audited by the v9.0 run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

matplotlib.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import normalize  # noqa: E402
from src.data import GAS_LABELS, TARGET_BATCHES, load_source, load_target  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.generate import FeatureGeneration  # noqa: E402
from src.model import to_input  # noqa: E402
from src.protocol import TargetAccessLog  # noqa: E402

CHECKPOINT = ("runs/20260923T071856507630Z_baseline_ladder_full/checkpoints/"
              "R-gen_lr0.0003_seed1042/final.pt")
SOURCE_C, ARTIFICIAL_C, TARGET_C = "#8a8983", "#eb6834", "#2a78d6"
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
ORDER = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def drift_aligned_basis(source: np.ndarray, artificial: np.ndarray,
                        target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(W, origin)`: W is [D, 2], column 0 the drift direction, column 1 the
    leading residual direction orthogonal to it."""
    mu_source, mu_target = source.mean(axis=0), target.mean(axis=0)
    d = mu_target - mu_source
    v1 = d / np.linalg.norm(d)
    pooled = np.vstack([source, artificial, target])
    centred = pooled - pooled.mean(axis=0)
    residual = centred - np.outer(centred @ v1, v1)
    _, _, vh = np.linalg.svd(residual, full_matrices=False)
    v2 = vh[0]
    return np.column_stack([v1, v2]), mu_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--style", default="position",
                        choices=("scalar", "channel", "position"))
    parser.add_argument("--batches", type=int, nargs="+", default=list(TARGET_BATCHES))
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out",
                        default="reports/figures/generation_drift_aligned_2d.png")
    args = parser.parse_args()

    model, payload = load_checkpoint(Path(args.checkpoint), "cpu")
    normaliser = payload["normalizer"]
    block = FeatureGeneration(style=args.style)
    log = TargetAccessLog()
    log.record_freeze(Path("post-hoc"), "diagnostic")

    @torch.no_grad()
    def block3(rows: np.ndarray, size: int = 256) -> torch.Tensor:
        inp = to_input(normalize.apply(normaliser, rows))
        return torch.cat([model.stem(inp[i:i + size])
                          for i in range(0, len(inp), size)])

    source_x, source_y = load_source()
    source = block3(source_x)
    generator = torch.Generator().manual_seed(args.seed)
    parts, origin = [], []
    with torch.no_grad():
        for _ in range(args.passes):
            order = torch.randperm(len(source), generator=generator)
            for start in range(0, len(order), args.minibatch):
                index = order[start:start + args.minibatch]
                if len(index) < 2:
                    continue
                parts.append(block(source[index], generator=generator))
                origin.append(index.numpy())
    artificial = torch.cat(parts).flatten(1).numpy()
    artificial_y = source_y[np.concatenate(origin)]
    source = source.flatten(1).numpy()

    targets = {b: load_target(b, log) for b in args.batches}
    target_features = {b: block3(x).flatten(1).numpy() for b, (x, _) in targets.items()}

    rows, cols = len(args.batches), len(ORDER)
    figure, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.85 * rows))
    axes = np.atleast_2d(axes)
    figure.patch.set_facecolor(SURFACE)

    for column, name in enumerate(ORDER):
        label = next(k for k, v in GAS_LABELS.items() if v == name)
        src = source[source_y == label]
        art = artificial[artificial_y == label]
        for row, batch in enumerate(args.batches):
            axis = axes[row, column]
            axis.set_facecolor(SURFACE)
            for spine in axis.spines.values():
                spine.set_color(GRID)
            axis.tick_params(labelsize=7.5, colors=INK_MUTED)
            if row == 0:
                axis.set_title(name, fontsize=11.5, color=INK)
            if column == 0:
                axis.set_ylabel(f"batch {batch}", fontsize=10.5, color=INK)

            _, target_y = targets[batch]
            tgt = target_features[batch][target_y == label]
            if not len(tgt):
                axis.set_xticks([]); axis.set_yticks([])
                axis.text(0.5, 0.5, "n = 0", transform=axis.transAxes,
                          ha="center", va="center", fontsize=9, color=INK_MUTED)
                continue

            W, origin_point = drift_aligned_basis(src, art, tgt)
            s2, a2, t2 = ((z - origin_point) @ W for z in (src, art, tgt))
            distance = float(np.linalg.norm(tgt.mean(axis=0) - src.mean(axis=0)))

            axis.axvline(0.0, color=GRID, linewidth=1.0, zorder=0)
            axis.axvline(distance, color=GRID, linewidth=1.0, zorder=0)
            axis.axhline(0.0, color=GRID, linewidth=1.0, zorder=0)
            axis.scatter(a2[:, 0], a2[:, 1], s=6, color=ARTIFICIAL_C, alpha=0.35,
                         linewidths=0, zorder=1)
            axis.scatter(s2[:, 0], s2[:, 1], s=7, color=SOURCE_C, alpha=0.6,
                         linewidths=0, zorder=2)
            axis.scatter(t2[:, 0], t2[:, 1], s=7, color=TARGET_C, alpha=0.6,
                         linewidths=0, zorder=3)
            for cloud, colour in ((s2, SOURCE_C), (a2, ARTIFICIAL_C), (t2, TARGET_C)):
                c = cloud.mean(axis=0)
                axis.scatter([c[0]], [c[1]], s=48, color=colour, edgecolor=SURFACE,
                             linewidths=1.5, zorder=5)

            everything = np.vstack([s2, a2, t2])
            lo = np.quantile(everything, 0.005, axis=0)
            hi = np.quantile(everything, 0.995, axis=0)
            half = 0.55 * max(hi - lo)
            centre = 0.5 * (lo + hi)
            axis.set_xlim(centre[0] - half, centre[0] + half)
            axis.set_ylim(centre[1] - half, centre[1] + half)
            axis.set_aspect("equal")
            axis.text(0.03, 0.95, f"n = {len(tgt)}", transform=axis.transAxes,
                      fontsize=8, va="top", color=INK_MUTED)

    for axis in axes[-1]:
        axis.set_xlabel("axis 1: real drift direction u", fontsize=8.5, color=INK_MUTED)
    for axis in axes[:, 0]:
        axis.set_ylabel(axis.get_ylabel() + "\naxis 2: residual PC1", fontsize=8.5,
                       color=INK)

    handles = [Line2D([], [], linestyle="none", marker="o", markersize=7,
                      markerfacecolor=c, markeredgecolor=SURFACE)
               for c in (SOURCE_C, ARTIFICIAL_C, TARGET_C)]
    figure.legend(handles, ["Batch 1, real features",
                            "Artificial features generated from Batch 1",
                            "Target batch, real features"],
                  loc="upper center", ncol=3, frameon=False, fontsize=10.5,
                  bbox_to_anchor=(0.5, 0.985))
    figure.suptitle(
        "Drift-aligned 2D projection at block 3: x = the (gas, batch) drift direction, "
        "y = the leading direction orthogonal to it. Vertical lines: Batch 1 centroid "
        "and target centroid; large dots: cloud centroids.",
        fontsize=12, color=INK, y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=130, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
