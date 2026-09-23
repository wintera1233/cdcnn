#!/usr/bin/env python
"""Our feature generation block in the format of the paper's Fig. 5.

Fig. 5 plots, per gas and in one test domain (batch 2), the original features in
blue and the artificial ones the generation block produces in orange, each with
the region it spans. This draws the same thing for this project's implementation
so the two can be compared directly.

Post-hoc diagnostic: the frozen checkpoint is used only to compute features, and
batch 2 is read after every checkpoint was frozen and evaluated.
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
from scipy.spatial import ConvexHull

matplotlib.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import normalize  # noqa: E402
from src.data import GAS_LABELS, load_target  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.generate import FeatureGeneration  # noqa: E402
from src.model import to_input  # noqa: E402
from src.protocol import TargetAccessLog  # noqa: E402

ORIGINAL, ARTIFICIAL = "#2a78d6", "#eb6834"
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
# Fig. 5's panel order.
ORDER = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def trim(points: np.ndarray, keep: float) -> np.ndarray:
    """Drop the points furthest from the centroid.

    The hulls are meant to show where a cloud lives, and a convex hull is
    dragged out of shape by a single stray point. Trimming is for the drawing
    only; no number reported anywhere in this project is computed on trimmed
    data.
    """
    if len(points) < 4:
        return points
    distance = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    return points[distance <= np.quantile(distance, keep)]


def hull(axis, points, colour):
    if len(points) < 3:
        return
    try:
        vertices = points[ConvexHull(points).vertices]
    except Exception:
        return
    axis.add_patch(Polygon(vertices, closed=True, facecolor=colour, alpha=0.22,
                           edgecolor=colour, linewidth=1.3, zorder=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="position",
                        choices=("scalar", "channel", "position"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--keep", type=float, default=0.95,
                        help="fraction of points kept when drawing; 1.0 keeps all")
    parser.add_argument("--out", default="reports/figures/feature_generation.png")
    args = parser.parse_args()

    checkpoint = glob.glob(
        "runs/*input_norm_5seed/checkpoints/R-fig-logps_lr0.0003_seed1042/final.pt")[0]
    model, payload = load_checkpoint(Path(checkpoint), "cpu")
    block = FeatureGeneration(style=args.style)

    log = TargetAccessLog()
    log.record_freeze(Path("post-hoc"), "diagnostic")
    x, y = load_target(args.batch, log)

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

    # The block is applied to the whole batch at once, mixed gases, because
    # Eqs. (12)-(13) estimate the style distribution from the spread *across* the
    # batch. Applied within one gas that spread is tiny and the restyle is
    # nearly an identity; during training the minibatch is mixed too.
    original_all = features(x)
    artificial_all = block(original_all, torch.Generator().manual_seed(args.seed))

    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    figure.patch.set_facecolor(SURFACE)
    for position, name in enumerate(ORDER):
        axis = axes.ravel()[position]
        axis.set_facecolor(SURFACE)
        label = next(k for k, v in GAS_LABELS.items() if v == name)
        rows = x[y == label]
        if len(rows) < 3:
            axis.set_title(name, fontsize=12.5, color=INK)
            axis.text(0.5, 0.5, "樣本過少", transform=axis.transAxes,
                      ha="center", color=INK_MUTED)
            continue
        mask = torch.as_tensor(y == label)
        original = original_all[mask]
        artificial = artificial_all[mask]
        both = torch.cat([original, artificial]).flatten(1).numpy()
        centred = both - both.mean(axis=0)
        _, _, components = np.linalg.svd(centred, full_matrices=False)
        projected = centred @ components[:2].T
        a, b = projected[:len(rows)], projected[len(rows):]

        a, b = trim(a, args.keep), trim(b, args.keep)
        hull(axis, a, ORIGINAL)
        hull(axis, b, ARTIFICIAL)
        axis.scatter(b[:, 0], b[:, 1], s=9, color=ARTIFICIAL, alpha=0.75,
                     linewidths=0, zorder=3, label="Artificial feature")
        axis.scatter(a[:, 0], a[:, 1], s=9, color=ORIGINAL, alpha=0.85,
                     linewidths=0, zorder=4, label="Original feature")
        axis.set_title(name, fontsize=12.5, color=INK)
        axis.set_xlabel("PC1", fontsize=9, color=INK_MUTED)
        axis.set_ylabel("PC2", fontsize=9, color=INK_MUTED)
        axis.tick_params(labelsize=8, colors=INK_MUTED)
        for spine in axis.spines.values():
            spine.set_color("#d9d8d4")

    handles, names = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles[::-1], names[::-1], loc="upper center", ncol=2,
                  frameon=True, fontsize=10.5, markerscale=2.2,
                  bbox_to_anchor=(0.5, 0.995))
    figure.suptitle(
        f"The outcome of the feature generation block in one of the test domains "
        f"(batch{args.batch})", fontsize=13, color=INK, y=0.925)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
