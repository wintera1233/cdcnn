#!/usr/bin/env python
"""Confusion matrix of the best baseline cell, R-fig-ps@lr0.0003.

Row-normalised (recall), pooled over Batches 2-10, summed over the three seeds.
Sequential single-hue blue ramp, light surface; a confusion matrix is a
magnitude encoding, so one hue light-to-dark, never a rainbow. Every cell is
labelled because the grid is 6x6 - small enough that direct labels beat a
colour-only read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import GAS_LABELS  # noqa: E402

# Sequential blue, steps 100 -> 700 of the reference ramp.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
CRITICAL = "#b42318"


def matrix(run: Path, cell: str) -> np.ndarray:
    results = json.loads((run / "target_results.json").read_text(encoding="utf-8"))
    entries = [r for r in results if (r.get("cell") or r["variant"]) == cell]
    if not entries:
        raise SystemExit(f"{cell} not found in {run}")
    total = np.zeros((6, 6), dtype=np.float64)
    for entry in entries:
        for batch in map(str, range(2, 11)):
            total += np.asarray(entry["confusion"][batch], dtype=np.float64)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--cell", default="R-fig-ps@lr0.0003")
    parser.add_argument("--out", default="reports/figures/confusion_best.png")
    args = parser.parse_args()

    counts = matrix(Path(args.run), args.cell)
    support = counts.sum(axis=1)
    recall = counts / support[:, None]
    names = [GAS_LABELS[i] for i in range(1, 7)]

    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    figure, axis = plt.subplots(figsize=(8.4, 6.4))
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)
    axis.imshow(recall, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    for row in range(6):
        for column in range(6):
            value = recall[row, column]
            if value < 0.0005:
                text, weight = "0", "normal"
            else:
                text, weight = f"{value:.2f}", "bold" if row == column else "normal"
            axis.text(column, row, text, ha="center", va="center", fontsize=10.5,
                      weight=weight,
                      color="#ffffff" if value > 0.45 else INK_MUTED)
            # 2px surface gap between fills
            axis.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1,
                                         fill=False, edgecolor=SURFACE, linewidth=2))

    axis.set_xticks(range(6), names, fontsize=10, color=INK)
    axis.set_yticks(range(6), [f"{n}\n{int(s/3):,}" for n, s in zip(names, support)],
                    fontsize=10, color=INK)
    axis.set_xlabel("predicted", fontsize=11, color=INK_MUTED, labelpad=8)
    axis.set_ylabel("true  (rows labelled with per-seed support)", fontsize=11,
                    color=INK_MUTED, labelpad=8)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    # The finding: one class has an empty diagonal. Find it rather than assume it.
    dead = int(np.argmin(recall.diagonal()))
    axis.add_patch(plt.Rectangle((-0.5, dead - 0.5), 6, 1, fill=False,
                                 edgecolor=CRITICAL, linewidth=2.0, zorder=5))
    axis.annotate(f"{names[dead]}: recall {recall[dead, dead]:.3f} in every batch",
                  xy=(5.5, dead), xytext=(6.1, dead), fontsize=10, color=CRITICAL,
                  va="center", annotation_clip=False)

    axis.set_title(f"{args.cell} — recall on Batches 2–10, three seeds pooled\n"
                   f"target mean 0.5222   (paper's ResNet 0.6344)",
                   fontsize=12.5, color=INK, pad=14, loc="left")
    figure.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
