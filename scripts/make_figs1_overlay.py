#!/usr/bin/env python
"""Overlay this project's per-epoch curves on the paper's Fig. S1.

Fig. S1(a) plots accuracy over 100 epochs "in a training set" for Resnet, CDWC
and CDCNN, plateauing near 0.58, 0.65 and 0.70. Those numbers are close to the
models' *target* means in Table 3 (0.6344, 0.6705, 0.7230) and far from any
training accuracy this project reaches. The caption does not say which it is.

This figure answers that by plotting both of ours against the paper's levels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Read off Fig. S1(a); the plateau each curve settles to.
PAPER_PLATEAU = {"Resnet": 0.58, "CDWC": 0.65, "CDCNN": 0.70}
# Table 3, for the comparison the figure invites.
PAPER_TARGET_MEAN = {"Resnet": 0.6344, "CDWC": 0.6705, "CDCNN": 0.7230}

COLOURS = {"R-txt": "#1E2761", "R-txt-ps": "#C1440E",
           "R-lite": "#4B77BE", "R-lite-ps": "#E8A33D"}
LABELS = {"R-txt": "R-txt (flatten, scaler)",
          "R-txt-ps": "R-txt-ps (flatten, per-sample)",
          "R-lite": "R-lite (GAP, scaler)",
          "R-lite-ps": "R-lite-ps (GAP, per-sample)"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default="reports/figures/figS1_overlay.png")
    args = parser.parse_args()

    curves = json.loads(
        (Path(args.run) / "epoch_curves.json").read_text(encoding="utf-8"))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)

    for name, points in sorted(curves.items()):
        variant = name.rsplit("_seed", 1)[0]
        colour = COLOURS.get(variant, "#666666")
        epochs = [p["epoch"] for p in points]
        left.plot(epochs, [p["source_accuracy"] for p in points],
                  color=colour, linewidth=1.8, label=LABELS.get(variant, variant))
        right.plot(epochs, [p["target_mean"] for p in points],
                   color=colour, linewidth=1.8)

    for axis, title in ((left, "this project: Batch 1 (training) accuracy"),
                        (right, "this project: target mean, Batches 2-10")):
        for label, level in PAPER_PLATEAU.items():
            axis.axhline(level, color="#999999", linestyle=":", linewidth=1.1)
            axis.annotate(f"paper Fig. S1  {label} {level:.2f}", xy=(2, level),
                          xytext=(2, level + 0.012), fontsize=8, color="#555555")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 1.03)
        axis.set_xlabel("epoch")
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.25, linewidth=0.6)
    left.set_ylabel("accuracy")

    handles = [Line2D([], [], color=COLOURS[v], linewidth=1.8, label=LABELS[v])
               for v in ("R-txt", "R-txt-ps", "R-lite", "R-lite-ps") if
               any(name.startswith(v + "_seed") for name in curves)]
    handles.append(Line2D([], [], color="#999999", linestyle=":", linewidth=1.1,
                          label="paper Fig. S1 plateaus"))
    left.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)

    figure.suptitle(
        "Is Fig. S1(a) a training curve or a target curve?  (seed 1042)",
        fontsize=13, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
