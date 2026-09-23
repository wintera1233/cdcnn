#!/usr/bin/env python
"""How far along the real drift axis does the block's artificial cloud reach?

`docs/v9-feature-generation.md` records three coverage measurements that
contradict each other: a 2D-PCA convex hull (0-3 %, but blind to the other
16,382 axes), a full-dimensional "within 2 radii" test (0.90-1.00 everywhere,
because the threshold exceeds every source-to-target centroid distance, so the
metric is saturated) and a bare centroid-distance comparison (which says nothing
about the clouds' extent). This is the one-dimensional measurement that document
asked for instead.

For each gas `c` and each target batch `b`, at the block-3 output where the
block acts:

    u = (mu_target - mu_source) / ||mu_target - mu_source||

Every block-3 feature is projected onto `u` and the coordinate is expressed as a
fraction of the drift: the source centroid sits at 0 and the target centroid at
1. Three clouds are drawn per (gas, batch): Batch 1's real features, the
artificial features the block generates **from Batch 1** - the only thing the
network is shown in training - and the target batch's real features. The
question is whether the orange cloud reaches the blue one.

This avoids both failure modes above: it is not a 2D projection, and it is not a
high-dimensional ball whose volume is empty. It also speaks directly to the sign
diagnosis in `docs/v9-feature-generation.md`: if Eq. (14) displaces
symmetrically, the artificial cloud should straddle the source centroid rather
than lean toward 1.

Post-hoc diagnostic on a frozen checkpoint. The target batches were opened and
audited by the v9.0 run; nothing here selects anything.
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
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def overlap_coefficient(a: np.ndarray, b: np.ndarray, bins: int = 60) -> float:
    """Shared area of two 1-D histograms on common bins, in [0, 1]."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, edges, density=True)
    pb, _ = np.histogram(b, edges, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(pa, pb).sum() * width)


def range_bar(axis, values: np.ndarray, y: float, colour: str) -> None:
    """5-95 % as a hairline, 25-75 % as the bar, median as a ringed dot."""
    q05, q25, q50, q75, q95 = np.quantile(values, QUANTILES)
    axis.plot([q05, q95], [y, y], color=colour, linewidth=1.0,
              solid_capstyle="round", zorder=2)
    axis.plot([q25, q75], [y, y], color=colour, linewidth=4.5,
              solid_capstyle="round", zorder=3)
    axis.scatter([q50], [y], s=34, color=colour, edgecolor=SURFACE,
                 linewidths=1.4, zorder=4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--style", default="position",
                        choices=("scalar", "channel", "position"))
    parser.add_argument("--passes", type=int, default=5,
                        help="epochs' worth of block draws to pool, so the "
                             "orange cloud samples Eq. (14) rather than one draw")
    parser.add_argument("--minibatch", type=int, default=64,
                        help="the training minibatch: Eqs. (12)-(13) estimate "
                             "the style spread across it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out",
                        default="reports/figures/generation_drift_projection.png")
    parser.add_argument("--json", default="reports/generation_drift_projection.json")
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

    # The artificial cloud, drawn the way training draws it: shuffled minibatches
    # of Batch 1, mixed gases, one block call per minibatch, several passes.
    generator = torch.Generator().manual_seed(args.seed)
    artificial_parts, artificial_origin = [], []
    with torch.no_grad():
        for _ in range(args.passes):
            order = torch.randperm(len(source), generator=generator)
            for start in range(0, len(order), args.minibatch):
                index = order[start:start + args.minibatch]
                if len(index) < 2:
                    continue
                artificial_parts.append(block(source[index], generator=generator))
                artificial_origin.append(index.numpy())
    artificial = torch.cat(artificial_parts).flatten(1).numpy()
    artificial_origin = np.concatenate(artificial_origin)
    artificial_y = source_y[artificial_origin]
    source = source.flatten(1).numpy()
    # Each artificial sample paired with the real sample it was restyled from,
    # so the block's own displacement can be read separately from Batch 1's
    # within-class spread.
    displacement = artificial - source[artificial_origin]

    targets = {b: load_target(b, log) for b in TARGET_BATCHES}
    target_features = {b: block3(x).flatten(1).numpy() for b, (x, _) in targets.items()}

    results: dict[str, dict[str, dict[str, float]]] = {}
    figure, axes = plt.subplots(1, len(ORDER), figsize=(3.05 * len(ORDER), 6.4),
                                sharey=True)
    figure.patch.set_facecolor(SURFACE)
    rows = list(TARGET_BATCHES)
    offsets = {"source": 0.24, "artificial": 0.0, "target": -0.24}
    x_min, x_max = 0.0, 1.0

    for column, name in enumerate(ORDER):
        label = next(k for k, v in GAS_LABELS.items() if v == name)
        axis = axes[column]
        axis.set_facecolor(SURFACE)
        for spine in axis.spines.values():
            spine.set_color(GRID)
        axis.tick_params(labelsize=8.5, colors=INK_MUTED)
        axis.set_title(name, fontsize=11.5, color=INK)
        axis.axvline(0.0, color=GRID, linewidth=1.0, zorder=0)
        axis.axvline(1.0, color=GRID, linewidth=1.0, zorder=0)

        src = source[source_y == label]
        art = artificial[artificial_y == label]
        mu_source = src.mean(axis=0)
        results[name] = {}

        for row, batch in enumerate(rows):
            y = len(rows) - 1 - row
            _, target_y = targets[batch]
            tgt = target_features[batch][target_y == label]
            if not len(tgt):
                axis.text(0.5, y, "n = 0", ha="center", va="center", fontsize=8.5,
                          color=INK_MUTED)
                continue
            mu_target = tgt.mean(axis=0)
            drift = mu_target - mu_source
            distance = float(np.linalg.norm(drift))
            u = drift / distance

            def coordinate(z: np.ndarray) -> np.ndarray:
                return ((z - mu_source) @ u) / distance

            s, a, t = coordinate(src), coordinate(art), coordinate(tgt)
            step = (displacement[artificial_y == label] @ u) / distance
            range_bar(axis, s, y + offsets["source"], SOURCE_C)
            range_bar(axis, a, y + offsets["artificial"], ARTIFICIAL_C)
            range_bar(axis, t, y + offsets["target"], TARGET_C)

            overlap = overlap_coefficient(a, t)
            reach = float(np.quantile(a, 0.95))
            beyond = float((a >= np.quantile(t, 0.05)).mean())
            results[name][f"batch{batch}"] = {
                "n_target": int(len(tgt)),
                "drift_distance": distance,
                "source_q": [float(v) for v in np.quantile(s, QUANTILES)],
                "artificial_q": [float(v) for v in np.quantile(a, QUANTILES)],
                "target_q": [float(v) for v in np.quantile(t, QUANTILES)],
                "artificial_mean": float(a.mean()),
                "block_step_along_u_mean": float(step.mean()),
                "block_step_along_u_sd": float(step.std()),
                "block_step_along_u_abs_mean": float(np.abs(step).mean()),
                "artificial_p95_reach": reach,
                "artificial_share_past_target_p05": beyond,
                "overlap_artificial_target": overlap,
            }
            axis.text(1.0, y, f"{overlap:.0%}", transform=axis.get_yaxis_transform(),
                      ha="right", va="center", fontsize=8, color=INK_MUTED,
                      clip_on=False)
            x_min = min(x_min, float(np.quantile(np.concatenate([s, a, t]), 0.01)))
            x_max = max(x_max, float(np.quantile(np.concatenate([s, a, t]), 0.99)))

    pad = 0.08 * (x_max - x_min)
    for axis in axes:
        axis.set_xlim(x_min - pad, x_max + pad)
        axis.set_ylim(-0.7, len(rows) - 0.3)
        axis.set_xlabel("position along drift axis u\n(0 = source centroid, "
                        "1 = target centroid)", fontsize=8.5, color=INK_MUTED)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"batch {b}" for b in reversed(rows)], fontsize=9.5,
                            color=INK)
    axes[0].tick_params(axis="y", length=0)

    handles = [Line2D([], [], color=c, linewidth=4.5, solid_capstyle="round",
                      marker="o", markersize=6, markeredgecolor=SURFACE)
               for c in (SOURCE_C, ARTIFICIAL_C, TARGET_C)]
    figure.legend(handles, ["Batch 1, real features",
                            "Artificial features generated from Batch 1",
                            "Target batch, real features"],
                  loc="upper center", ncol=3, frameon=False, fontsize=10,
                  bbox_to_anchor=(0.5, 0.955))
    figure.suptitle(
        "Along each (gas, batch) drift axis: the artificial cloud generated from "
        "Batch 1 against where the target really is\n"
        "bars: 5-95 % hairline, 25-75 % thick, median dot; right-edge figure: "
        "1-D histogram overlap between artificial and target",
        fontsize=12, color=INK, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.92))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160, facecolor=SURFACE)
    summary = {"checkpoint": args.checkpoint, "style": args.style,
               "passes": args.passes, "minibatch": args.minibatch,
               "seed": args.seed, "quantiles": list(QUANTILES),
               "block3_dims": int(source.shape[1]), "per_gas": results}
    (ROOT / args.json).write_text(json.dumps(summary, indent=2))
    print(out)
    print(ROOT / args.json)

    print(f"\n{'gas':<13}{'batch':>6}{'art p05':>9}{'art p50':>9}{'art p95':>9}"
          f"{'tgt p05':>9}{'tgt p50':>9}{'overlap':>9}{'step mean':>11}"
          f"{'step sd':>9}{'|step|':>9}")
    for name in ORDER:
        for key, r in results[name].items():
            a, t = r["artificial_q"], r["target_q"]
            print(f"{name:<13}{key[5:]:>6}{a[0]:>9.2f}{a[2]:>9.2f}{a[4]:>9.2f}"
                  f"{t[0]:>9.2f}{t[2]:>9.2f}{r['overlap_artificial_target']:>9.2f}"
                  f"{r['block_step_along_u_mean']:>+11.3f}"
                  f"{r['block_step_along_u_sd']:>9.3f}"
                  f"{r['block_step_along_u_abs_mean']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
