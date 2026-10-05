#!/usr/bin/env python
"""Batch 1 in a 3-D principal-component space, before and after each augmentation.

Drawn in the same form as the paper's Fig. 4, which plots the raw data and three
augmentations in PC1-PC3. All panels share one projection, fitted on the
unaugmented cloud, and one set of limits, so the inflation is comparable
across them.

Reads Batch 1 for everything it trains on. The one target quantity is the
centroid of target Acetaldehyde, drawn as a cross: it answers whether an
augmented cloud reaches where that class actually goes. It is a diagnostic,
computed after every checkpoint was frozen, and is marked as such.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams["font.family"] = ["Noto Sans CJK TC", "Noto Serif CJK TC",
                                      "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import augment, normalize  # noqa: E402
from src.data import GAS_LABELS, TARGET_BATCHES, load_source, load_target  # noqa: E402
from src.model import AUGMENT_DISPLACEMENT, AUGMENT_LAMBDA, VARIANTS  # noqa: E402
from src.protocol import TargetAccessLog  # noqa: E402

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK_MUTED, CRITICAL = "#fcfcfb", "#0b0b0b", "#52514e", "#b42318"

PANELS = [("R-fig-logps", "原始（無擴充）"),
          ("R-aug-paper", "論文 Eq. (5)–(7)：等向"),
          ("R-aug-ethd", "有方向（Ethanol）：無等向"),
          ("R-aug-eth", "有方向（Ethanol）＋ 等向")]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--no-target-marker", action="store_true")
    parser.add_argument("--out", default="reports/figures/augmentation_pca.png")
    parser.add_argument("--panels", nargs="+", default=None,
                        help="variant names to draw, in order; default all four")
    parser.add_argument("--font-scale", type=float, default=1.0)
    parser.add_argument("--marker-scale", type=float, default=1.0)
    parser.add_argument("--no-suptitle", action="store_true")
    parser.add_argument("--lang", choices=("zh", "en"), default="zh")
    parser.add_argument("--two-d", action="store_true",
                        help="PC1 against PC2 instead of the three-component box")
    parser.add_argument("--clip", type=float, nargs=2, default=(1.0, 99.0),
                        help="percentiles of the drawn clouds that set the axis range")
    args = parser.parse_args()

    x, y = load_source()
    transform = lambda v: normalize.apply(
        normalize.fit("signed_log_then_per_sample", x), v)
    base = transform(x)
    centre = base.mean(axis=0)
    _, singular, components = np.linalg.svd(base - centre, full_matrices=False)
    explained = singular ** 2 / (singular ** 2).sum()
    project = lambda v: (v - centre) @ components[:3].T

    target_full = target_point = None
    if not args.no_target_marker:
        log = TargetAccessLog()
        log.record_freeze(Path("post-hoc"), "diagnostic")
        rows = [transform(xt)[yt == 4] for b in TARGET_BATCHES
                for xt, yt in [load_target(b, log)] if (yt == 4).sum()]
        target_full = np.concatenate(rows).mean(axis=0)
        target_point = project(target_full[None, :])[0]
    # The class radius of Acetaldehyde in the unaugmented source cloud, in the
    # full space: the yardstick the coverage check uses.
    acet = base[y == 4]
    acet_radius = float(np.linalg.norm(acet - acet.mean(axis=0), axis=1).mean())

    panels = PANELS if args.panels is None else [
        next(p for p in PANELS if p[0] == name) for name in args.panels]
    if args.lang == "en":
        english = {"R-fig-logps": "No augmentation",
                   "R-aug-paper": "Paper: isotropic noise",
                   "R-aug-ethd": "Directed, no noise",
                   "R-aug-eth": "Directed + noise"}
        panels = [(v, english[v]) for v, _ in panels]
    clouds = {}
    for variant, _ in panels:
        spec = VARIANTS[variant].get("augment")
        if spec is None:
            clouds[variant] = (base, y)
            continue
        extra = augment.augment(base, y, np.random.default_rng(args.seed),
                                isotropic=spec["isotropic"],
                                direction=spec["direction"], lam=AUGMENT_LAMBDA,
                                displacement=AUGMENT_DISPLACEMENT)
        clouds[variant] = (np.concatenate([base, extra]), np.concatenate([y, y]))

    every = np.concatenate([project(v) for v, _ in clouds.values()])
    if target_point is not None:
        every = np.concatenate([every, target_point[None, :]])
    # Percentile limits, not min/max: a handful of far outliers would otherwise
    # squeeze every cloud into the middle of the box.
    lim = [(np.percentile(every[:, k], args.clip[0]), np.percentile(every[:, k], args.clip[1]))
           for k in range(3)]
    pad = [(b - a) * 0.08 for a, b in lim]
    lim = [(a - p, b + p) for (a, b), p in zip(lim, pad)]

    fs = args.font_scale
    figure = plt.figure(figsize=(4.25 * len(panels), 4.9 if args.two_d else 5.6),
                        facecolor=SURFACE)
    radius0 = None
    marker_label = ("target Acetaldehyde centroid" if args.lang == "en"
                    else "target Acetaldehyde 重心（診斷）")
    for index, (variant, title) in enumerate(panels, start=1):
        z, labels = clouds[variant]
        points = project(z)
        if args.two_d:
            axis = figure.add_subplot(1, len(panels), index)
            axis.set_facecolor(SURFACE)
            for spine in axis.spines.values():
                spine.set_color("#d9d8d4")
            for c in range(1, 7):
                m = labels == c
                axis.scatter(points[m, 0], points[m, 1], s=6 * args.marker_scale,
                             color=COLOURS[c - 1], alpha=0.45, linewidths=0,
                             label=GAS_LABELS[c])
            if target_point is not None:
                axis.scatter(target_point[0], target_point[1], marker="X", s=180,
                             color=CRITICAL, edgecolors="white", linewidths=1.4,
                             zorder=10, label=marker_label)
                # The Acetaldehyde centroid of this panel's cloud (source or
                # augmented), its line to the target centroid, and the distance
                # between the two measured in the full 128-dimensional space.
                mine_full = z[labels == 4]
                centroid_full = mine_full.mean(axis=0)
                centroid_2d = project(centroid_full[None, :])[0][:2]
                gap = float(np.linalg.norm(centroid_full - target_full))
                axis.plot([centroid_2d[0], target_point[0]], [centroid_2d[1], target_point[1]],
                          color=CRITICAL, linewidth=1.6, linestyle="--", zorder=9)
                axis.scatter(centroid_2d[0], centroid_2d[1], marker="D", s=90,
                             facecolor="white", edgecolors=CRITICAL, linewidths=1.8,
                             zorder=10, label=("Acetaldehyde centroid of this cloud"
                                               if index == 1 else None))
                mid = 0.5 * (centroid_2d + target_point[:2])
                axis.annotate(f"d = {gap:.2f}", xy=mid, xytext=(0, 13), textcoords="offset points",
                              ha="center", fontsize=10 * fs, color=CRITICAL, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.2", facecolor=SURFACE,
                                        edgecolor="none", alpha=0.9))
            axis.set_xlim(*lim[0]); axis.set_ylim(*lim[1])
            axis.set_aspect("equal")
            axis.set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=9 * fs, color=INK_MUTED)
            if index == 1:
                axis.set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=9 * fs, color=INK_MUTED)
            axis.tick_params(labelsize=8 * fs, colors=INK_MUTED)
            axis.grid(True, color="#e6e5e1", linewidth=0.6)
        else:
            axis = figure.add_subplot(1, len(panels), index, projection="3d")
            axis.set_facecolor(SURFACE)
            for c in range(1, 7):
                m = labels == c
                axis.scatter(points[m, 0], points[m, 1], points[m, 2], s=3.5 * args.marker_scale,
                             color=COLOURS[c - 1], alpha=0.35, linewidths=0,
                             label=GAS_LABELS[c], depthshade=False)
            if target_point is not None:
                axis.scatter(*target_point, marker="X", s=150, color=CRITICAL,
                             edgecolors="white", linewidths=1.4, depthshade=False,
                             zorder=10, label=marker_label)
        radius = np.mean([np.linalg.norm(points[labels == c]
                                         - points[labels == c].mean(0), axis=1).mean()
                          for c in range(1, 7)])
        radius0 = radius if radius0 is None else radius0
        # Coverage is judged in the full 128-dim space on this class's own
        # samples; the three-component projection discards most of the drift.
        if target_full is None:
            reach = ""
        else:
            mine = z[labels == 4]
            nearest = float(np.linalg.norm(mine - target_full, axis=1).min())
            if args.lang == "en":
                gap = float(np.linalg.norm(z[labels == 4].mean(axis=0) - target_full))
                reach = (f"\nAcetaldehyde centroid to target: {gap:.2f}"
                         f"  (class radius {acet_radius:.2f})")
            else:
                reach = (f"\n到 target 的最近距離 {nearest:.2f}"
                         f"（半徑 {acet_radius:.2f}，{'涵蓋' if nearest < acet_radius else '未涵蓋'}）")
        if args.two_d:
            subtitle = title
        elif args.lang == "en":
            subtitle = f"{title}\nwithin-class radius {radius:.2f} ({radius / radius0:.2f}x){reach}"
        else:
            subtitle = f"{title}\n類內半徑 {radius:.2f}（{radius / radius0:.2f}×）{reach}"
        axis.set_title(subtitle, fontsize=(12 if args.two_d else 10) * fs, color=INK, pad=6)
        if args.two_d:
            continue
        axis.set_xlim(*lim[0]); axis.set_ylim(*lim[1]); axis.set_zlim(*lim[2])
        axis.set_xlabel(f"PC1 ({explained[0]:.1%})", fontsize=8.5 * fs, color=INK_MUTED)
        axis.set_ylabel(f"PC2 ({explained[1]:.1%})", fontsize=8.5 * fs, color=INK_MUTED)
        axis.set_zlabel(f"PC3 ({explained[2]:.1%})", fontsize=8.5 * fs, color=INK_MUTED)
        axis.tick_params(labelsize=7.5 * fs, colors=INK_MUTED)
        axis.view_init(elev=18, azim=-58)
        axis.set_box_aspect((1.0, 1.0, 0.72), zoom=1.22)
        for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
            pane.pane.set_facecolor(SURFACE)
            pane.pane.set_edgecolor("#d9d8d4")

    handles, names = figure.axes[0].get_legend_handles_labels()
    if args.two_d:
        figure.legend(handles, names, loc="lower center", ncol=len(names), frameon=False,
                      fontsize=9 * fs, markerscale=1.3, bbox_to_anchor=(0.5, 0.0),
                      handletextpad=0.3, columnspacing=1.2)
    else:
        figure.legend(handles, names, loc="lower center", ncol=7 if len(panels) == 4 else 4,
                      frameon=False, fontsize=10 * fs, markerscale=2.6,
                      bbox_to_anchor=(0.5, -0.01))
    if not args.no_suptitle:
        figure.suptitle("Batch 1 擴充前後的三維主成分空間（對照論文 Fig. 4）\n"
                        "所有面板共用同一組投影與座標範圍；範圍取全體的 1–99 百分位",
                        fontsize=13 * fs, color=INK, y=0.99)
    if args.two_d:
        figure.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.2, wspace=0.12)
    else:
        figure.subplots_adjust(left=0.01, right=0.99,
                               top=0.80 if not args.no_suptitle else 0.88,
                               bottom=0.10 if len(panels) == 4 else 0.14, wspace=0.02)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=155, facecolor=SURFACE)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
