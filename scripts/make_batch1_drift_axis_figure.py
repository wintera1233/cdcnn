#!/usr/bin/env python
"""Where the drift axis sits in Batch 1, and what removing it does. Batch 1 only.

Three panels in the signed-log -> per-sample space:

  left    Batch 1 in its own PCA plane (PC1, PC2), coloured by gas. The three
          classes with two acquisition sessions show the first session filled
          and the second hollow; an arrow joins their centroids. The common
          drift axis (leading direction of the three unit offsets) is drawn
          through the grand mean.
  middle  each class's coordinate along that drift axis, first and second
          session separately. The sessions separate along it; the gases do not.
  right   the same points after x - u u^T x, drawn in the same PCA basis. The
          second-session points land on the first-session points.

No target file is opened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import normalize  # noqa: E402
from src.augment import SOURCE_BLOCKS, block_offsets  # noqa: E402
from src.data import GAS_LABELS, load_source  # noqa: E402

ORDER = [1, 2, 3, 4, 5, 6]
COLOURS = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7", 5: "#D55E00", 6: "#56B4E9"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"


def sessions(y: np.ndarray) -> np.ndarray:
    """0 = first acquisition session (or the only one), 1 = second."""
    second = np.zeros(len(y), dtype=bool)
    for (_, _), (c, d) in SOURCE_BLOCKS.values():
        second[c:d] = True
    return second.astype(int)


def style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.tick_params(labelsize=8.5, colors=MUTED)


def scatter_plane(axis, coords, y, second, *, title):
    for label in ORDER:
        for sess, filled in ((0, True), (1, False)):
            m = (y == label) & (second == sess)
            if not m.any():
                continue
            axis.scatter(coords[m, 0], coords[m, 1], s=16 if filled else 30,
                         facecolor=COLOURS[label] if filled else "none",
                         edgecolor=COLOURS[label], linewidths=0 if filled else 1.2,
                         alpha=0.8, zorder=3)
    axis.set_title(title, fontsize=11, color=INK)


def main() -> int:
    x, y = load_source()
    z = normalize.apply(normalize.fit("signed_log_then_per_sample", x), x)
    second = sessions(y)

    centre = z.mean(axis=0)
    _, s, vt = np.linalg.svd(z - centre, full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    basis = vt[:2].T

    offsets = block_offsets(z, y)
    unit = np.stack([v / np.linalg.norm(v) for _, v in sorted(offsets.items())])
    _, _, vh = np.linalg.svd(unit, full_matrices=False)
    u = vh[0]
    # Orient u so the second sessions sit on its positive side.
    if np.mean([offsets[k] @ u for k in offsets]) < 0:
        u = -u

    projected = z - np.outer(z @ u, u)
    plane = (z - centre) @ basis
    plane_after = (projected - centre) @ basis
    along = (z - centre) @ u

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                                gridspec_kw={"width_ratios": [1.25, 0.8, 1.25]})
    figure.patch.set_facecolor(SURFACE)
    for axis in axes:
        style_axis(axis)

    # left: the plane, the session offsets, the common axis
    ax = axes[0]
    scatter_plane(ax, plane, y, second,
                  title=f"Batch 1 自己的 PCA 平面（PC1 {share[0]:.0%}，PC2 {share[1]:.0%}）")
    for label, off in offsets.items():
        a = (z[y == label][second[y == label] == 0].mean(axis=0) - centre) @ basis
        b = (z[y == label][second[y == label] == 1].mean(axis=0) - centre) @ basis
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>", color=COLOURS[label], lw=1.8))
    span = 1.1 * np.abs(plane).max(axis=0)
    direction = u @ basis
    direction = direction / np.linalg.norm(direction)
    reach = 0.9 * min(span[1] / abs(direction[1]), span[0] / max(abs(direction[0]), 1e-6))
    ax.plot([-reach * direction[0], reach * direction[0]],
            [-reach * direction[1], reach * direction[1]],
            color=INK, lw=1.4, ls="--", zorder=2)
    ax.text(reach * direction[0], reach * direction[1], "  漂移軸 u", fontsize=9.5,
            color=INK, va="center")
    ax.set_xlabel("PC1（裝 91% 的類間變異）", fontsize=9.5, color=MUTED)
    ax.set_ylabel("PC2（與 u 的 |cos| 0.91）", fontsize=9.5, color=MUTED)
    ax.set_xlim(-span[0], span[0]); ax.set_ylim(-span[1], span[1])

    # middle: coordinate along u, per class, by session
    ax = axes[1]
    rng = np.random.default_rng(0)
    for i, label in enumerate(ORDER):
        yy = len(ORDER) - 1 - i
        for sess, filled in ((0, True), (1, False)):
            m = (y == label) & (second == sess)
            if not m.any():
                continue
            jitter = rng.uniform(-0.18, 0.18, m.sum()) + (0.0 if sess == 0 else 0.0)
            ax.scatter(along[m], yy + jitter, s=14 if filled else 26,
                       facecolor=COLOURS[label] if filled else "none",
                       edgecolor=COLOURS[label], linewidths=0 if filled else 1.1,
                       alpha=0.75, zorder=3)
            ax.plot([along[m].mean()] * 2, [yy - 0.3, yy + 0.3], color=COLOURS[label],
                    lw=2.2 if filled else 1.2, ls="-" if filled else "--", zorder=4)
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([GAS_LABELS[l] for l in reversed(ORDER)], fontsize=9.5, color=INK)
    ax.set_xlabel("沿漂移軸 u 的座標", fontsize=9.5, color=MUTED)
    ax.set_title("每種氣體沿 u 的分布\n實心 = 第一次採集，空心 = 第二次", fontsize=11, color=INK)

    # right: after projection, same basis
    ax = axes[2]
    scatter_plane(ax, plane_after, y, second,
                  title="拿掉 u 之後（x − u uᵀ x），同一組 PCA 軸")
    ax.set_xlabel("PC1", fontsize=9.5, color=MUTED)
    ax.set_ylabel("PC2", fontsize=9.5, color=MUTED)
    ax.set_xlim(-span[0], span[0]); ax.set_ylim(-span[1], span[1])

    handles = [Line2D([], [], ls="none", marker="o", markersize=7,
                      markerfacecolor=COLOURS[l], markeredgecolor=COLOURS[l])
               for l in ORDER]
    handles += [Line2D([], [], ls="none", marker="o", markersize=7, markerfacecolor=INK,
                       markeredgecolor=INK),
                Line2D([], [], ls="none", marker="o", markersize=8, markerfacecolor="none",
                       markeredgecolor=INK, markeredgewidth=1.3)]
    names = [GAS_LABELS[l] for l in ORDER] + ["第一次採集", "第二次採集（只有三類有）"]
    figure.legend(handles, names, loc="upper center", ncol=8, frameon=False,
                  fontsize=9.5, bbox_to_anchor=(0.5, 0.985))
    figure.suptitle("漂移軸在 Batch 1 裡的位置：氣體沿 PC1 分開，採集時間沿 PC2 分開，兩者幾乎正交",
                    fontsize=12.5, color=INK, y=1.04)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "reports" / "figures" / "batch1_drift_axis.png"
    figure.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
