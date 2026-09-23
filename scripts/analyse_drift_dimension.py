#!/usr/bin/env python
"""How many dimensions does the drift occupy, and how many can Batch 1 reach?

Builds the matrix of per-class per-batch centroid displacements, takes its
singular value spectrum, and asks how much of each component lies in the
subspace spanned by Batch 1's own acquisition-block offsets.

Post-hoc diagnosis. Every checkpoint was frozen and evaluated before this ran;
it selects nothing. The block offsets it compares against read Batch 1 only.
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

matplotlib.rcParams["font.family"] = ["Noto Sans CJK TC", "Noto Serif CJK TC",
                                      "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import augment, normalize  # noqa: E402
from src.data import GAS_LABELS, TARGET_BATCHES, load_source, load_target  # noqa: E402
from src.protocol import TargetAccessLog  # noqa: E402

COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, INK, INK_MUTED, ACCENT = "#fcfcfb", "#0b0b0b", "#52514e", "#b42318"


def collect():
    x, y = load_source()
    transform = lambda v: normalize.apply(
        normalize.fit("signed_log_then_per_sample", x), v)
    source = transform(x)
    centroid = {c: source[y == c].mean(axis=0) for c in range(1, 7)}

    log = TargetAccessLog()
    log.record_freeze(Path("post-hoc"), "diagnostic")
    rows, tags = [], []
    for batch in TARGET_BATCHES:
        xt, yt = load_target(batch, log)
        zt = transform(xt)
        for c in range(1, 7):
            mask = yt == c
            if mask.sum():
                rows.append(zt[mask].mean(axis=0) - centroid[c])
                tags.append((batch, c))
    return np.asarray(rows), tags, augment.block_offsets(source, y)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/figures/drift_dimension.png")
    parser.add_argument("--json", default="reports/drift_dimension.json")
    args = parser.parse_args()

    drift, tags, offsets = collect()
    # Singular spectrum of the raw displacements: no centring, because the
    # question is how many directions the displacements occupy, not how they
    # vary about their own mean.
    left, singular, right = np.linalg.svd(drift, full_matrices=False)
    energy = singular ** 2 / (singular ** 2).sum()
    cumulative = np.cumsum(energy)
    rank90 = int(np.searchsorted(cumulative, 0.90) + 1)

    # The source-only subspace: an orthonormal basis for the three block offsets.
    basis, _ = np.linalg.qr(
        np.stack([v / np.linalg.norm(v) for v in offsets.values()], axis=1))
    reachable = np.linalg.norm(basis.T @ right[:6].T, axis=0) ** 2

    per_class = {}
    for c in range(1, 7):
        rows = np.asarray([d for d, (_, k) in zip(drift, tags) if k == c])
        total = (np.linalg.norm(rows, axis=1) ** 2).sum()
        by_k = [float(((rows @ right[:k].T) ** 2).sum() / total) for k in range(1, 7)]
        inside = float((np.linalg.norm(rows @ basis, axis=1) ** 2).sum() / total)
        per_class[GAS_LABELS[c]] = {"cumulative_by_component": by_k,
                                    "inside_source_subspace": inside}

    report = {
        "displacements": len(drift),
        "singular_values": singular[:10].tolist(),
        "energy_fraction": energy[:10].tolist(),
        "cumulative": cumulative[:10].tolist(),
        "components_for_90_percent": rank90,
        "component_reachable_from_batch1": reachable.tolist(),
        "per_class": per_class,
    }
    (ROOT / args.json).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    figure.patch.set_facecolor(SURFACE)

    k = np.arange(1, 11)
    left_axis.bar(k, energy[:10] * 100, color="#2a78d6", width=0.62,
                  label="單一成分")
    left_axis.plot(k, cumulative[:10] * 100, color=ACCENT, marker="o",
                   markersize=5, linewidth=2, label="累積")
    left_axis.axhline(90, color=INK_MUTED, linestyle=":", linewidth=1.2)
    left_axis.annotate(f"90% 需要 {rank90} 個成分", xy=(rank90, 90),
                       xytext=(rank90 + 0.4, 74), fontsize=10, color=ACCENT)
    for position, value in zip(k[:4], energy[:4] * 100):
        left_axis.annotate(f"{value:.0f}%", xy=(position, value),
                           xytext=(position, value + 3), ha="center",
                           fontsize=9, color=INK_MUTED)
    left_axis.set_xticks(k)
    left_axis.set_xlabel("主成分", fontsize=10, color=INK_MUTED)
    left_axis.set_ylabel("漂移能量佔比 (%)", fontsize=10, color=INK_MUTED)
    left_axis.set_title(f"漂移的奇異值譜（{len(drift)} 個類別×batch 位移向量）",
                        fontsize=11.5, color=INK)
    left_axis.legend(fontsize=9, frameon=False)

    for index, c in enumerate(range(1, 7)):
        values = np.asarray(per_class[GAS_LABELS[c]]["cumulative_by_component"]) * 100
        right_axis.plot(np.arange(1, 7), values, marker="o", markersize=5,
                        linewidth=1.9, color=COLOURS[index], label=GAS_LABELS[c])
    right_axis.axvline(1, color=INK_MUTED, linestyle=":", linewidth=1.2)
    right_axis.annotate("v8.0／v8.1 只用 1 個方向", xy=(1, 30), xytext=(1.15, 24),
                        fontsize=9.5, color=INK_MUTED)
    right_axis.set_xticks(range(1, 7))
    right_axis.set_xlabel("使用前 k 個漂移主成分", fontsize=10, color=INK_MUTED)
    right_axis.set_ylabel("該類別漂移被涵蓋的比例 (%)", fontsize=10, color=INK_MUTED)
    right_axis.set_title("逐類別：用 k 個方向能抓到多少漂移", fontsize=11.5, color=INK)
    right_axis.legend(fontsize=9, frameon=False, ncol=2)
    right_axis.set_ylim(0, 102)

    for axis in (left_axis, right_axis):
        axis.set_facecolor(SURFACE)
        axis.grid(alpha=0.22, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color("#d9d8d4")

    figure.tight_layout()
    figure.savefig(ROOT / args.out, dpi=165, facecolor=SURFACE)

    print(f"{len(drift)} 個位移向量，90% 能量需要 {rank90} 個成分")
    print(f"{'成分':6s}{'能量':>9s}{'累積':>9s}{'Batch 1 可及':>14s}")
    for i in range(6):
        print(f"{i+1:<6d}{energy[i]:9.1%}{cumulative[i]:9.1%}{reachable[i]:14.1%}")
    print(f"\n{'gas':14s}" + "".join(f"{f'k={k}':>9s}" for k in range(1, 7))
          + f"{'子空間內':>11s}")
    for c in range(1, 7):
        e = per_class[GAS_LABELS[c]]
        print(f"{GAS_LABELS[c]:14s}"
              + "".join(f"{v:9.1%}" for v in e["cumulative_by_component"])
              + f"{e['inside_source_subspace']:11.1%}")
    print(f"\n{ROOT / args.out}\n{ROOT / args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
