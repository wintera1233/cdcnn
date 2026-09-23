#!/usr/bin/env python
"""PCA of Batch 1. Nothing else."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_source  # noqa: E402

x, y = load_source()
z = x - x.mean(0)
_, s, v = np.linalg.svd(z, full_matrices=False)
p = z @ v[:3].T
ev = s**2 / (s**2).sum()

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for i, (a, b) in enumerate([(0, 1), (0, 2)]):
    for c in range(1, 7):
        m = y == c
        ax[i].scatter(p[m, a], p[m, b], s=18, label=f"{c}", alpha=0.8)
    ax[i].set_xlabel(f"PC{a+1} ({ev[a]:.1%})")
    ax[i].set_ylabel(f"PC{b+1} ({ev[b]:.1%})")
ax[0].legend(title="label", markerscale=2)
fig.suptitle("Batch 1 PCA (raw features)")
fig.tight_layout()
out = ROOT / "reports/figures/batch1_pca.png"
fig.savefig(out, dpi=150)
print(out)
