#!/usr/bin/env python
"""v12 mechanism check: how much real drift survives each projection, at the input.

Post-hoc. The target batches were opened and audited by the v12.0 run; this
reads them again through the same Normal kinds the four cells trained with,
fitted on Batch 1 in file order exactly as training fitted them.

For each Normal kind and each (gas, target batch) with that gas present:

    drift / radius = ||centroid_target - centroid_source|| / mean ||x - centroid_source||

over the Batch 1 members of the class, in the normalised (and projected) input
space. The baseline's median is 1.80 (docs/drift-geometry.md). Also reported:
which Batch 1 class centroid each target Acetaldehyde centroid lands nearest to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import normalize  # noqa: E402
from src.data import GAS_LABELS, TARGET_BATCHES, load_source, load_target  # noqa: E402
from src.protocol import TargetAccessLog  # noqa: E402

KINDS = {"R-fig-logps": "signed_log_then_per_sample",
         "R-proj-axis": "logps_proj_offset_axis",
         "R-proj-sub3": "logps_proj_sub3",
         "R-proj-eth": "logps_proj_eth"}


def main() -> int:
    log = TargetAccessLog()
    log.record_freeze(Path("post-hoc"), "diagnostic")
    sx, sy = load_source()
    targets = {b: load_target(b, log) for b in TARGET_BATCHES}

    print(f"{'cell':<13}" + "".join(f"{GAS_LABELS[c]:>13}" for c in sorted(GAS_LABELS))
          + f"{'median':>9}")
    nearest = {}
    for cell, kind in KINDS.items():
        params = normalize.fit(kind, sx)
        zs = normalize.apply(params, sx)
        centroids = {c: zs[sy == c].mean(axis=0) for c in sorted(GAS_LABELS)}
        radius = {c: float(np.linalg.norm(zs[sy == c] - centroids[c], axis=1).mean())
                  for c in centroids}
        ratios = {c: [] for c in centroids}
        nearest[cell] = []
        for b, (tx, ty) in targets.items():
            zt = normalize.apply(params, tx)
            for c in centroids:
                if not (ty == c).any():
                    continue
                ct = zt[ty == c].mean(axis=0)
                ratios[c].append(float(np.linalg.norm(ct - centroids[c])) / radius[c])
                if c == 4:
                    d = {k: float(np.linalg.norm(ct - centroids[k])) for k in centroids}
                    nearest[cell].append(GAS_LABELS[min(d, key=d.get)])
        per_class = [float(np.mean(ratios[c])) for c in sorted(centroids)]
        all_cells = [r for c in ratios for r in ratios[c]]
        print(f"{cell:<13}" + "".join(f"{v:>13.3f}" for v in per_class)
              + f"{np.median(all_cells):>9.3f}")

    print("\nwhere each target batch's Acetaldehyde centroid lands (nearest Batch 1 centroid)")
    for cell, hits in nearest.items():
        print(f"{cell:<13}" + " ".join(f"{h[:4]:>5}" for h in hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
