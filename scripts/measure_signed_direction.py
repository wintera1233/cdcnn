#!/usr/bin/env python
"""Did v10.0's sign point toward the target, and how far did the shift reach?

Post-hoc, on a frozen v10.0 checkpoint; the target batches were opened and
audited by that run. In the block's own style space (per-position mean and std
of the block-3 residual), three things:

  1. the cosine between Batch 1's Ethanol block offset - the direction the two
     signed cells used - and each class's real source-to-target drift;
  2. the same for the mean of the three block offsets, the source-only choice a
     practitioner without target data would make;
  3. the shift's magnitude, T x ||offset||, against the real drift's magnitude.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import normalize  # noqa: E402
from src.augment import SOURCE_BLOCKS  # noqa: E402
from src.data import GAS_LABELS, TARGET_BATCHES, load_source, load_target  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.generate import FeatureGeneration  # noqa: E402
from src.model import to_input  # noqa: E402
from src.protocol import TargetAccessLog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=(
        "runs/20260929T031927475103Z_baseline_ladder_full/checkpoints/"
        "R-gen-shift_lr0.0003_seed1042/final.pt"))
    parser.add_argument("--displacement", type=float, default=2.0)
    args = parser.parse_args()

    model, payload = load_checkpoint(Path(args.checkpoint), "cpu")
    block = FeatureGeneration(style="position")
    log = TargetAccessLog()
    log.record_freeze(Path("post-hoc"), "diagnostic")

    @torch.no_grad()
    def style(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inp = to_input(normalize.apply(payload["normalizer"], rows))
        mid = torch.cat([model.stem(inp[i:i + 256]) for i in range(0, len(inp), 256)])
        mean, std = block.style_moments(mid)
        return mean.flatten(1).numpy(), std.flatten(1).numpy()

    sx, sy = load_source()
    s_mean, s_std = style(sx)

    offsets = {}
    for label, ((a, b), (c, d)) in SOURCE_BLOCKS.items():
        offsets[label] = (s_mean[c:d].mean(0) - s_mean[a:b].mean(0),
                          s_std[c:d].mean(0) - s_std[a:b].mean(0))
    unit = lambda v: v / np.linalg.norm(v)
    directions = {
        "ethanol (used)": offsets[1],
        "average of three": (unit(np.mean([unit(o[0]) for o in offsets.values()], 0)),
                             unit(np.mean([unit(o[1]) for o in offsets.values()], 0))),
    }

    print(f"style space: {s_mean.shape[1]} coordinates each for mean and std")
    print(f"Ethanol block offset norm: mean {np.linalg.norm(offsets[1][0]):.4f}, "
          f"std {np.linalg.norm(offsets[1][1]):.4f}; "
          f"shift at T={args.displacement}: {args.displacement * np.linalg.norm(offsets[1][0]):.4f}\n")

    rows = []
    for b in TARGET_BATCHES:
        tx, ty = load_target(b, log)
        t_mean, t_std = style(tx)
        for c in sorted(GAS_LABELS):
            if not (ty == c).any():
                continue
            drift_mean = t_mean[ty == c].mean(0) - s_mean[sy == c].mean(0)
            drift_std = t_std[ty == c].mean(0) - s_std[sy == c].mean(0)
            rows.append((c, b, drift_mean, drift_std))

    for name, (d_mean, d_std) in directions.items():
        print(f"direction: {name}")
        print(f"{'class':<14}{'n':>3}{'cos mean-axis':>15}{'cos std-axis':>14}"
              f"{'|drift| mean':>14}{'shift/drift':>13}")
        all_cos = []
        for c in sorted(GAS_LABELS):
            mine = [r for r in rows if r[0] == c]
            cm = [float(unit(r[2]) @ unit(d_mean)) for r in mine]
            cs = [float(unit(r[3]) @ unit(d_std)) for r in mine]
            mag = float(np.mean([np.linalg.norm(r[2]) for r in mine]))
            reach = args.displacement * np.linalg.norm(offsets[1][0]) / mag
            all_cos += cm
            print(f"{GAS_LABELS[c]:<14}{len(mine):>3}{np.mean(cm):>+15.3f}"
                  f"{np.mean(cs):>+14.3f}{mag:>14.4f}{reach:>13.2f}")
        print(f"{'all 51':<14}{len(all_cos):>3}{np.mean(all_cos):>+15.3f}   "
              f"(random-direction null {1 / np.sqrt(s_mean.shape[1]):.3f})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
