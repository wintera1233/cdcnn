"""Block perturbation vs real drift, both at block 3, both in source radii."""
import numpy as np, torch
from src import normalize
from src.data import TARGET_BATCHES, load_source, load_target
from src.evaluate import load_checkpoint
from src.generate import FeatureGeneration
from src.model import to_input
from src.protocol import TargetAccessLog

CK = "runs/20260923T071856507630Z_baseline_ladder_full/checkpoints/R-gen_lr0.0003_seed1042/final.pt"
model, payload = load_checkpoint(CK, "cpu"); norm = payload["normalizer"]
log = TargetAccessLog(); sx, sy = load_source()

@torch.no_grad()
def mid(x, bs=256):
    z = to_input(normalize.apply(norm, x))
    return torch.cat([model.stem(z[i:i+bs]) for i in range(0, len(z), bs)])

src = mid(sx).flatten(1)
radius = {int(c): float((src[sy == c] - src[sy == c].mean(0)).norm(dim=1).mean())
          for c in np.unique(sy)}

torch.manual_seed(0)
block = FeatureGeneration(style="position")
pert = {}
with torch.no_grad():
    for i in range(0, len(src), 64):
        chunk = mid(sx[i:i+64])
        if len(chunk) < 2: continue
        p = (block(chunk) - chunk).flatten(1).norm(dim=1)
        for c, v in zip(sy[i:i+len(chunk)], p):
            pert.setdefault(int(c), []).append(float(v))

drift = {}
for b in TARGET_BATCHES:
    tx, ty = load_target(b, log); tm = mid(tx).flatten(1)
    for c in np.unique(sy):
        if (ty == c).any():
            drift.setdefault(int(c), []).append(
                float((tm[ty == c].mean(0) - src[sy == c].mean(0)).norm()))

print(f"{'class':>6}{'block step':>13}{'real drift':>13}{'ratio':>9}")
bs, ds = [], []
for c in sorted(radius):
    b_ = np.mean(pert[c]) / radius[c]; d_ = np.mean(drift[c]) / radius[c]
    bs.append(b_); ds.append(d_)
    print(f"{c:>6}{b_:>13.3f}{d_:>13.3f}{d_/b_:>8.1f}x")
print(f"{'mean':>6}{np.mean(bs):>13.3f}{np.mean(ds):>13.3f}{np.mean(ds)/np.mean(bs):>8.1f}x")
