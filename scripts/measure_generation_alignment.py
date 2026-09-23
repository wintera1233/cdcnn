"""Is the block's perturbation misaligned with the real drift, or just random?

Post-hoc: frozen checkpoints, target batches already opened and audited by the
v9.0 run. Measured at the block-3 output, which is where the block acts.

The null matters. At block 3 the feature is [256, 128] = 32768 dimensions, so a
random direction has cosine about 1/sqrt(32768) = 0.0055 with anything. A
near-zero cosine is therefore what an *isotropic* perturbation gives and is not
by itself evidence of a wrongly-chosen direction.
"""
import numpy as np, torch
from src import normalize
from src.data import TARGET_BATCHES, load_source, load_target
from src.evaluate import load_checkpoint
from src.generate import FeatureGeneration
from src.model import to_input
from src.protocol import TargetAccessLog

CK = "runs/20260923T071856507630Z_baseline_ladder_full/checkpoints/R-gen_lr0.0003_seed1042/final.pt"
model, payload = load_checkpoint(CK, "cpu"); norm = payload["normalizer"]
log = TargetAccessLog()
sx, sy = load_source()

@torch.no_grad()
def mid(x, bs=256):
    z = to_input(normalize.apply(norm, x))
    return torch.cat([model.stem(z[i:i+bs]) for i in range(0, len(z), bs)])

src_mid = mid(sx)
D = src_mid[0].numel()
print(f"block-3 feature: {tuple(src_mid.shape[1:])} = {D} dims")
print(f"null cosine for a random direction: {1/np.sqrt(D):.4f}\n")

# Real drift direction per class, source -> each target batch.
drifts = {}
for b in TARGET_BATCHES:
    tx, ty = load_target(b, log)
    t_mid = mid(tx)
    for c in np.unique(sy):
        if not (ty == c).any():
            continue
        d = (t_mid[ty == c].mean(0) - src_mid[sy == c].mean(0)).flatten()
        drifts.setdefault(int(c), []).append(d / d.norm())

# The block's own perturbation, on the same minibatching the training loop uses.
torch.manual_seed(0)
block = FeatureGeneration(style="position")
perturb, labels = [], []
with torch.no_grad():
    for i in range(0, len(src_mid), 64):
        chunk = src_mid[i:i+64]
        if len(chunk) < 2:
            continue
        p = (block(chunk) - chunk).flatten(1)
        perturb.append(p); labels.append(sy[i:i+len(chunk)])
perturb = torch.cat(perturb); labels = np.concatenate(labels)
perturb = perturb / perturb.norm(dim=1, keepdim=True)

print(f"{'class':>6}{'|cos| vs real drift':>22}{'signed mean':>14}{'vs null':>10}")
alls = []
for c, ds in sorted(drifts.items()):
    m = labels == c
    if not m.any():
        continue
    cos = perturb[m] @ torch.stack(ds).T           # [n_samples, n_batches]
    alls.append(float(cos.abs().mean()))
    print(f"{c:>6}{float(cos.abs().mean()):>22.4f}{float(cos.mean()):>14.4f}"
          f"{float(cos.abs().mean())/(1/np.sqrt(D)):>9.1f}x")

# How much of the perturbation lands in the span of all the drift directions?
basis = torch.stack([d for ds in drifts.values() for d in ds])
q, _ = torch.linalg.qr(basis.T)
share = (perturb @ q).pow(2).sum(1)
print(f"\nperturbation energy inside the {q.shape[1]}-dim span of every real "
      f"drift direction: {float(share.mean()):.1%}")
print(f"null for a random direction in {D} dims: {q.shape[1]/D:.1%}")
