"""How much does the generation block actually move anything?

Source-only: Batch 1 and a frozen checkpoint. No target file is opened.
"""
import torch, numpy as np
from src import normalize
from src.data import load_source
from src.model import build, to_input
from src.generate import FeatureGeneration

CKPT = "runs/20260923T045843638020Z_displacement_full/checkpoints/R-aug-t2_lr0.0003_seed1042/final.pt"
payload = torch.load(CKPT, map_location="cpu", weights_only=False)
model = build("R-aug-t2"); model.load_state_dict(payload["state_dict"]); model.eval()

x, y = load_source()
z0 = to_input(normalize.apply(payload["normalizer"], x))

def rel(a, b):
    return float((a - b).norm() / a.norm())

torch.manual_seed(0)
print(f"{'style':>9} {'block out':>10} {'z_f':>8} {'softmax':>9} {'flips':>7} {'neg delta':>10}")
for style in ("scalar", "channel", "position"):
    block = FeatureGeneration(style=style)
    outs, zfs, sms, flips = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(z0), 64):          # the training minibatch
            chunk = z0[start:start + 64]
            if len(chunk) < 2:
                continue
            mid = model.stem(chunk)
            gen = block(mid)
            zf, zfg = model.trunk(mid), model.trunk(gen)
            p, pg = model.head(zf).softmax(1), model.head(zfg).softmax(1)
            outs.append(rel(mid, gen)); zfs.append(rel(zf, zfg))
            sms.append(rel(p, pg))
            flips.append(float((p.argmax(1) != pg.argmax(1)).float().mean()))
    print(f"{style:>9} {np.mean(outs):>10.4f} {np.mean(zfs):>8.4f} "
          f"{np.mean(sms):>9.4f} {np.mean(flips):>7.3f} "
          f"{block.negative_scale_fraction:>10.3f}")

# For scale: how far does the real drift move z_f? Measured inside Batch 1 only,
# using its own two acquisition blocks (docs/batch1-internal-drift.md). The
# SOURCE_BLOCKS ranges are absolute row indices, not per-class ones.
from src.augment import SOURCE_BLOCKS
with torch.no_grad():
    zf_all = model.features(z0)
scale = float(zf_all.norm(dim=(1, 2)).mean())
for label, ((a, b), (c, d)) in SOURCE_BLOCKS.items():
    offset = zf_all[c:d].mean(0) - zf_all[a:b].mean(0)
    print(f"Batch 1 block offset at z_f, label {label}: {float(offset.norm()) / scale:.4f}")
