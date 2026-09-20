# Input normalization ladder (v6.4)

## Why

Two observations motivate this experiment.

1. The paper's plain ResNet baseline scores 0.6344 target mean; this project's B0
   scores 0.4077, and its best stage 0.4922. The gap predates every CDCNN
   component, so it lives in the backbone's inputs or training, not in
   augmentation, feature generation, or contrastive learning. See
   `docs/paper-vs-implementation.md`.
2. The A3 confound ablation showed that swapping `BatchNorm1d(128)` for
   `LayerNorm(128)` — a per-sample normalization — is worth +0.0845 target mean
   on 5/5 seeds, while the contrastive loss is worth +0.0029. Normalization, not
   the CDCNN modules, is what moves this dataset.

Under the Batch-1-fitted `StandardScaler`, drifted target rows reach extreme
values: Batch 2 contains |z| near 1.5e4 and a mean row variance of 1448 against
0.95 for Batch 1. Batches 4, 9, and 10 also carry heavy tails. This ladder tests
whether handling those inputs differently recovers accuracy.

## Stages

All six share the canonical backbone, optimizer, scheduler, 100 epochs, batch
size, seeds, and folds. No augmentation, feature generation, or contrastive loss
is involved, so the only differences are the FC128 normalization layer and the
input transform.

| Stage | FC128 normalization | Input transform after the scaler |
|---|---|---|
| `B0` | BatchNorm | none (reference) |
| `B0-LN` | LayerNorm | none (reference) |
| `B0-PS` | BatchNorm | per-sample standardization |
| `B0-LN-PS` | LayerNorm | per-sample standardization |
| `B0-LN-LOG` | LayerNorm | `sign(z) * log1p(|z|)` |
| `B0-LN-CLIP` | LayerNorm | clip to `[-5, 5]` |

Reading the comparisons:

- `B0` vs `B0-PS`: does per-sample input normalization recover the LayerNorm gain
  without changing the network?
- `B0-LN` vs `B0-LN-PS`: do the two per-sample normalizations compose, or is the
  effect already saturated?
- `B0-LN-LOG` and `B0-LN-CLIP`: is it the heavy tail specifically, rather than
  per-sample scale, that costs accuracy?

## Leakage properties

Every transform is parameter-free: `per_sample` uses only the row it transforms,
`signed_log` is a fixed function, and `clip` uses a limit fixed at 5.0 before
training. Nothing is fitted, so no statistic can cross between samples, folds, or
batches. The `StandardScaler` that runs first is still fitted on Batch 1
training rows only, and the transform kind is recorded in each checkpoint so the
post-freeze target path applies exactly what training applied.

## Running it

```bash
python scripts/run_cdcnn_v6_full.py gpu-smoke --config configs/cdcnn_v6_4_input_norm.json
python scripts/run_cdcnn_v6_full.py launch --config configs/cdcnn_v6_4_input_norm.json \
  --max-workers 1 --max-attempts 2 --gpu-smoke-run runs/<passing_gpu_smoke>
```

Implementation version `CDCNN_v6.4_input_normalization`; transforms in
`src/input_transform.py`; 30 source checkpoints (six stages x five seeds).
