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

## Results (2026-09-20)

Source run: `20260920T172721243924Z_cdcnn_v6_4_input_norm_full` (30 checkpoints).
Target evaluation: `20260920T181354711407Z_cdcnn_v6_4_input_norm_eval`, audit `passed`.

| Stage | Batch 1 CV | Target mean | SD | Pooled |
|---|---:|---:|---:|---:|
| `B0` | 0.9784 | 0.4077 | 0.0186 | 0.3943 |
| `B0-LN` | 0.9910 | 0.4922 | 0.0415 | 0.5077 |
| `B0-PS` | 0.9856 | 0.5164 | 0.0458 | 0.5179 |
| `B0-LN-PS` | 0.9933 | 0.5569 | 0.0342 | 0.5507 |
| `B0-LN-LOG` | 0.9874 | 0.4935 | 0.0287 | 0.4905 |
| `B0-LN-CLIP` | 0.9910 | 0.4849 | 0.0429 | 0.4991 |

### Paired steps

| Step | Isolates | Mean change | SD | Seeds improved |
|---|---|---:|---:|---:|
| `B0` -> `B0-LN` | LayerNorm alone | +0.0845 | 0.0339 | 5/5 |
| `B0` -> `B0-PS` | per-sample input alone | +0.1087 | 0.0526 | 5/5 |
| `B0-LN` -> `B0-LN-PS` | adding per-sample input to LayerNorm | +0.0647 | 0.0417 | 5/5 |
| `B0-PS` -> `B0-LN-PS` | adding LayerNorm to per-sample input | +0.0405 | 0.0466 | 4/5 |
| `B0-LN` -> `B0-LN-LOG` | signed-log instead of no transform | +0.0013 | 0.0289 | 2/5 |
| `B0-LN` -> `B0-LN-CLIP` | clipping to ±5 instead of no transform | -0.0073 | 0.0048 | 0/5 |
| `B0` -> `B0-LN-PS` | both per-sample normalizations | +0.1492 | 0.0433 | 5/5 |

### Findings

1. **Per-sample input normalization is the single strongest change measured so
   far: +0.1087 on 5/5 seeds**, larger than the LayerNorm swap
   (+0.0845) that the confound ablation identified, and far larger than
   anything contributed by augmentation, feature generation, or contrastive
   learning.
2. **The two normalizations compose.** Applying both reaches
   0.5569 target mean, +0.1492 over B0 on 5/5 seeds. Neither
   subsumes the other: adding per-sample input to LayerNorm is worth
   +0.0647, and adding LayerNorm to per-sample input is worth
   +0.0405.
3. **`B0-LN-PS` also has the best Batch 1 CV of any stage trained in this
   project (0.9933)**, so the source-only protocol selects it without
   consulting a single target metric.
4. **Tail compression alone is not the mechanism.** Signed-log is worth
   +0.0013 and clipping to ±5 is worth -0.0073; neither helps.
   What matters is removing each sample's own offset and gain, not bounding
   extreme values. This also rules out the extreme magnitudes (|z| ~ 1.5e4) as
   the primary cause: clipping removes them and changes nothing.

### Per-batch accuracy

| Batch | B0 | B0-LN | B0-PS | B0-LN-PS | B0-LN-LOG | B0-LN-CLIP | paper CDCNN | `B0-LN-PS` - paper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.738 | 0.873 | 0.860 | 0.889 | 0.889 | 0.878 | 0.827 | +0.061 |
| 3 | 0.507 | 0.692 | 0.646 | 0.773 | 0.707 | 0.690 | 0.768 | +0.004 |
| 4 | 0.494 | 0.512 | 0.522 | 0.648 | 0.525 | 0.503 | 0.724 | -0.076 |
| 5 | 0.451 | 0.458 | 0.601 | 0.653 | 0.545 | 0.439 | 0.789 | -0.136 |
| 6 | 0.478 | 0.487 | 0.512 | 0.531 | 0.459 | 0.483 | 0.932 | -0.401 |
| 7 | 0.309 | 0.374 | 0.413 | 0.448 | 0.353 | 0.371 | 0.618 | -0.170 |
| 8 | 0.195 | 0.193 | 0.258 | 0.233 | 0.163 | 0.185 | 0.715 | -0.483 |
| 9 | 0.205 | 0.344 | 0.344 | 0.343 | 0.343 | 0.343 | 0.566 | -0.223 |
| 10 | 0.291 | 0.497 | 0.492 | 0.495 | 0.457 | 0.472 | 0.566 | -0.071 |

The remaining deficit is concentrated in Batches 6 and 8 (-0.401 and
-0.483). Batches 2 and 3 now match or exceed the paper's CDCNN
(+0.061, +0.004) using a plain ResNet with no CDCNN component
at all.

### Status against the paper

| Reference | Target mean |
|---|---:|
| paper ResNet baseline | 0.6344 |
| paper CDWC | 0.6705 |
| paper CDCNN | 0.7230 |
| this project, previous best (`B0-LN`, v6.3 ablation) | 0.4922 |
| **this project, new best (`B0-LN-PS`)** | **0.5569** |

The gap to the paper's plain baseline narrows from 0.1422 to 0.0775.

### Execution note

The first evaluation attempt failed after all 30 checkpoints had been evaluated:
`write_report` assumed any non-canonical configuration carries a
`confound_ablation` block, which this one does not. The failed attempt is
retained at `20260920T172721243924Z_cdcnn_v6_4_input_norm_full` with its
`evaluation_failure.json`. The report generator was generalized, a regression
test covering all three configurations was added, and the evaluation was re-run
into the directory named above. Target batches were therefore opened twice, both
times after every checkpoint was frozen; no training, selection, or
configuration decision followed either load.
