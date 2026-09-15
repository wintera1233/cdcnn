# A3 confound ablation (v6.3)

## Why

In the v6.3 full run (`runs/20260911T092326995583Z_cdcnn_v6_3_full`) A3 was the only
stage to improve target accuracy (target mean 0.460 vs B0 0.410). A3 differs from
A2-semantic in more than the contrastive loss: v6.3 also gave A3 LayerNorm instead of
BatchNorm, ±20 hard bounds after every residual block and on the generated features,
sigma bounds on style sampling, and gradient clipping at max-norm 1.0. The gain can
therefore not be attributed to supervised contrastive learning without separating
those changes.

## Stages

| Stage | Normalization | Hard bounds + gradient clipping | Feature generation + MSE | Contrastive |
|---|---|---|---|---|
| B0 | BatchNorm | no | no | no |
| B0-LN | LayerNorm | no | no | no |
| B0-stab | LayerNorm | yes | no | no |
| A2-semantic | BatchNorm | no | yes | no |
| A2-stab | LayerNorm | yes | yes | no |
| A3 | LayerNorm | yes | yes | yes |

Reading the ladder:

- B0 → B0-LN: effect of LayerNorm alone.
- B0-LN → B0-stab: effect of hard bounds and gradient clipping.
- B0-stab → A2-stab: effect of A1 augmentation + feature generation + MSE under the A3 package.
- A2-stab → A3: effect of the contrastive loss alone.

B0, A2-semantic, and A3 are re-run as in-run references. The refactor was checked to
reproduce the canonical stages bit-for-bit on the same seed, so they should match the
v6.3 full run.

## Protocol

Everything else is unchanged from `configs/cdcnn_v6.json`: Batch 1 only for training,
scaling, and CV; five seeds; 100 epochs; SGD/StepLR; batch size 64; targets loaded once
after all 30 checkpoints are frozen. The configuration is
`configs/cdcnn_v6_3_a3_confound.json` (implementation version
`CDCNN_v6.3_A3_confound_ablation`).

```bash
python scripts/run_cdcnn_v6_full.py gpu-smoke --config configs/cdcnn_v6_3_a3_confound.json
python scripts/run_cdcnn_v6_full.py launch --config configs/cdcnn_v6_3_a3_confound.json \
  --max-workers 1 --max-attempts 2 --gpu-smoke-run runs/<passing_gpu_smoke>
```
