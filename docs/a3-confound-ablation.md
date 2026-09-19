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

## Results (2026-09-19)

Run: `runs/20260915T025701880293Z_cdcnn_v6_3_a3_confound_full`. Thirty
checkpoints (six stages x five seeds), all frozen before any target file was
opened; the run's `leakage_target_access_audit.json` status is `passed`.

| Stage | Batch 1 CV | Target mean | Target-mean SD | Pooled |
|---|---:|---:|---:|---:|
| B0 | 0.9784 | 0.4077 | 0.0186 | 0.3943 |
| B0-LN | 0.9910 | 0.4922 | 0.0415 | 0.5077 |
| B0-stab | 0.9654 | 0.4901 | 0.0304 | 0.4993 |
| A2-semantic | 0.9806 | 0.3931 | 0.0084 | 0.3831 |
| A2-stab | 0.9672 | 0.4573 | 0.0233 | 0.4604 |
| A3 | 0.9676 | 0.4602 | 0.0251 | 0.4607 |

### Ladder steps

| Step | Isolates | Mean change | SD | Seeds improved |
|---|---|---:|---:|---:|
| B0 -> B0-LN | LayerNorm alone | +0.0845 | 0.0339 | 5/5 |
| B0-LN -> B0-stab | hard bounds + gradient clipping | -0.0022 | 0.0377 | 1/5 |
| B0-stab -> A2-stab | A1 augmentation + feature generation + MSE | -0.0328 | 0.0169 | 0/5 |
| A2-stab -> A3 | supervised contrastive loss | +0.0029 | 0.0107 | 4/5 |
| B0 -> A3 | all v6.3 A3 changes together | +0.0525 | 0.0264 | 5/5 |
| A2-semantic -> A3 | the comparison v6.3 reported | +0.0671 | 0.0289 | 5/5 |

### Findings

1. **LayerNorm accounts for the whole A3 gain.** Replacing `BatchNorm1d(128)`
   with `LayerNorm(128)` in an otherwise unchanged B0 adds +0.0845 target mean on
   5/5 seeds and also gives the best Batch 1 CV of any stage (0.9910). Under the
   source-only protocol it would have been selected without consulting target
   data.
2. **The contrastive loss contributes about +0.003**, roughly a tenth of A3's
   apparent advantage over A2-semantic and within the seed-to-seed spread.
3. **Hard bounds and gradient clipping do not help accuracy** (-0.0022) and cost
   Batch 1 CV (0.9910 to 0.9654). They remain justified as numerical safety, not
   as performance features.
4. **The CDCNN pipeline is a net negative here.** Adding A1 augmentation, feature
   generation, and the MSE term on top of the stabilized backbone costs -0.0328
   with 0/5 seeds improving.

The resulting ranking is B0-LN ~ B0-stab > A3 ~ A2-stab > B0 > A2-semantic, which
does not reproduce the paper's reported ordering.

### Per-batch mean accuracy

| Batch | B0 | B0-LN | B0-stab | A2-semantic | A2-stab | A3 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.738 | 0.873 | 0.866 | 0.767 | 0.886 | 0.894 |
| 3 | 0.507 | 0.692 | 0.668 | 0.509 | 0.653 | 0.683 |
| 4 | 0.494 | 0.512 | 0.520 | 0.493 | 0.553 | 0.576 |
| 5 | 0.451 | 0.458 | 0.397 | 0.339 | 0.347 | 0.349 |
| 6 | 0.478 | 0.487 | 0.443 | 0.414 | 0.435 | 0.414 |
| 7 | 0.309 | 0.374 | 0.384 | 0.281 | 0.332 | 0.330 |
| 8 | 0.195 | 0.193 | 0.265 | 0.173 | 0.170 | 0.172 |
| 9 | 0.205 | 0.344 | 0.377 | 0.253 | 0.322 | 0.304 |
| 10 | 0.291 | 0.497 | 0.489 | 0.309 | 0.417 | 0.419 |

LayerNorm helps most on the most drifted batches (B10 0.291 to 0.497, B9 0.205 to
0.344, B3 0.507 to 0.692), consistent with BatchNorm applying stored Batch 1
statistics to shifted inputs while LayerNorm normalizes each sample against
itself.

### Execution caveats

- The controller process was killed at 23/30 when its session ended. The
  remaining tasks were retrained from the unchanged code (verified by
  implementation hash) and the interrupted attempt is preserved in its original
  `attempt_1` directory and listed in `source_failure_records.json`.
- `CDCNN_four_experiment_spec_v6.md` was edited between task 28 and the
  evaluation phase, so `input_manifest.json` records two different specification
  hashes across the run. The specification is documentation and is not executed;
  no code, configuration, or data changed during the run.
- Environment: torch 2.8.0+cu126. The 2026-09-11 canonical run used torch
  2.5.1+cu121, so its B0/A2-semantic/A3 numbers differ slightly from the
  references re-run here. Compare stages only within this run.
