# CDCNN v6 leakage and target-access audit

Audit date: 2026-09-10 UTC  
Outcome: **passed for the unified v6 implementation and Batch-1 smoke run**

## Access boundary

`DataAccessGuard` is the only raw-batch loader used by the unified pipeline. It
starts in source-only state and raises `ProtocolError` for any declared target
batch. The state changes only after all five final source checkpoints have been
written and hashed. Full-run target loading appears after that transition.

The unit test attempted to access Batch 2 before freeze and confirmed that the
request was rejected before the file loader ran. The Batch-1 smoke access log
contains exactly one raw-file event, for `Dataset/batch1.dat`, with the expected
SHA-256 digest. It contains no target access and no target metric.

## Data-flow findings

| Boundary | Finding | Status |
|---|---|---|
| Fold definition | The saved assignments are identity/line/label checked against Batch 1 by `load_folds`. | Pass |
| Scaling | Each CV scaler fits only original Batch-1 training-fold rows. Each final scaler fits only all original Batch-1 rows. Augmented and target rows never enter `fit`. | Pass |
| A1 partner selection | Candidates come only from the active source training context and must be same-class and non-self. Validation rows are absent. | Pass |
| A2 style estimation | Means and standard deviations are computed only from the active source-training mini-batch. | Pass |
| Validation | Batch-1 validation accuracy is observational under the fixed schedule. It does not choose a checkpoint or trigger early stopping. | Pass |
| Checkpoint | Epoch 100 is fixed in configuration and checked by the validator. All seed checkpoints are saved and hashed before the guard is frozen. | Pass |
| Target loading | Batches 2–10 are loaded once, after freeze, for final evaluation. | Pass by control-flow review; not exercised by the intentionally target-free smoke run. |
| Target labels/metrics | Available only in the post-freeze evaluation block; no value flows back into configuration, augmentation, fitting, or checkpoint selection. | Pass |
| Exclusions/preprocessing | No exclusion, deduplication, clipping, outlier removal, or imputation code exists in the v6 training path. | Pass |
| Raw mutation | All dataset operations are reads. No dataset path is passed to a write operation. | Pass |

## Smoke evidence

Run: `runs/20260910T195333901824Z_cdcnn_v6_batch1_smoke`

- Modes exercised: B0, A1, A2-semantic, A3, A2-paper-literal.
- Source file opened: Batch 1 only, 445 rows × 128 features, six labels, finite.
- Scaler fit: the 355 original rows in saved fold 1's training partition.
- Training: one epoch, one batch of 64 per mode; explicitly not the full protocol.
- Target files opened: zero.
- Target metrics computed: zero.
- Audit artifact: `leakage_target_access_audit.json`, status `passed`.

## Historical implementation note

The pre-v6 B0 module loaded all ten batches before training. Even though it did
not pass target arrays into model fitting, this did not satisfy the strict v6
load-order rule. Old A1/A2 were better ordered, but old A2 consumed prior
target-derived structural validation evidence during preflight. These modules
remain for old-run provenance only; the normal wrapper scripts now route new
runs to the guarded v6 implementation.

## Residual risk and operational control

Directly importing a historical module and calling its `main` function can still
reproduce a pre-v6 workflow. This is intentionally retained for provenance and
is not a canonical path. New work should use `scripts/run_cdcnn_ablation.py` or
the v6-routed stage wrappers. Every full run writes its own access log and fails
at post-run audit if pre-freeze target access, scaler leakage, incorrect epoch,
incomplete prediction coverage, nonpositive sampled sigma, or dataset-structure
validation is detected.
