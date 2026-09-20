# CDCNN v6.3 contract

Read the canonical config and implementation while working; this reference is a decision guide, not a substitute for live code.

## Shared protocol

- Canonical stages: B0, A1, A2-semantic, A3. `A2-paper-literal` is separately named and diagnostic only. `B0-LN`, `B0-stab`, and `A2-stab` exist only in the A3 confound ablation configuration.
- Seeds: `1042`, `2024`, `3407`, `42`, `123` for every canonical stage.
- Source CV: persisted five-fold Batch 1 assignments, with exact duplicates kept within one fold.
- Each CV `StandardScaler` fits original Batch 1 training-fold rows only. Each final scaler fits all original Batch 1 rows only. Augmented rows and target rows never enter `fit`.
- Training: CUDA `cuda:0`; SGD, learning rate 0.001, momentum 0.9, weight decay 1e-4; `StepLR(step_size=25, gamma=0.5)`; batch size 64; exactly 100 epochs; no early stopping; epoch 100 checkpoint.
- Architecture: `[N,1,128]`; five residual blocks with channels 32, 64, 128, 256, 128; kernel-3 Conv1d main paths and 1x1 Conv1d shortcuts; no backbone pooling; feature generation after Block 3; flatten 128x128; FC128 then FC6.
- B0, A1, and A2 modes use BatchNorm1d at FC128. A3 v6.3 uses LayerNorm(128) with no running statistics.

`validate_config` intentionally rejects drift from these values. Changing them defines a new protocol and requires explicit experiment-design approval, synchronized documentation/config/code/tests, and new validation artifacts.

## Stage deltas

### B0

Original standardized source rows; cross-entropy only.

### A1

Adds one fixed generated view per source-training context while retaining originals. Pair each anchor with a same-class, non-self source row. Mix population variances, then map mathematical variance to NumPy's standard-deviation API with `scale = sqrt(v_mix)`. Lambda is Uniform(0,1); perturbation scale is 1.0. Preserve the anchor label and record provenance.

### A2-semantic

Uses A1 inputs and decomposes the Block-3 tensor:

- pooled/nearest-upsampled branch = low-frequency-like style component;
- residual = high-frequency-like detail component.

Restyle the low-frequency-like component. Channel statistics reduce over length and have shape `[B,128,1]`; use population variance and `sqrt(var + epsilon)`. Sample means normally. Sample positive standard deviations by the canonical Log-Normal rule using `log(max(sigma_center, epsilon))` and `sigma_spread / max(sigma_center, epsilon)`. Detach sampled-style distribution statistics. Recombine with the untouched residual.

Pass original and generated latents through the shared tail. The loss is the average of the two raw-logit cross-entropies plus 0.5 times the mean per-sample squared L2 distance between softmax probability vectors.

### A2-paper-literal

Restyles the residual branch printed as L in the paper while retaining the project's channel-wise adaptation. It is not an alias, replacement, or preferred version of A2-semantic.

### A3 v6.3

Extends A2-semantic with a shared linear 128-dimensional projection, unit-sphere normalization, supervised contrastive temperature 0.07, and weight 0.5. The current reduction averages positive terms per anchor and then averages over anchors. Do not copy the older v6 sum reduction.

A3-only code lives in `src/a3_stage.py` (bounds, contrastive objective, gradient clipping, and the stage-membership tuples); `src/cdcnn_ablation.py` imports it and re-exports the names, so existing import paths still work.

A3-only stabilization:

- LayerNorm instead of BatchNorm running state;
- log-sigma dispersion `[0,2]`, sigma `[0.001,10]`;
- residual/generated and contrastive inputs/outputs `[-20,20]`;
- gradient norm clipping at 1.0 with nonfinite gradients treated as errors.

These bounds and exact optimizer/loss constants are project-controlled, not author-provided paper values.

## A3 confound ablation

`configs/cdcnn_v6_3_a3_confound.json` (implementation version
`CDCNN_v6.3_A3_confound_ablation`) runs six stages: B0, B0-LN, B0-stab,
A2-semantic, A2-stab, and A3. It exists because v6.3 gave A3 LayerNorm, hard
bounds, and gradient clipping together with the contrastive loss, so the A3
gain cannot be attributed to contrastive learning alone.

- `B0-LN`: B0 with LayerNorm only.
- `B0-stab`: B0 with LayerNorm, hard bounds, and gradient clipping.
- `A2-stab`: A2-semantic with that package; identical to A3 minus the contrastive loss.

`LAYERNORM_STAGES` and `STABILIZED_STAGES` in `src/cdcnn_ablation.py` decide which
stages get which safeguard; `validate_config` pins the stage list per
implementation version, and the launcher reads its stage list from the validated
config. Every other protocol value matches the canonical configuration. Do not
mix stages from this configuration with canonical four-stage results, and
compare only within one software environment: torch build differences can change
a single validation sample across 100 epochs.

## Leakage boundary

`DataAccessGuard` must remain the only unified raw-batch loader. Before `freeze()`, any target access must raise before the file loader runs. Full orchestration trains and hashes all source checkpoints, verifies global coverage and freeze evidence, and only then loads each target batch once for evaluation. No target-derived result may flow back to retries, selection, or configuration.

Do not import or reuse loaders from historical training implementations for a canonical path. Do not cache target-derived structures into pre-freeze validation.

## Review checklist

When changing CDCNN code or config, verify:

1. Stage equality: only the intended ablation component changes.
2. Tensor shape and branch identity after Block 3.
3. A1 variance-versus-standard-deviation semantics.
4. A2 population-statistics axes, Log-Normal positivity, branch recombination, shared tail, and probability-MSE reduction.
5. A3 mean-over-anchors contrastive reduction and A3-only bounds/normalization/clipping.
6. Source-only scaler and augmentation/statistics scopes.
7. Target access remains impossible before all relevant checkpoints are frozen.
8. No Conv2d constructor or 16x8 reshape enters active Python.
9. Fixed seeds, folds, epochs, scheduler boundaries, and checkpoint metadata remain synchronized.
10. Output audits recompute metrics from predictions and record exclusions as empty when none occurred.

Run `.venv/bin/python -m unittest -q tests.test_cdcnn_v6`. Add behavioral tests for changed invariants rather than tests that merely match documentation text. Any implementation/config hash change invalidates earlier GPU-smoke artifacts for a later launch.
