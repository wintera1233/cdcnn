# CDCNN v6 implementation audit

Audit date: 2026-09-10 UTC  
Specification: `CDCNN_four_experiment_spec_v6.md`  
Outcome: **implemented and validation-ready; full 100-epoch experiments not started**

## Scope and evidence

The audit covered the executable model, loss, augmentation, training, checkpoint,
data-loading, and artifact paths. The primary pre-v6 files were
`src/resnet_1d_baseline.py`, `src/a1_formal.py`, `src/a2_formal.py`, their JSON
configurations, and `scripts/run_cdcnn_ablation.py`. Existing data and run
directories were treated as immutable. No existing run was overwritten or
deleted.

The v6 canonical implementation is now `src/cdcnn_ablation.py`, configured only
by `configs/cdcnn_v6.json`. B0, A1, A2-semantic, and A3 use the same model and
training functions. A2-paper-literal is an explicitly named diagnostic mode.

## Findings and dispositions

| Area | Pre-v6 finding | Severity | v6 disposition |
|---|---|---:|---|
| Unified entry point | `scripts/run_cdcnn_ablation.py` imported the nonexistent `src.cdcnn_ablation`; its choices omitted B0 and distinguished neither A2 mode. | Blocking | Implemented the module and explicit B0, A1, A2-semantic, A2-paper-literal, and A3 choices. `A2` is a documented compatibility alias for A2-semantic. |
| Shared training protocol | B0 searched scheduler candidates under a 30-epoch/patience-8 rule; A1 and old A2 used 30 epochs, patience 8, and no scheduler. | Blocking | One strict validator now locks all stages to SGD, lr 0.001, momentum 0.9, weight decay 1e-4, batch 64, StepLR(25, 0.5), exactly 100 epochs, and no early stopping. The epoch-100 checkpoint is fixed before training. |
| Seeds | Earlier formal code used one final seed plus derived fold seeds, not the five required fixed seeds. | Major | Full v6 runs use 1042, 2024, 3407, 42, and 123 for every stage and the same saved five folds. |
| B0 target access | The old B0 loader opened Batches 1–10 before CV and model fitting, even though target arrays were not passed to training. | Major protocol violation | The unified loader refuses target access before all source checkpoints are frozen. Only Batch 1 is loaded for validation, CV, and final fitting. |
| A1 Gaussian parameter | Old A1 mixed per-sample standard deviations and passed the mixture directly as NumPy `scale`. v6 specifies mixing variances and passing `sqrt(v_mix)`. | Blocking semantic error | `generate_a1_views` computes population variances, mixes those variances, and uses their square root as the API scale. Provenance records both values. |
| A1 perturbation multiplier | Old A2 inherited an A1 sensitivity setting of 0.5. | Major | Canonical v6 fixes `perturbation_scale=1.0`; noncanonical scales are rejected. |
| A2 identity | Old A2 used pooled/upsampled plus residual branches but exposed only the ambiguous name A2. | Major | Canonical A2-semantic restyles the pooled/upsampled low-frequency-like component. A2-paper-literal restyles the residual component printed as L. They are separate generated-feature procedures and separate stage names. |
| A2 component sigma | Old A2 used `std(...) + epsilon`. v6 requires `sqrt(var(...) + epsilon)`. | Major | Corrected to population variance over length with shape `[B,128,1]`, followed by square root after epsilon addition. |
| A2 style distribution | Old A2 fit a Gaussian to `log(sigma)` using its empirical log-space spread. That is not the v6 canonical rule. | Blocking semantic error | The new implementation uses `log(max(mean(sigma), eps))` as center and `std(sigma)/max(mean(sigma), eps)` as log-space spread, then exponentiates. Positivity is guaranteed without clipping or softplus. |
| A2 objective | Old A2 explicitly disabled MSE and optimized only equal-weight branch CE. | Blocking | A2-semantic and A2-paper-literal use average raw-logit CE plus 0.5 times mean per-sample squared-L2 distance between softmax probabilities. |
| A3 | No executable A3 implementation existed. | Blocking | Added one shared linear 128-dimensional projection head, unit normalization, temperature 0.07, and the specified supervised contrastive objective over both branches. A3 reuses A2-semantic generation, CE, and MSE. |
| Contrastive reduction | The v6 equation sums the positive-averaged loss over anchors. | High-impact setting | Implemented the stated sum exactly. The smoke loss is correspondingly much larger than CE; this is recorded, finite, and not silently changed to a mean. |
| Backbone equality | Historical stages used related backbone classes but separate training implementations. | Major | All stages instantiate the same five blocks (32,64,128,256,128), 1D kernel-3 main path, 1x1 shortcuts, flatten, FC128, BatchNorm1d, and FC6. Only A3 adds a training-only projection head. |
| Feature insertion and sharing | Old A2 inserted generation after Block 3 and shared the tail correctly. | Pass | Preserved. Both branches are concatenated and passed through the same Blocks 4–5, FC128, BatchNorm, and FC6 objects. |
| Scaling | Old formal A1/A2 fit fold-local and all-source scalers correctly. | Pass | Preserved and strengthened with per-context `scaler_fit_scopes.csv`; augmentation is applied only after source-fitted scaling. |
| Target-derived preflight | Old A2 read a prior all-batch validation artifact before training. It used structural facts rather than target metrics, but this weakens the strict load-order proof. | Moderate | Removed from the canonical pre-freeze path. Batch 1 validates directly; full structural validation happens only after freeze in a full run. |
| Reproducibility artifacts | Earlier formal runs had substantial provenance, but protocols and naming differed. | Mixed | Full v6 runs save resolved configuration, raw/config/spec/fold hashes, software and code identities, fold and epoch histories, checkpoints, scaler scopes, augmentation provenance, feature geometry, style statistics, target predictions, confusion matrices, metrics, access log, and integrity audit. |

## Canonical stage deltas

| Stage | Training inputs | Generated feature branch | Loss |
|---|---|---|---|
| B0 | Original source rows | None | CE |
| A1 | Originals plus one fixed v6 statistical view per source row | None | CE |
| A2-semantic | Same as A1 | Restyle pooled/upsampled low-frequency-like component | average branch CE + 0.5 probability MSE |
| A3 | Same as A1 | Identical to A2-semantic | A2 loss + 0.5 supervised contrastive sum |
| A2-paper-literal | Same as A1 | Restyle residual component printed as L | same CE/MSE form as A2-semantic; diagnostic only |

## Project-controlled choices retained explicitly

- Same-class non-self A1 pairing and inclusion of both originals and fixed views.
- Detaching the sampled style-distribution statistics from gradient flow. This
  affects backpropagation semantics, not their source-only scope, and is recorded
  in configuration.
- Momentum, weight decay, exact StepLR schedule, projection-head architecture,
  temperature, and loss weights.
- Channel-wise A2 statistics for both A2 modes. “Paper-literal” refers to the
  paper's branch assignment and H/L notation; it does not silently replace the
  v6 channel-wise project adaptation with the paper's globally reduced printed
  statistics.
- Population (`unbiased=False`) spread estimates across each active source
  mini-batch for the style-sampling distributions.

## Validation completed

- Unit tests: 14 passed. Coverage includes locked configuration, architecture,
  variance-correct A1 sampling, A2 branch semantics, population statistics,
  exact log-normal sampling, positive sigma, probability MSE, A2 loss assembly,
  identical shared initialization under a fixed seed, A3 inheritance of
  A2-semantic, supervised contrastive math, StepLR boundaries, and target-access
  denial.
- Batch-1-only smoke suite: all five modes passed one epoch and one optimization
  batch with finite losses and predictions. This run is deliberately truncated
  and is not result-bearing.
- Leakage audit: passed; see `leakage_target_access_audit_v6.md` and the smoke
  run's machine-readable audit.

## Historical-code status

The old implementation modules and JSON files remain unchanged as historical
evidence for existing runs. They are not v6-compliant and must not be used for a
new canonical experiment. The existing public wrappers for B0, A1, and A2 now
route to the unified v6 module and configuration. Historical runs remain
interpretable through their saved implementation hashes and configurations.

## Release decision

The implementation is ready for a deliberate full run, but none was started in
this session. Before an overnight launch, select the stage sequence explicitly;
the canonical order is B0, A1, A2-semantic, A3. A2-paper-literal should be run
only as the separately reported diagnostic.
