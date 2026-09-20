# CDCNN v6.3 A3 numerical stabilization

Status: implemented, unit-tested, and executed. The five-seed run
`runs/20260911T092326995583Z_cdcnn_v6_3_full` completed under these settings.

CDCNN v6.3 keeps the v6 experiment protocol and changes only A3 numerical
behavior. Batch 1 remains the sole pre-freeze data source. The five seeds, SGD
settings, StepLR settings, batch size, fixed 100 epochs, and disabled early
stopping are unchanged.

## A3 safeguards

- The A3 FC128 normalization is `LayerNorm(128)`. It has no global running mean,
  running variance, or batch counter, so synthetic samples cannot corrupt
  inference-time normalization state. B0, A1, A2-semantic, and
  A2-paper-literal retain `BatchNorm1d(128)`.
- Log-Normal dispersion is clamped to `[0, 2]`, and sampled sigma is clamped to
  `[0.001, 10]`. The log-sigma value is clamped before exponentiation.
- A3 residual/intermediate generated values and the input/output vectors of the
  contrastive projection are clamped to `[-20, 20]`.
- The supervised contrastive objective averages over anchors rather than
  summing them. Cross-entropy and probability MSE retain their existing
  batch-mean reductions and weights.
- A3 calls `clip_grad_norm_` with `max_norm=1.0` and
  `error_if_nonfinite=True` after backpropagation and immediately before every
  optimizer step. Other stages do not receive this A3-only operation.

All bounds are explicit project-controlled numerical safeguards. They were not
tuned on Batches 2–10 and are not represented as paper-derived values.

## Source layout

The A3-specific code is `src/a3_stage.py`: `A3_STABILITY_DEFAULTS`,
`A3NumericalStabilizer`, `supervised_contrastive_mean`, `contrastive_term`,
`clip_a3_gradients`, and the `LAYERNORM_STAGES`/`STABILIZED_STAGES` tuples.
`src/protocol.py` holds `ProtocolError` so that module and the shared pipeline
can raise the same type without importing each other. `src/cdcnn_ablation.py`
imports and re-exports those names, so `from src.cdcnn_ablation import ...`
continues to work. Both new modules are hashed into every run's
`implementation_files` manifest.

## Traceability

The canonical configuration remains `configs/cdcnn_v6.json`, and
`scripts/run_cdcnn_v6_full.py` remains the only full-training entry point. The
A3 confound ablation adds one further permitted configuration,
`configs/cdcnn_v6_3_a3_confound.json` (see `docs/a3-confound-ablation.md`).
Because these safeguards apply to A3 alone, they are confounded with the
contrastive loss in any A3-minus-A2 comparison; that ablation separates them.

The canonical configuration records implementation version
`CDCNN_v6.3_A3_numerical_stabilization` and the exact A3 constants. Future
checkpoints record the same configuration, and source-run audits verify the
absence of A3 BatchNorm running state plus sigma and post-clipping gradient
bounds.

Any previously passing GPU-smoke artifact has an obsolete implementation/config
hash after this change. A new GPU smoke must be run only after explicit approval
and before any full run.
