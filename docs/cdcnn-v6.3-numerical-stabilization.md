# CDCNN v6.3 A3 numerical stabilization

Status: implemented and unit-tested; no smoke, pilot, CV, or full training run was started.

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

## Traceability

The canonical configuration remains `configs/cdcnn_v6.json` because project
instructions permit full execution only through that configuration and
`scripts/run_cdcnn_v6_full.py`. It now records implementation version
`CDCNN_v6.3_A3_numerical_stabilization` and the exact A3 constants. Future
checkpoints record the same configuration, and source-run audits verify the
absence of A3 BatchNorm running state plus sigma and post-clipping gradient
bounds.

Any previously passing GPU-smoke artifact has an obsolete implementation/config
hash after this change. A new GPU smoke must be run only after explicit approval
and before any full run.
