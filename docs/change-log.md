# Change log

Branch `exp/a3-confound-ablation`, newest first. Each entry records what changed,
why, and what was verified. Run directories are never overwritten; each
experiment writes its own.

## v6.4 input normalization — results (2026-09-20)

**Result.** Per-sample input normalization is the strongest single change
measured in this project: +0.1087 target mean on 5/5 seeds. Combined with
LayerNorm, `B0-LN-PS` reaches **0.5569** target mean and 0.9933 Batch 1 CV, the
best of either metric here, up from 0.4922. Bounding extreme values (signed-log,
clip ±5) does nothing, so the mechanism is removing each sample's offset and
gain, not taming outliers. Details in `docs/input-normalization.md`.

**Fix.** `write_report` assumed a `confound_ablation` block existed in any
non-canonical config and raised `KeyError` on the v6.4 config after all 30
checkpoints had been evaluated. Report generation is now descriptor-driven, and
`tests/test_input_normalization.py` renders a report for all three
configurations as a regression test. The failed attempt is retained; the
evaluation was re-run into a new directory.

## v6.4 input normalization — implementation (commit `3f04cb1`)

**Why.** The paper's plain ResNet baseline (0.6344) beat every stage trained
here, and the confound ablation showed normalization, not the CDCNN modules,
moves this dataset.

**What.** `src/input_transform.py` with four parameter-free transforms applied
after the Batch-1-fitted scaler; stages `B0-PS`, `B0-LN-PS`, `B0-LN-LOG`,
`B0-LN-CLIP`; implementation version `CDCNN_v6.4_input_normalization` and its
config; the transform kind recorded in every checkpoint so the post-freeze
target path reproduces training exactly.

**Verified.** 37/37 tests; all eight pre-existing stages train bit-identically;
twelve-mode smoke suite passes; GPU gate passes. The confound config stayed
byte-identical to the one its completed run recorded (`960026829038`).

## Paper comparison (commit `fffd44f`)

Read the main text, supplement S1–S5, and Fig. 2, and recorded what the paper
specifies against what this project implements: three undeclared differences
(contrastive loss applied to the pre-FC128 feature with no learned head;
variance-versus-std in Eq. (16); ResNet5's 512-wide inner convolution), the
paper's internal contradiction over the statistics shape, and five ranked
improvement points. The specification's traceability table was corrected.
See `docs/paper-vs-implementation.md`.

## A3 code split (commit `ef7c9ce`)

A3's hard bounds, contrastive objective, gradient clipping and stage-membership
tuples moved to `src/a3_stage.py`, with `ProtocolError` in `src/protocol.py` to
avoid a circular import. `src/cdcnn_ablation.py` re-exports the names, so
existing imports still work. Verified bit-identical across all eight stages, and
a frozen A3 checkpoint still loads and predicts.

## A3 confound ablation — results (commit `41ca140`)

LayerNorm accounts for +0.0845 target mean (5/5 seeds); the supervised
contrastive loss accounts for +0.0029; hard bounds and clipping are
accuracy-neutral; augmentation plus feature generation costs −0.0328. The v6.3
A3 result therefore measures a normalization change, not contrastive learning.
See `docs/a3-confound-ablation.md`.

## Documentation sync (commit `0768f50`)

The specification still described pre-v6.3 A3, the implementation audit recorded
the superseded contrastive sum as current, and the README claimed no full run
existed. Fixed the form-feed-corrupted `\frac`, filled the comparison table from
the completed run's own CSVs, and added a v6.3 deviations section.

## A3 confound ablation — implementation (commit `b4a5446`)

Added stages `B0-LN`, `B0-stab`, `A2-stab` so that A3's bundled changes could be
separated from its contrastive loss, plus a dedicated config; the launcher now
takes its stage list from the validated configuration.
