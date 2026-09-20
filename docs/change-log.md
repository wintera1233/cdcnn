# Change log

Branch `exp/a3-confound-ablation`, newest first. Each entry records what changed,
why, and what was verified. Run directories are never overwritten; each
experiment writes its own.

## v6.6 paper-literal and augmentation ladder — results (2026-09-20)

**Result.** Augmentation alone is the harmful component: −0.0131 target mean,
improving on 1/5 seeds. The paper's own restyled branch beats this project's
semantic choice on 5/5 seeds (+0.0137 for A2, +0.0201 for A3), so the
physical-semantic renaming in the v6 specification costs accuracy. Feature
generation turns mildly positive on the paper's branch (+0.0044, 4/5), and the
contrastive loss stays neutral (+0.0016, 3/5). The full paper-literal model is
still −0.0071 against the plain backbone because augmentation dominates.
`B0-stab-PS` reproduced bit-identically across the v6.5 and v6.6 runs, evidence
that cross-run stage comparisons in this environment are exact.
Details in `docs/paper-literal-ladder.md`.

## v6.6 paper-literal ladder — implementation (commit `64eb69f`)

Stages `B0-stab-PS -> A1-stab-PS -> A2-lit-PS -> A3-lit-PS`, isolating
augmentation for the first time and measuring the paper's residual-restyling
branch at five seeds. Branch choice follows a `PAPER_LITERAL_STAGES` membership
tuple. Verified 47/47 tests, fifteen pre-existing stages bit-identical.

## v6.5 normalized-input ladder — results (2026-09-20)

**Result.** The CDCNN components do not help even on per-sample-normalized
inputs. Augmentation plus feature generation costs −0.0224 (1/5 seeds improved),
the contrastive loss costs −0.0049 (2/5), and together they cost −0.0229 against
the plain normalized backbone. Input conditioning was therefore not what held
them back: the same steps measured −0.0328 and +0.0029 on unnormalized inputs.
Per-sample inputs do lift the full CDCNN model by +0.0738 (A3 0.4602 to A3-PS
0.5340), confirming the effect is the normalization, not the method. Selected
stage under the source-only rule remains `B0-LN-PS` (best CV 0.9933).
Details in `docs/normalized-input-ladder.md`.

## v6.5 normalized-input ladder — implementation (commit `5e36d9e`)

Stages `B0-LN-PS -> B0-stab-PS -> A2-stab-PS -> A3-PS`, one component per step,
all on per-sample-normalized inputs; `A3-PS` equals v6.3 A3 plus the transform.
The projection head and contrastive term now follow a `CONTRASTIVE_STAGES`
membership tuple. Each shipped config pins its stage lists as frozen literals so
that adding a stage later cannot change what an older config must contain; the
three existing configs keep their hashes. Verified 42/42 tests, all twelve
pre-existing stages bit-identical, fifteen-mode smoke suite.

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
