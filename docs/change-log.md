# Change log

Branch `exp/a3-confound-ablation`, newest first. Each entry records what changed,
why, and what was verified. Run directories are never overwritten; each
experiment writes its own.

## v6.10 contrastive placement — results (2026-09-22)

**Result.** Moving the contrastive loss to the paper's position — the pre-FC128
latent, unit-sphere normalized, no learned head — leaves it worth +0.0002 against
the same model without it (2/5 seeds), against +0.0025 for this project's
post-FC128 learned head (3/5). The two placements differ by −0.0023 (1/5), so the
head was not what held the term back.

Five independent measurements now agree: across v6.3, v6.5, v6.6, v6.9 and v6.10
the supervised contrastive term measures between −0.005 and +0.003 target mean.
Under this protocol its effect is consistently indistinguishable from zero.
Likely because Batch 1 features are already near-separable, so a term that pulls
same-label samples together has little left to do — and nothing in it sees the
target domains. Details in `docs/contrastive-placement.md`.

**Determinism.** Both in-run controls reproduced their v6.9 target means to ten
decimal places across separate container launches.

## v6.9 epoch alignment — results (2026-09-21)

**Result.** Matching the optimizer-step count recovers about a third of the
augmentation penalty (+0.0043); augmentation still costs −0.0089 against no
augmentation. The doubled schedule was a real confound but not the whole story:
with twice the data at a fixed epoch budget, step count and per-sample exposure
cannot both be matched, and the aligned arm sees each row ~50 times instead of
100 (its Batch 1 CV falls to 0.9622 from 0.9708).

Under the aligned schedule the CDCNN components stop being negative: feature
generation on the paper's branch +0.0014, contrastive loss +0.0025, and the full
aligned model sits level with the plain backbone (−0.0050, 2/5 seeds). Alignment
also tightens the seed spread monotonically, from 0.0364 for the anchor to
0.0136 for the full aligned model — the most stable stage trained here.

The alignment is a separate implementation version and has **not** been adopted
as canonical; `docs/epoch-alignment.md` records both schedules and what each
holds constant.

**Environment.** Run in the pinned container after the Xid 31 fault. The anchor
`B0-stab-PS` reproduced its source CV folds bit-identically against the host
stack (torch 2.8.0+cu126 vs 2.5.1+cu121), all five seeds; its target mean
differs in the fourth decimal (0.561334 against 0.561300), so cross-stack
target differences below about 0.004 are environment, not effect.

## v6.8 duplication control — results (2026-09-20)

**Result, and the most consequential finding of the session.** With
`perturbation_scale = 0.0` each generated view is a bit-exact copy of its anchor,
yet the penalty is unchanged: −0.0114 against −0.0131 for canonical augmentation.
Across bit-exact copies to full-variance noise the target mean moves by 0.0032.

**The A1 penalty is therefore a schedule artefact, not an augmentation effect.**
Augmented stages train on 890 rows instead of 445 under a fixed 100-epoch budget,
so they take twice the optimizer and scheduler steps. Every A1-versus-B0
comparison in this project — including the canonical v6.3 result that A1 is worse
than B0 (−0.0376) — measures augmentation plus a doubled schedule. Feature
generation and contrastive comparisons are unaffected, since both of their arms
are augmented and therefore matched.

Resolving it changes the protocol (100 epochs, early stopping disabled), so
**nothing was changed**; options are recorded in `docs/duplication-control.md`
for a decision.

## v6.7 augmentation scale sensitivity — results (2026-09-20)

**Result.** Reducing the augmentation noise does not rescue it. Scales 0.5, 0.2
and 0.05 all land 0.010–0.013 below the un-augmented backbone, improving on 1/5
seeds each; a twentyfold reduction in noise changes the target mean by +0.0026.
Batch 1 CV rises as noise falls (0.9784 at scale 0.05) while target accuracy does
not follow. Since a near-copy of each anchor still costs accuracy, the suspect is
the procedure rather than the noise: augmented stages train on 890 rows instead
of 445 and therefore take twice as many optimizer steps under the fixed
100-epoch schedule. Details in `docs/augmentation-scale.md`.

**Next.** A `perturbation_scale = 0.0` control (exact duplicates, no noise)
separates the doubled schedule from the augmentation itself.

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
