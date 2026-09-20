# Duplication control (v6.8)

## Result

Setting `perturbation_scale = 0.0` makes each generated view a bit-exact copy of
its anchor: no noise, no new information, only a doubled training set. The
penalty is unchanged.

| A1 noise scale | Training rows | Batch 1 CV | Target mean | vs no augmentation | Seeds improved |
|---|---:|---:|---:|---:|---:|
| 0.00 (exact duplicates) | 890 | 0.9789 | 0.5499 | -0.0114 | 1/5 |
| 0.05 | 890 | 0.9784 | 0.5508 | -0.0105 | 1/5 |
| 0.20 | 890 | 0.9739 | 0.5514 | -0.0099 | 1/5 |
| 0.50 | 890 | 0.9712 | 0.5486 | -0.0127 | 1/5 |
| 1.00 (canonical) | 890 | 0.9717 | 0.5482 | -0.0131 | 1/5 |
| no augmentation | 445 | 0.9708 | 0.5613 | — | — |

Run: `20260920T205038327586Z_cdcnn_v6_8_duplication_control_full`, 5 checkpoints, audit `passed`.
Other rows are the v6.6 and v6.7 runs, whose stages reproduce bit-identically.

## Finding

**The A1 "augmentation penalty" is not caused by augmentation.** Exact
duplication costs -0.0114, indistinguishable from the canonical noisy
augmentation at -0.0131. Across the whole range from bit-exact copies to
full-variance noise the target mean varies by only 0.0032, while every
variant sits about 0.01 below the un-augmented backbone.

What augmented stages share is not their content but their schedule: 890 rows
instead of 445 under a fixed 100-epoch budget, so **twice as many optimizer steps
and twice as many learning-rate-schedule steps per epoch**. That, not the
generated data, is what costs target accuracy.

## Consequence for earlier results

Every comparison in this project that adds augmentation also doubles the number
of gradient updates. That includes:

- the v6.3 canonical result that A1 is worse than B0 (-0.0376 in the
  2026-09-11 full run),
- the v6.5 step "augmentation + feature generation" (-0.0224),
- the v6.6 step "A1 augmentation alone" (-0.0131).

None of these isolate augmentation: each measures augmentation *plus* a doubled
schedule, and this control shows the schedule alone accounts for essentially all
of it. The feature-generation and contrastive steps are unaffected, because both
sides of those comparisons are augmented and therefore matched.

## What this does not say

It does not show that augmentation helps. It shows the current protocol cannot
answer the question: the augmented and un-augmented arms differ in two ways at
once. It also does not show that 100 epochs is wrong for the un-augmented
stages; `B0-stab-PS` is the best stage measured in this project.

## Options, none applied

Resolving this changes the training protocol, which
`CDCNN_four_experiment_spec_v6.md` fixes at exactly 100 epochs with early
stopping disabled. That is a protocol decision, so nothing here has been changed.

1. **Match optimizer steps.** Train augmented stages for 50 epochs so both arms
   take the same number of updates. Cheapest, and directly tests the hypothesis.
2. **Match epochs and rows.** Subsample the augmented set to 445 rows per epoch
   (originals and views mixed), keeping the schedule identical.
3. **Report both.** Keep 100 epochs as canonical and add a step-matched arm as a
   declared sensitivity experiment, the way `perturbation_scale` is handled.

Option 1 or 2 would let the project state, for the first time, what the paper's
augmentation actually contributes.
