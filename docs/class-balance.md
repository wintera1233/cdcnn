# Class balance: a one-seed screen (v6.11)

## Question

`docs/per-class-failure.md` showed that Ethylene is never predicted on any target
batch, that Ammonia and Toluene collapse from Batch 6 onwards, and that nothing
in this project had addressed class imbalance. Batch 1 holds 30 to 98 samples per
gas, the loss is unweighted, and the augmentation preserves the imbalance exactly.

Two standard corrections were screened, both fitted on source training labels
only:

- `B0-wce-PS`: inverse-frequency class weights in the cross-entropy, mean 1,
  Ethylene weighted 2.12.
- `B0-bal-PS`: balanced sampling, each gas drawn equally often per epoch, with
  the epoch size and step count unchanged.

**This was a one-seed pilot (seed 42), not a five-seed experiment.** Run
`20260921T164436164787Z_cdcnn_v6_11_class_balance_one_seed_pilot`, three checkpoints, audit `passed`.
It exists to decide whether a five-seed run is worth 35 minutes of GPU time.

## Result: it works in-domain and does not transfer

Batch 1 cross-validation recall, per gas:

| Gas | `B0-stab-PS` | `B0-wce-PS` | `B0-bal-PS` |
|---|---:|---:|---:|
| Acetone | 0.989 | 0.989 | 0.989 |
| Acetaldehyde | 1.000 | 1.000 | 1.000 |
| Ethanol | 0.988 | 0.988 | 0.988 |
| **Ethylene** | **0.667** | **0.800** | **0.867** |
| Ammonia | 1.000 | 1.000 | 1.000 |
| Toluene | 0.973 | 0.986 | 1.000 |
| **overall CV** | **0.9686** | **0.9797** | **0.9865** |

Target recall, per gas:

| Gas | `B0-stab-PS` | `B0-wce-PS` | `B0-bal-PS` |
|---|---:|---:|---:|
| Acetone | 0.700 | 0.659 | 0.619 |
| Acetaldehyde | 0.858 | 0.852 | 0.854 |
| Ethanol | 0.938 | 0.926 | 0.909 |
| **Ethylene** | **0.000** | **0.000** | **0.000** |
| Ammonia | 0.383 | 0.393 | 0.390 |
| Toluene | 0.032 | 0.060 | 0.146 |
| **target mean** | **0.5521** | **0.5485** | **0.5454** |

## Findings

1. **Balancing does exactly what it should, in-domain.** Ethylene's Batch 1
   recall rises from 0.667 to 0.800 with weighting and 0.867 with balanced
   sampling; Toluene reaches 1.000. Overall CV rises from 0.9686 to 0.9865.
2. **None of it reaches the target batches.** Ethylene stays at exactly 0.000
   under both methods, and the model still never emits the label: its predicted
   share is 0.0% in all three stages.
3. **Toluene is the one class that responds**, 0.032 to 0.146 with balanced
   sampling — but it is paid for by Acetone, 0.700 down to 0.619.
4. **Target accuracy does not improve**: 0.5521 without balancing, 0.5485
   weighted, 0.5454 balanced. At one seed these differences are not meaningful,
   which is precisely the point — there is no effect worth five seeds.

## What this establishes

**Ethylene's target failure is not an imbalance problem.** A model that
classifies Ethylene at 0.867 on Batch 1 still never predicts it on Batches 2–10.
The cause is domain shift: the drifted Ethylene samples land where the model has
learned Acetone, and giving the class more weight or more draws during training
does not move that boundary in the target domain.

## A protocol warning

Batch 1 CV ranks these three stages in the exact reverse of their target
accuracy:

| Stage | Batch 1 CV | Target mean |
|---|---:|---:|
| `B0-bal-PS` | **0.9865** (best) | 0.5454 (worst) |
| `B0-wce-PS` | 0.9797 | 0.5485 |
| `B0-stab-PS` | 0.9686 (worst) | **0.5521** (best) |

The source-only protocol selects on Batch 1 CV, so it would have chosen the worst
of the three. This is a concrete instance of a limit the protocol has always had:
CV measures in-domain fit, and under drift that can anti-correlate with target
accuracy. Worth remembering whenever a future change improves CV.

## Recommendation

Do not run the five-seed version. The screening cost six minutes against
thirty-five, and it answers the question: class balance is not the lever for
Ethylene.

If Ethylene is pursued further, the evidence points at the feature space rather
than the loss — its target samples fall inside Acetone's region, so the useful
questions are whether any Batch-1-fittable representation separates them, and
whether the 30 available training samples can support that at all.
