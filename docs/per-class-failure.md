# Per-class failure: Ethylene, Ammonia, Toluene

## Summary

The remaining gap to the paper is not spread across the six gases. Three of them
collapse on the target batches, and together they are about half of every target
batch. **Ethylene is never predicted correctly anywhere: 0 correct out of 9,530
target samples across five seeds.**

## The gas label mapping was wrong

Before reading any confusion matrix: this project's documented mapping
(`1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene`) does
not match the data. Matching the per-label counts in each `.dat` file against the
paper's Table 2 per-gas counts gives:

| Label | Gas | Batch 1 n |
|---|---|---:|
| 1 | Acetone | 90 |
| 2 | Acetaldehyde | 98 |
| 3 | Ethanol | 83 |
| 4 | Ethylene | 30 |
| 5 | Ammonia | 70 |
| 6 | Toluene | 74 |

This holds exactly in eight of ten batches. The two exceptions are errors in the
paper's own table: its Batch 5 row lists 189 samples across the gases but a total
of 191 (the files hold 197), and its Batch 7 Ethylene count is one higher than
the file. The distinctive count patterns elsewhere (514 / 574 / 110 / 29 / 606 /
467 in Batch 6) leave no ambiguity.

Corrected in `configs/pca.json`, `scripts/plot_cdcnn_v6_full_results.py`,
`scripts/predict_a3_v6_3_batch2_gas4.py` and the skill reference. **Gas names and
figure legends in artifacts produced before 2026-09-22 are mislabelled; the
accuracy numbers are unaffected**, because nothing in the training or evaluation
path uses gas names — only integer labels.

## Per-class recall by batch

Best stage `B0-stab-PS`, five-seed pooled:

| Gas | Batch 1 train n | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acetone | 90 | 0.94 | 0.95 | 0.87 | 0.99 | 0.80 | 0.85 | 1.00 | 1.00 | 0.94 |
| Acetaldehyde | 98 | 0.97 | 0.99 | 0.95 | 1.00 | 0.91 | 0.92 | 0.39 | 0.07 | 0.65 |
| Ethanol | 83 | 1.00 | 0.91 | 0.78 | 0.75 | 0.75 | 0.89 | 0.93 | 1.00 | 0.81 |
| **Ethylene** | **30** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** |
| **Ammonia** | 70 | 0.99 | 0.82 | 0.78 | 0.73 | **0.11** | **0.09** | **0.00** | **0.00** | **0.33** |
| **Toluene** | 74 | 1.00 | — | — | — | **0.02** | **0.01** | **0.03** | **0.00** | **0.13** |

Ethylene, Ammonia and Toluene make up 26–66% of each target batch (52% of B2,
48% of B6, 66% of B8).

## What the errors look like

Batch 6, `B0-stab-PS`, share of each true gas predicted as each gas:

| true \ pred | Acetone | Acetaldehyde | Ethanol | Ethylene | Ammonia | Toluene | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| Acetone | **0.80** | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 514 |
| Acetaldehyde | 0.03 | **0.91** | 0.05 | 0.00 | 0.00 | 0.01 | 574 |
| Ethanol | 0.20 | 0.04 | **0.75** | 0.00 | 0.01 | 0.00 | 110 |
| Ethylene | 0.86 | 0.14 | 0.00 | **0.00** | 0.00 | 0.00 | 29 |
| Ammonia | 0.18 | 0.20 | 0.40 | 0.00 | **0.11** | 0.11 | 606 |
| Toluene | 0.98 | 0.00 | 0.00 | 0.00 | 0.00 | **0.02** | 467 |

Batch 8 is the same story: Ammonia (143 of 294 samples) goes 68% to Ethanol and
30% to Acetone; Toluene goes 92% to Acetone.

## Ethylene

1. **It is already the weakest class in-domain.** Batch 1 CV recall is 0.660,
   while every other gas is above 0.98. It has 30 training samples, the smallest
   class.
2. **On the targets the model never outputs it at all.** Predicted label shares
   across all target samples: Acetone 44.2% (true 18.4%), Acetaldehyde 25.0%
   (21.0%), Ethanol 19.6% (11.6%), Ammonia 8.7% (21.8%), Toluene 2.5% (13.1%),
   **Ethylene 0.0% (14.2%)**.
3. **Its samples go to Acetone**: 78%, then Ethanol 13% and Acetaldehyde 9%.
4. **It is not a property of this implementation.** Every stage gives the same
   result (0.0000–0.0004 pooled recall), and so do the historical SVM baselines
   trained on Batch 1: PCA-SVM 0.002, no-PCA SVM 0.002.

Since Ethylene is 14.2% of the target samples, a model that never predicts it is
capped at 0.858 target accuracy before any other error.

## Interpretation

The prediction distribution is badly skewed towards the three largest Batch 1
classes. Acetone is predicted 2.4 times as often as it occurs; Ethylene, Ammonia
and Toluene are predicted far less often than they occur. That is the signature
of a classifier whose decision regions for small, drifting classes have been
absorbed by their neighbours — not of a model that has learned the wrong
features for one gas.

Nothing in this project has yet addressed class imbalance. Batch 1 is imbalanced
(30 to 98 samples per gas), the loss is unweighted cross-entropy, and the
augmentation generates one view per sample, so it preserves the imbalance exactly
rather than correcting it.

## Next step this suggests

Class-balanced training, selectable on Batch 1 CV alone and therefore inside the
source-only protocol: class weights in the cross-entropy, or balanced sampling,
or generating more augmented views for the smaller classes. The existing
augmentation machinery already supports per-stage view counts, so the third
option is a small change.

The target for that experiment is concrete: bring Ethylene above zero and recover
part of Ammonia and Toluene on Batches 6–10, which is where essentially all of
the remaining distance to the paper lies.
