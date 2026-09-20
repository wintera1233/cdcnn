# Augmentation scale sensitivity (v6.7)

## Question

v6.6 isolated A1 augmentation and found it costs -0.0131 target mean, improving on
1/5 seeds. With the canonical `perturbation_scale = 1.0` the added noise has
roughly the variance of the standardized signal, so each generated view is about
as much noise as data. Does a smaller noise magnitude make augmentation helpful?

The paper's Eq. (7) has no multiplier, so 1.0 remains canonical and every other
value is a declared sensitivity experiment.

## Results

Run: `20260920T201524271480Z_cdcnn_v6_7_augmentation_scale_full`, 15 checkpoints, audit `passed`. The scale-1.0 and
no-augmentation rows come from `20260920T191634939156Z_cdcnn_v6_6_paper_literal_full`, whose
stages reproduce bit-identically across runs.

| Stage | Noise scale | Batch 1 CV | Target mean | vs no augmentation | Seeds improved |
|---|---:|---:|---:|---:|---:|
| `A1-PS-s05` | 0.05 | 0.9784 | 0.5508 | -0.0105 | 1/5 |
| `A1-PS-s20` | 0.20 | 0.9739 | 0.5514 | -0.0099 | 1/5 |
| `A1-PS-s50` | 0.50 | 0.9712 | 0.5486 | -0.0127 | 1/5 |
| `A1-stab-PS` (v6.6) | 1.00 | 0.9717 | 0.5482 | -0.0131 | 1/5 |
| `B0-stab-PS` (v6.6) | none | 0.9708 | 0.5613 | — | — |

## Findings

1. **Reducing the noise does not rescue augmentation.** Every scale lands within
   0.0028 target mean of the others, and all remain about 0.010 to 0.013
   below the un-augmented backbone. Each improves on only 1/5 seeds.
2. **The magnitude barely matters at all.** Going from scale 1.0 to 0.05, a
   twentyfold reduction, changes the target mean by +0.0026. Whatever
   augmentation does to this model, it is not dominated by how much noise it adds.
3. **Batch 1 CV rises as the noise falls** (0.9712 at 0.5 to
   0.9784 at 0.05), approaching the un-augmented 0.9708, exactly as
   expected when the generated views converge on copies of their anchors. Target
   accuracy does not follow.

## Interpretation and the next control

At scale 0.05 each generated view is nearly a copy of its anchor, yet the target
penalty persists at -0.0105. That points away from the noise and towards the
augmentation procedure itself: augmented stages train on 890 rows instead of 445,
so under the fixed 100-epoch schedule they take twice as many optimizer steps as
un-augmented stages, with a duplicated sample in every batch.

The clean control is `perturbation_scale = 0.0`: exact duplicates of the source
rows, no noise at all. If that reproduces the same penalty, the cost is the
doubled dataset and step count rather than the augmentation, and the entire A1
comparison in this project has been confounded by schedule length rather than
measuring augmentation.
