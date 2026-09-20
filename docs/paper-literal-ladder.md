# Paper-literal and augmentation ladder (v6.6)

## Questions

1. **What does augmentation do on its own?** Every previous ladder bundled A1
   augmentation with feature generation, so its individual effect was unmeasured.
2. **Does the paper's own restyled branch behave differently?** Eqs. (8)-(9) name
   the pooled/upsampled branch H and the residual L, and Eq. (15) restyles L.
   This project's canonical stages restyle the pooled branch instead, calling it
   the low-frequency-like style component. No five-seed result existed for the
   paper's assignment.

Both are measured on per-sample-normalized inputs, the best configuration found
in v6.4/v6.5.

## Results

Run: `20260920T191634939156Z_cdcnn_v6_6_paper_literal_full`, 20 checkpoints, audit `passed`.

| Stage | Batch 1 CV | Target mean | SD | Pooled |
|---|---:|---:|---:|---:|
| `B0-stab-PS` | 0.9708 | 0.5613 | 0.0365 | 0.5349 |
| `A1-stab-PS` | 0.9717 | 0.5482 | 0.0287 | 0.5167 |
| `A2-lit-PS` | 0.9712 | 0.5526 | 0.0270 | 0.5234 |
| `A3-lit-PS` | 0.9712 | 0.5542 | 0.0250 | 0.5238 |

| Step | Isolates | Mean change | SD | Seeds improved |
|---|---|---:|---:|---:|
| `B0-stab-PS` -> `A1-stab-PS` | A1 augmentation alone | -0.0131 | 0.0274 | 1/5 |
| `A1-stab-PS` -> `A2-lit-PS` | paper-literal feature generation + MSE | +0.0044 | 0.0034 | 4/5 |
| `A2-lit-PS` -> `A3-lit-PS` | supervised contrastive loss | +0.0016 | 0.0143 | 3/5 |
| `B0-stab-PS` -> `A3-lit-PS` | all three together | -0.0071 | 0.0377 | 3/5 |

## Findings

1. **Augmentation is the component that hurts: -0.0131, improving on
   1/5 seeds.** With `perturbation_scale = 1.0` the added noise has roughly the
   same variance as the standardized signal, so each generated view is about as
   much noise as data. This is the canonical setting because the paper's Eq. (7)
   introduces no multiplier.
2. **The paper's restyled branch is better than this project's semantic choice,
   on every seed.** Comparing identical configurations that differ only in which
   branch is restyled:

| Comparison | Mean change | SD | Seeds improved |
|---|---:|---:|---:|
| `A2-lit-PS` vs `A2-stab-PS` (v6.5) | +0.0137 | 0.0063 | 5/5 |
| `A3-lit-PS` vs `A3-PS` (v6.5) | +0.0201 | 0.0110 | 5/5 |

   Restyling the residual — the paper's L, this project's high-frequency-like
   detail component — is worth about +0.014 to +0.020 target mean over restyling
   the pooled branch. The physical-semantic renaming adopted in the v6
   specification therefore costs accuracy relative to the paper's literal
   procedure.
3. **Feature generation becomes mildly positive once the paper's branch is
   used**: +0.0044 on 4/5 seeds, against -0.0224 for the semantic branch
   bundled with augmentation in v6.5.
4. **The contrastive loss stays neutral**: +0.0016 on 3/5 seeds, consistent
   with +0.0029 (v6.3) and -0.0049 (v6.5).
5. **The full paper-literal model still does not beat the plain backbone**
   (-0.0071), because augmentation's penalty outweighs the gains from
   the other two components.

## Per-batch accuracy

| Batch | B0-stab-PS | A1-stab-PS | A2-lit-PS | A3-lit-PS |
|---|---:|---:|---:|---:|
| 2 | 0.890 | 0.895 | 0.896 | 0.898 |
| 3 | 0.792 | 0.792 | 0.792 | 0.802 |
| 4 | 0.717 | 0.720 | 0.725 | 0.750 |
| 5 | 0.653 | 0.605 | 0.609 | 0.587 |
| 6 | 0.474 | 0.447 | 0.456 | 0.432 |
| 7 | 0.426 | 0.368 | 0.381 | 0.396 |
| 8 | 0.271 | 0.259 | 0.259 | 0.271 |
| 9 | 0.351 | 0.362 | 0.364 | 0.364 |
| 10 | 0.478 | 0.486 | 0.491 | 0.487 |

## Determinism evidence

`B0-stab-PS` was trained in both the v6.5 and v6.6 runs, under different
configurations and separate launches. All five seeds reproduced **bit-identically**
(target mean 0.561300 in both), which confirms that stage comparisons across
runs in this environment are exact, and that the config-driven stage machinery
does not perturb training.

## Next

Augmentation is now the only clearly harmful component. Its noise magnitude is
the obvious suspect, and the specification already predeclares a different
`perturbation_scale` as a separate sensitivity experiment.
