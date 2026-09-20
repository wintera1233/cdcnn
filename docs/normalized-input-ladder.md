# Normalized-input ladder (v6.5)

## Question

v6.4 established that per-sample input normalization is worth +0.1087 target
mean, and the v6.3 confound ablation found the CDCNN components neutral to
negative on unnormalized inputs. One explanation would be that augmentation,
feature generation, and contrastive learning need well-conditioned inputs to
help. This ladder tests that directly: the same four ablation steps, each on
per-sample-normalized inputs.

`A3-PS` is exactly the v6.3 A3 stage plus the per-sample input transform.

## Results

Run: `20260920T182057522943Z_cdcnn_v6_5_normalized_ladder_full`, 20 checkpoints, audit `passed`.

| Stage | Batch 1 CV | Target mean | SD | Pooled |
|---|---:|---:|---:|---:|
| `B0-LN-PS` | 0.9933 | 0.5569 | 0.0342 | 0.5507 |
| `B0-stab-PS` | 0.9708 | 0.5613 | 0.0365 | 0.5349 |
| `A2-stab-PS` | 0.9721 | 0.5389 | 0.0309 | 0.5042 |
| `A3-PS` | 0.9708 | 0.5340 | 0.0250 | 0.5061 |

| Step | Isolates | Mean change | SD | Seeds improved |
|---|---|---:|---:|---:|
| `B0-LN-PS` -> `B0-stab-PS` | hard bounds + gradient clipping | +0.0044 | 0.0124 | 4/5 |
| `B0-stab-PS` -> `A2-stab-PS` | A1 augmentation + feature generation + MSE | -0.0224 | 0.0285 | 1/5 |
| `A2-stab-PS` -> `A3-PS` | supervised contrastive loss | -0.0049 | 0.0122 | 2/5 |
| `B0-LN-PS` -> `A3-PS` | every CDCNN component together | -0.0229 | 0.0244 | 2/5 |

## Findings

1. **The CDCNN components still do not help on normalized inputs.** Augmentation
   plus feature generation costs -0.0224 (1/5 seeds improved) and the
   contrastive loss costs -0.0049 (2/5). Together they cost
   -0.0229 against the plain normalized backbone.
2. **Input conditioning was not the blocker.** Comparing identical ablation steps
   with and without per-sample inputs:

| Step | Unnormalized inputs (v6.3 ladder) | Per-sample inputs (v6.5 ladder) |
|---|---:|---:|
| bounds + clipping | -0.0022 | +0.0044 |
| augmentation + feature generation + MSE | -0.0328 | -0.0224 |
| contrastive loss | +0.0029 | -0.0049 |

   The augmentation/feature-generation penalty shrinks but stays clearly
   negative, and the contrastive term moves from marginally positive to
   marginally negative. Neither is rescued by better inputs.
3. **Per-sample normalization helps the full CDCNN model as much as the
   baseline.** `A3-PS` scores 0.5340 against 0.4602 for v6.3 A3, a gain
   of +0.0738 from the input transform alone.
4. **Hard bounds and clipping remain accuracy-neutral** (+0.0044) but still
   cost Batch 1 CV (0.9933 to 0.9708).

## Stage selection

`B0-stab-PS` has the highest target mean (0.5613) but `B0-LN-PS` has the
best Batch 1 CV (0.9933 against 0.9708), and the difference in target
mean (+0.0044) is well inside seed noise. Under the source-only
protocol the selected model is **`B0-LN-PS`**, chosen on CV alone.

## Per-batch accuracy

| Batch | B0-LN-PS | B0-stab-PS | A2-stab-PS | A3-PS |
|---|---:|---:|---:|---:|
| 2 | 0.889 | 0.890 | 0.895 | 0.898 |
| 3 | 0.773 | 0.792 | 0.769 | 0.774 |
| 4 | 0.648 | 0.717 | 0.718 | 0.742 |
| 5 | 0.653 | 0.653 | 0.586 | 0.513 |
| 6 | 0.531 | 0.474 | 0.432 | 0.407 |
| 7 | 0.448 | 0.426 | 0.354 | 0.372 |
| 8 | 0.233 | 0.271 | 0.258 | 0.259 |
| 9 | 0.343 | 0.351 | 0.365 | 0.363 |
| 10 | 0.495 | 0.478 | 0.473 | 0.478 |

The CDCNN components help slightly on Batches 2, 4, and 9 and hurt on 5, 6, and
7; the net is negative.

## Interpretation

Across three experiments the ordering is consistent: normalization changes the
result by 0.07 to 0.15 target mean, while every CDCNN component is worth at most
0.003 and usually less than zero. Under this project's strict source-only
protocol, the paper's mechanism does not reproduce, and the accuracy that does
exist comes from how the inputs and features are normalized.

The remaining candidates for closing the gap to the paper's 0.7230 are the
faithfulness items in `docs/paper-vs-implementation.md`: the augmentation noise
magnitude (Eq. (7) with `perturbation_scale` 1.0 adds noise as large as the
signal), the restyled branch (the paper restyles the residual, this project's
canonical stages restyle the pooled branch), and the contrastive input (Fig. 2
uses the pre-FC128 feature with no learned head).
