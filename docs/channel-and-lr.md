# Channel widths and learning rate (v7.1)

Run: `runs/20260922T182154907065Z_channel_restore_full`. Eighteen checkpoints (six cells x three seeds), frozen and hashed
before any target file was opened; `leakage_target_access_audit.json` is
`passed`. Flatten head and per-sample inputs throughout, `L_ce` only,
100 epochs, SGD(momentum 0.9, weight decay 1e-4), StepLR(25, 0.5), batch 64.

## The grid

| Cell | Backbone | lr | Batch 1 | Target mean | SD |
|---|---|---:|---:|---:|---:|
| `R-txt-ps@lr0.001` | capped 128, 336,416 | 0.001 | 0.9813 | 0.4673 | 0.0443 |
| `R-txt-ps@lr0.0003` | capped 128 | 0.0003 | 0.9700 | 0.4928 | 0.0062 |
| `R-txt-ps@lr0.0001` | capped 128 | 0.0001 | 0.9303 | 0.4621 | 0.0233 |
| `R-fig-ps@lr0.001` | Fig. 2, 1,058,080 | 0.001 | 0.9813 | 0.4770 | 0.0362 |
| **`R-fig-ps@lr0.0003`** | Fig. 2 | 0.0003 | 0.9633 | **0.5222** | 0.0277 |
| `R-fig-ps@lr0.0001` | Fig. 2 | 0.0001 | 0.9318 | 0.4724 | 0.0183 |

`R-txt-ps@lr0.001` re-runs a v7.0 cell and returns **0.4673, the same value to
four decimals**. Training is reproducible across runs, so every comparison below
is exact.

## Effects, and why none of them can be claimed

Pooled within-cell SD 0.0287; the standard error of a three-against-three
difference is 0.0235, so a difference is separable above 0.0469.

| Effect | Value | Separable |
|---|---:|---|
| channel: capped 128 -> Fig. 2, at lr 0.001 | +0.0098 | no |
| channel, at lr 0.0003 | +0.0294 | no |
| channel, at lr 0.0001 | +0.0103 | no |
| lr 0.001 -> 0.0003, capped 128 | +0.0255 | no |
| lr 0.001 -> 0.0003, Fig. 2 | +0.0452 | no |
| lr 0.001 -> 0.0001, either backbone | -0.005 | no |
| main effect, channel | +0.0165 | no |

**Both factors point the way the evidence predicted and neither reaches
separability.** Restoring Fig. 2's widths helps at all three learning rates, and
lr 0.0003 beats both 0.001 and 0.0001 on both backbones, so the rate is
non-monotonic with an interior optimum. But the largest single effect, +0.0452,
sits below the 0.0469 threshold this project declared in advance.

Separating an effect of the observed size would take about eight seeds per cell,
not the five that `proposal.md` holds in reserve. At roughly ten seconds a run
that is affordable, but see the next section for why it is not the best use of
the next run.

The escalation rule fired: four of six cells exceed the 0.02 standard-deviation
threshold.

## Where the remaining gap actually is

The best cell against the paper's ResNet, per batch:

| Batch | `R-fig-ps@lr0.0003` | paper | gap |
|---|---:|---:|---:|
| 2 | 0.827 | 0.769 | **+0.058** |
| 3 | 0.729 | 0.662 | **+0.067** |
| 4 | 0.673 | 0.642 | **+0.030** |
| 6 | 0.688 | 0.725 | -0.038 |
| 7 | 0.466 | 0.502 | -0.036 |
| 10 | 0.390 | 0.430 | -0.040 |
| 5 | 0.501 | 0.716 | **-0.215** |
| 9 | 0.255 | 0.604 | **-0.349** |
| 8 | 0.171 | 0.660 | **-0.489** |

**Six of the nine batches are now at or above the paper.** This project has
never been in that position. The entire deficit of 0.112 in the mean comes from
three batches, and those three have a single shared cause.

## Two dead classes

Per-class recall of the best cell, summed over three seeds:

| Batch | Acetone | Acetaldehyde | Ethanol | Ethylene | Ammonia | Toluene |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.921 | 0.921 | 0.333 | **0.006** | 0.998 | 1.000 |
| 3 | 0.986 | 0.989 | 0.261 | **0.000** | 0.930 | - |
| 4 | 1.000 | 0.605 | 0.528 | **0.000** | 1.000 | - |
| 5 | 1.000 | 0.375 | 0.150 | **0.000** | 0.836 | - |
| 6 | 0.972 | 0.802 | 0.655 | **0.000** | 0.905 | 0.002 |
| 7 | 0.221 | 0.923 | 0.772 | **0.000** | **0.175** | 0.952 |
| 8 | 0.011 | 0.956 | 0.275 | **0.000** | **0.000** | 0.574 |
| 9 | 1.000 | 1.000 | 0.027 | **0.000** | **0.000** | 0.010 |
| 10 | 0.576 | 0.732 | 0.828 | **0.000** | **0.000** | 0.207 |
| **all** | 0.668 | 0.858 | 0.616 | **0.000** | 0.514 | 0.388 |

**Ethylene is never predicted correctly, in any batch.** Across 5,718 true
Ethylene rows the recall is zero, and the model emits an Ethylene prediction only
672 times in total. It is the smallest class in Batch 1 at 30 rows of 445.

**Ammonia collapses after Batch 6** and its samples go to Toluene. In Batch 8,
417 of 429 Ammonia rows are predicted Toluene; Ammonia is 49 % of that batch, so
that one confusion caps B8 near 0.51 before anything else goes wrong.

### Headroom

Recomputing each batch's accuracy with those classes corrected, holding
everything else fixed:

| | Target mean |
|---|---:|
| as measured | 0.5222 |
| with Ethylene recovered | **0.6683** |
| with Ethylene and Ammonia recovered | **0.7854** |
| paper ResNet | 0.6344 |
| paper CDCNN | 0.7230 |

Recovering Ethylene alone is worth **+0.146** and would put this baseline past
the paper's ResNet. Recovering both would put it past the paper's CDCNN.

For comparison, everything measured in v7.0 and v7.1 together - head, input
normalisation, channel widths, learning rate - spans about 0.23, and the best
single lever among them is worth 0.093.

## Conclusion

The architecture and optimizer questions are settled enough. Both remaining
architectural effects are real in direction, small in size, and would need eight
seeds each to prove; the dead-class problem is an order of magnitude larger and
has never been attacked. The previous project found the same zero-recall
Ethylene and recorded it as an open question without acting on it.

Next work should target the two dead classes, not the backbone.
