# Contrastive placement (v6.10)

## Question

Fig. 2 feeds `z_f` and its generated twin — the ResNet-5 output **before** FC128 —
into the contrast loss, and supplement S1 describes `P` only as a function
mapping features to the unit sphere. This project instead takes the post-FC128
128-d feature and passes it through a learned `Linear(128, 128)` before
normalizing. That was the last untested structural difference from the paper,
and the contrastive loss had measured close to zero in every ladder so far
(+0.0029 in v6.3, −0.0049 in v6.5, +0.0016 in v6.6, +0.0025 in v6.9).

`A3-zf-aln-PS` applies the loss where the paper does: to the 16384-d pre-FC128
latent, bounded and then L2-normalized, with no learned head. Everything else —
the paper's residual restyling branch, the aligned epoch budget, per-sample
inputs, temperature, weights — is identical to `A3-lit-aln-PS`, and because the
projection head is constructed last, both stages start from bit-identical
backbone weights for a given seed.

## Results

Run: `20260921T155203687340Z_cdcnn_v6_10_contrastive_placement_full`, 15 checkpoints, audit `passed`.

| Stage | Batch 1 CV | Target mean | SD | Pooled |
|---|---:|---:|---:|---:|
| `A2-lit-aln-PS` | 0.9631 | 0.5539 | 0.0204 | 0.5289 |
| `A3-lit-aln-PS` | 0.9627 | 0.5564 | 0.0136 | 0.5346 |
| `A3-zf-aln-PS` | 0.9658 | 0.5541 | 0.0189 | 0.5282 |

| Comparison | Mean change | SD | Seeds improved |
|---|---:|---:|---:|
| contrastive at this project's placement | +0.0025 | 0.0086 | 3/5 |
| contrastive at the paper's placement | +0.0002 | 0.0056 | 2/5 |
| paper placement versus this project's | -0.0023 | 0.0054 | 1/5 |

## Findings

1. **The placement was not the explanation.** Moving the loss to the paper's
   position leaves it worth +0.0002 against the same model without it,
   improving on 2/5 seeds. The project's own placement is worth +0.0025.
   Both are indistinguishable from zero at this seed spread.
2. **Neither placement is better than the other**: -0.0023 for the paper's
   position, 1/5 seeds. The learned head is not what was holding the term back.
3. **Five independent measurements now agree.** Across v6.3, v6.5, v6.6, v6.9 and
   v6.10 — different backbones, branch choices, input handling, schedules and now
   placements — the supervised contrastive term has measured between −0.005 and
   +0.003 target mean. Under this project's strict source-only protocol its
   effect is not merely small, it is consistently indistinguishable from zero.
4. **A plausible reason.** Batch 1 features are already close to separable:
   source CV runs 0.963–0.966, and the canonical stages reach 0.99. A
   supervised contrastive term pulls same-label samples together and pushes
   different labels apart; when the encoder already does that on the source
   domain, the term has little left to change, and nothing in it is informed by
   the target domains it would need to help.

## Determinism

`A2-lit-aln-PS` and `A3-lit-aln-PS` were retrained here as in-run controls and
reproduced their v6.9 target means to all printed digits
(0.5538556980 and 0.5563593135), across separate container launches.

## What remains

The structural faithfulness items from `docs/paper-vs-implementation.md` are now
closed except the small numerical ones (Gaussian sigma sampling, the
variance-versus-standard-deviation denominator, the 512-wide inner convolution).
The untested lever with the most room is the contrastive hyperparameters
themselves: temperature 0.07, weight 0.5 and the 128-d head are all
project-controlled, none is given by the paper, and none has ever been tuned —
which can be done on Batch 1 CV alone without touching target data.
