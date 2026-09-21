# All results: accuracy on Batches 2–10

Every approach trained in this project that reached a completed, audited target
evaluation. Values are five-seed means of the per-batch accuracy; "mean" is the
unweighted mean over the nine target batches, the same statistic the paper
reports. Sorted by that mean.

Where a stage was trained in more than one run, the first completed evaluation is
listed; repeats inside one software environment reproduce exactly.

| Approach | What it changes | Batch 1 CV | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | Target mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **paper ResNet** | paper reference | — | 0.769 | 0.662 | 0.642 | 0.716 | 0.725 | 0.502 | 0.660 | 0.604 | 0.430 | **0.6346** |
| **paper CDWC** | paper reference | — | 0.760 | 0.772 | 0.705 | 0.789 | 0.851 | 0.578 | 0.582 | 0.568 | 0.429 | **0.6705** |
| **paper CDCNN** | paper reference | — | 0.827 | 0.768 | 0.724 | 0.789 | 0.932 | 0.618 | 0.715 | 0.566 | 0.566 | **0.7230** |
| `B0-stab-PS` | B0-LN-PS + bounds and clipping | 0.9708 | 0.890 | 0.792 | 0.717 | 0.653 | 0.474 | 0.426 | 0.271 | 0.351 | 0.478 | 0.5613 |
| `B0-LN-PS` | LayerNorm + per-sample inputs | 0.9933 | 0.889 | 0.773 | 0.648 | 0.653 | 0.531 | 0.448 | 0.233 | 0.343 | 0.495 | 0.5569 |
| `A3-lit-aln-PS` | + contrastive loss, aligned | 0.9627 | 0.896 | 0.804 | 0.745 | 0.554 | 0.463 | 0.426 | 0.273 | 0.367 | 0.478 | 0.5564 |
| `A3-lit-PS` | + contrastive loss, paper's branch | 0.9712 | 0.898 | 0.802 | 0.750 | 0.587 | 0.432 | 0.396 | 0.271 | 0.364 | 0.487 | 0.5542 |
| `A3-zf-aln-PS` | + contrastive at the paper's placement | 0.9658 | 0.893 | 0.799 | 0.716 | 0.581 | 0.459 | 0.412 | 0.288 | 0.366 | 0.473 | 0.5541 |
| `A2-lit-aln-PS` | + feature generation, aligned | 0.9631 | 0.892 | 0.801 | 0.720 | 0.584 | 0.454 | 0.413 | 0.277 | 0.366 | 0.479 | 0.5539 |
| `A2-lit-PS` | + feature generation, paper's branch | 0.9712 | 0.896 | 0.792 | 0.725 | 0.609 | 0.456 | 0.381 | 0.259 | 0.364 | 0.491 | 0.5526 |
| `A1-aln-PS` | augmentation, aligned epoch budget | 0.9622 | 0.890 | 0.800 | 0.720 | 0.585 | 0.444 | 0.410 | 0.280 | 0.366 | 0.477 | 0.5524 |
| `A1-PS-s20` | augmentation at noise 0.2 | 0.9739 | 0.891 | 0.770 | 0.670 | 0.645 | 0.498 | 0.397 | 0.244 | 0.356 | 0.493 | 0.5514 |
| `A1-PS-s05` | augmentation at noise 0.05 | 0.9784 | 0.891 | 0.768 | 0.668 | 0.652 | 0.498 | 0.407 | 0.235 | 0.346 | 0.491 | 0.5508 |
| `A1-PS-s00` | duplication control, noise 0 | 0.9789 | 0.892 | 0.768 | 0.668 | 0.651 | 0.497 | 0.405 | 0.233 | 0.345 | 0.490 | 0.5499 |
| `A1-PS-s50` | augmentation at noise 0.5 | 0.9712 | 0.895 | 0.777 | 0.688 | 0.640 | 0.469 | 0.366 | 0.254 | 0.362 | 0.487 | 0.5486 |
| `A1-stab-PS` | + augmentation only | 0.9717 | 0.895 | 0.792 | 0.720 | 0.605 | 0.447 | 0.368 | 0.259 | 0.362 | 0.486 | 0.5482 |
| `A2-stab-PS` | + feature generation (pooled branch) | 0.9721 | 0.895 | 0.769 | 0.718 | 0.586 | 0.432 | 0.354 | 0.258 | 0.365 | 0.473 | 0.5389 |
| `A3-PS` | + contrastive loss, per-sample inputs | 0.9708 | 0.898 | 0.774 | 0.742 | 0.513 | 0.407 | 0.372 | 0.259 | 0.363 | 0.478 | 0.5340 |
| `B0-PS` | B0 + per-sample inputs | 0.9856 | 0.860 | 0.646 | 0.522 | 0.601 | 0.512 | 0.413 | 0.258 | 0.344 | 0.492 | 0.5164 |
| `B0-LN-LOG` | LayerNorm + signed-log inputs | 0.9874 | 0.889 | 0.707 | 0.525 | 0.545 | 0.459 | 0.353 | 0.163 | 0.343 | 0.457 | 0.4935 |
| `B0-LN` | B0 with LayerNorm | 0.9910 | 0.873 | 0.692 | 0.512 | 0.458 | 0.487 | 0.374 | 0.193 | 0.344 | 0.497 | 0.4922 |
| `B0-stab` | B0-LN + bounds and clipping | 0.9654 | 0.866 | 0.668 | 0.520 | 0.397 | 0.443 | 0.384 | 0.265 | 0.377 | 0.489 | 0.4901 |
| `B0-LN-CLIP` | LayerNorm + clipped inputs | 0.9910 | 0.878 | 0.690 | 0.503 | 0.439 | 0.483 | 0.371 | 0.185 | 0.343 | 0.472 | 0.4849 |
| `A3` | + contrastive loss (v6.3 CDCNN) | 0.9676 | 0.893 | 0.683 | 0.576 | 0.349 | 0.415 | 0.330 | 0.172 | 0.303 | 0.419 | 0.4600 |
| `A2-stab` | A2 + stability package | 0.9672 | 0.886 | 0.653 | 0.553 | 0.347 | 0.435 | 0.332 | 0.170 | 0.322 | 0.417 | 0.4573 |
| `B0` | canonical baseline (BatchNorm, scaler inputs) | 0.9779 | 0.741 | 0.509 | 0.497 | 0.451 | 0.476 | 0.309 | 0.201 | 0.212 | 0.291 | 0.4097 |
| `A2-semantic` | + feature generation, pooled branch | 0.9824 | 0.767 | 0.509 | 0.494 | 0.359 | 0.415 | 0.285 | 0.171 | 0.261 | 0.309 | 0.3968 |
| `A1` | + VAE-style augmentation | 0.9797 | 0.732 | 0.498 | 0.487 | 0.330 | 0.418 | 0.273 | 0.126 | 0.182 | 0.304 | 0.3721 |

## Reading the table

**The project's best is `B0-stab-PS` at 0.5613**, against the paper's plain ResNet
baseline at 0.6344 and its CDCNN at 0.7230. The original canonical stages sit at
the bottom: B0 0.4097, A1 0.3721, A2-semantic 0.3968, A3 0.4600.

**Where this project already matches or beats the paper**: B2, B3.
On Batch 2 the best stages reach 0.89–0.90 against the paper's CDCNN 0.827, and
on Batch 3 0.80 against 0.768 — with no CDCNN component involved.

**Where the deficit is concentrated**, best stage against paper CDCNN:

| Batch | paper CDCNN | this project | gap |
|---|---:|---:|---:|
| B6 | 0.932 | 0.474 | -0.458 |
| B8 | 0.715 | 0.271 | -0.444 |
| B9 | 0.566 | 0.351 | -0.215 |
| B7 | 0.618 | 0.426 | -0.192 |
| B5 | 0.789 | 0.653 | -0.136 |

Batches 6 and 8 account for most of the remaining distance. Those are also the
two batches where the paper's own CDCNN scores unusually high (0.932 and 0.715,
above its Batch 3 and 4 results), so the gap is not simply "later batches are
harder": something about those two batches is handled well there and not here.

## What moved the number, and what did not

| Change | Effect on target mean | Evidence |
|---|---|---|
| Per-sample input normalization | **+0.109** | `docs/input-normalization.md` |
| LayerNorm instead of BatchNorm | **+0.085** | `docs/a3-confound-ablation.md` |
| Both together | **+0.149** | `docs/input-normalization.md` |
| Epoch alignment for augmented stages | +0.004 | `docs/epoch-alignment.md` |
| Feature generation, paper's branch | +0.001 to +0.014 | `docs/paper-literal-ladder.md` |
| Supervised contrastive loss | −0.005 to +0.003 | `docs/contrastive-placement.md` |
| Hard bounds and gradient clipping | −0.002 to +0.004 | `docs/a3-confound-ablation.md` |
| Augmentation (any noise scale) | −0.009 to −0.013 | `docs/augmentation-scale.md` |
| Signed-log or clipped inputs | −0.007 to +0.001 | `docs/input-normalization.md` |

Normalization accounts for essentially all of the improvement; the three CDCNN
components together are worth approximately zero under this protocol.

## Environment note

Rows come from two software stacks: torch 2.5.1+cu121 (the 2026-09-11 canonical
run and everything from 2026-09-21 onwards, in the pinned container) and torch
2.8.0+cu126 (the host runs of 2026-09-15 to 2026-09-20). Stages trained under
both agree closely but not always exactly: source CV folds reproduce identically,
while target means can differ in the fourth decimal (`B0-stab-PS` 0.561300
against 0.561334; `B0` 0.409667 against 0.407677). Comparisons inside one run are
exact; cross-run differences below about 0.004 should not be read as effects.
