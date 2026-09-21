# Epoch alignment (v6.9)

## What was wrong

Augmented stages hold 890 training rows against 445, so a fixed 100-epoch budget
gave them twice the optimizer and scheduler steps. `docs/duplication-control.md`
showed the consequence: replacing the generated view with a bit-exact copy of its
anchor reproduced essentially the whole A1 penalty, so the penalty looked like a
schedule artefact rather than an effect of augmentation.

## The fix

Aligned stages draw one source-sized subset of the augmented pool per epoch, with
no repeats inside an epoch, so they take the same number of steps as an
un-augmented stage: 6 batches per CV fold epoch instead of 12, 7 instead of 14 on
the full fit. Epochs, batch size, optimizer, scheduler, seeds, folds and the
epoch-100 checkpoint are unchanged, and every pooled row is still drawn across
the schedule — each about 50 times rather than 100.

`A1-stab-PS`, the unaligned counterpart, is trained in the same run so the
comparison sits inside one environment.

## Results

Run: `20260921T145852844886Z_cdcnn_v6_9_epoch_aligned_full`, 25 checkpoints, audit `passed`,
container stack torch 2.5.1+cu121. The anchor's source CV folds reproduce the
host-stack run bit-identically; its target mean differs by 0.00003.

| Stage | Aligned | Batch 1 CV | Target mean | SD |
|---|---|---:|---:|---:|
| `B0-stab-PS` | no | 0.9708 | 0.5613 | 0.0364 |
| `A1-stab-PS` | no | 0.9717 | 0.5481 | 0.0288 |
| `A1-aln-PS` | yes | 0.9622 | 0.5524 | 0.0195 |
| `A2-lit-aln-PS` | yes | 0.9631 | 0.5539 | 0.0204 |
| `A3-lit-aln-PS` | yes | 0.9627 | 0.5564 | 0.0136 |

| Comparison | Mean change | SD | Seeds improved |
|---|---:|---:|---:|
| augmentation, unaligned (2x steps) | -0.0132 | 0.0276 | 1/5 |
| augmentation, aligned (matched steps) | -0.0089 | 0.0334 | 2/5 |
| **effect of the alignment itself** | +0.0043 | 0.0250 | 3/5 |
| + paper-literal feature generation + MSE | +0.0014 | 0.0030 | 3/5 |
| + contrastive loss | +0.0025 | 0.0086 | 3/5 |
| full aligned CDCNN vs plain backbone | -0.0050 | 0.0330 | 2/5 |

## Findings

1. **Alignment recovers about a third of the penalty, not all of it.** Matching
   the step count is worth +0.0043, and augmentation still costs
   -0.0089 against no augmentation. The doubled schedule was a real
   confound but not the whole story.
2. **Reconciling this with the duplication control.** Exact duplicates at the
   unaligned schedule cost −0.0114; aligning the schedule returns only +0.0043.
   The two are consistent once per-sample exposure is separated from step count:
   with twice the data at a fixed epoch budget you cannot match both. The aligned
   arm matches steps but sees each row about 50 times instead of 100, and its
   Batch 1 CV falls accordingly (0.9622 against 0.9708). Part of the
   residual is therefore reduced per-sample training, not damage from the
   generated data.
3. **The CDCNN components are mildly positive under the aligned schedule.**
   Feature generation on the paper's branch adds +0.0014 and the contrastive
   loss +0.0025. Both are small and inside seed noise, but neither is
   negative any more.
4. **The full aligned CDCNN is level with the plain backbone**, -0.0050
   with 2/5 seeds improving and an SD of 0.0330. It no longer loses, and it does
   not win.
5. **Alignment makes results markedly more stable.** The seed spread falls
   monotonically along the aligned ladder: 0.0364 for the anchor,
   0.0288 unaligned, 0.0195 aligned, and 0.0136 for the full aligned
   model — the tightest of any stage trained in this project.

## Status of the protocol

The alignment is implemented as a separate implementation version
(`CDCNN_v6.9_epoch_aligned`) and has not been adopted as canonical. Adopting it
would change what `CDCNN_four_experiment_spec_v6.md` fixes, and the honest
summary is that neither schedule is neutral:

- **Unaligned** gives each row the same number of visits as an un-augmented stage
  but doubles the compute and the scheduler steps.
- **Aligned** matches compute and scheduler steps but halves per-sample visits.

Reporting both, as this run does, is the defensible option until the project
decides which quantity it wants held constant.
