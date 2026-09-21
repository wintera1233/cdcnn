# Development journal

How the CDCNN reproduction went, in order, and what each step established. The
consolidated table of every approach is in `docs/all-results.md`; this file is
the narrative.

---

## Part 1 — The canonical four-stage result

Run `20260911T092326995583Z_cdcnn_v6_3_full`, five seeds
(1042, 2024, 3407, 42, 123), torch 2.5.1+cu121, audit `passed`. Batch 1 is the
only pre-freeze data; Batches 2–10 were opened once, after all 20 checkpoints
were frozen. Target mean is the unweighted mean of the nine batch accuracies, the
same statistic the paper reports.

| Stage | Batch 1 CV | Target mean | SD | Pooled | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | Paper |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** ResNet baseline | 0.9779 | 0.4097 | 0.0213 | 0.3949 | 0.741 | 0.509 | 0.497 | 0.451 | 0.476 | 0.309 | 0.201 | 0.212 | 0.291 | 0.6344 |
| **A1** + input augmentation | 0.9797 | 0.3721 | 0.0190 | 0.3718 | 0.732 | 0.498 | 0.487 | 0.330 | 0.418 | 0.273 | 0.126 | 0.182 | 0.304 | — |
| **A2-semantic** + feature generation | 0.9824 | 0.3968 | 0.0121 | 0.3849 | 0.767 | 0.509 | 0.494 | 0.359 | 0.415 | 0.285 | 0.171 | 0.261 | 0.309 | 0.6705 |
| **A3** + contrastive (CDCNN) | 0.9676 | 0.4600 | 0.0254 | 0.4607 | 0.893 | 0.683 | 0.576 | 0.349 | 0.415 | 0.330 | 0.172 | 0.303 | 0.419 | 0.7230 |

Per seed (target mean):

| Stage | 1042 | 2024 | 3407 | 42 | 123 |
|---|---:|---:|---:|---:|---:|
| `B0` | 0.3790 | 0.4253 | 0.4190 | 0.3964 | 0.4286 |
| `A1` | 0.3879 | 0.3623 | 0.3563 | 0.3969 | 0.3569 |
| `A2-semantic` | 0.3846 | 0.3891 | 0.4070 | 0.4124 | 0.3911 |
| `A3` | 0.4534 | 0.4572 | 0.4285 | 0.4618 | 0.4993 |

**What this says.** The ordering B0 → A1 → A2 → A3 does not reproduce the
paper's. A1 and A2-semantic are *below* the plain baseline, and while A3 is the
best of the four, all four sit far below the paper's own ResNet baseline
(0.6344), let alone its CDCNN (0.7230). Every stage is near-perfect on Batch 1
(CV 0.97–0.98) and collapses on the drifted batches — the gap is generalization,
not fitting.

---

## Part 2 — The iterations

### 1. Where does A3's advantage come from? (v6.3 confound ablation)

A3 beat A2-semantic by +0.067, which looked like evidence for contrastive
learning. But v6.3 had given A3 four things at once: `LayerNorm` instead of
`BatchNorm`, ±20 hard bounds, gradient clipping, and the contrastive loss.
Stages `B0-LN`, `B0-stab` and `A2-stab` separated them.

**Finding: LayerNorm is the whole effect.** +0.0845 target mean on 5/5 seeds,
against +0.0029 for the contrastive loss. Bounds and clipping were
accuracy-neutral. `docs/a3-confound-ablation.md`

### 2. Reading the paper properly

With the mechanism in doubt, the paper and supplement were read end to end
against the code.

**Finding: the paper's plain ResNet (0.6344) beats every stage this project had
trained.** The shortfall therefore sat upstream of the CDCNN components. Three
undeclared differences also surfaced — the contrastive loss belongs before
FC128 with no learned head, Eq. (16) divides by a variance, block 5 has a
512-wide inner convolution — plus an internal contradiction in the paper about
the statistics shape. `docs/paper-vs-implementation.md`

### 3. Input handling (v6.4)

Under the Batch-1 `StandardScaler`, Batch 2 reaches |z| ≈ 1.5 × 10⁴. Four
parameter-free transforms were tested: none, per-sample standardization,
signed-log, clipping.

**Finding: per-sample input normalization is the single biggest lever, +0.1087
on 5/5 seeds**, and it composes with LayerNorm for +0.1492. Signed-log and
clipping do nothing, so the mechanism is removing each sample's own offset and
gain — not taming outliers. Best stage went 0.4922 → 0.5569.
`docs/input-normalization.md`

### 4. Do the CDCNN parts help once inputs are fixed? (v6.5)

The v6.3 ladder was repeated on per-sample-normalized inputs.

**Finding: no.** Augmentation plus feature generation −0.0224, contrastive
−0.0049. Poor input conditioning was not what had been holding them back.
Per-sample inputs did lift the full model as much as the baseline (A3 0.4602 →
A3-PS 0.5340). `docs/normalized-input-ladder.md`

### 5. The paper's own branch, and augmentation alone (v6.6)

Eqs. (8)–(9) name the pooled branch H and the residual L, and Eq. (15) restyles
**L**. This project's canonical stages restyle the pooled branch instead. A1 had
also never been measured apart from feature generation.

**Finding: the paper's branch is better on 5/5 seeds** (+0.0137 at A2, +0.0201
at A3), so the physical-semantic renaming in the specification costs accuracy.
And **augmentation alone is the component that hurts**: −0.0131.
`docs/paper-literal-ladder.md`

### 6. Is it the noise? (v6.7)

`perturbation_scale` was wired through and swept: 0.5, 0.2, 0.05 against the
canonical 1.0.

**Finding: the magnitude barely matters.** A twentyfold noise reduction changed
the target mean by +0.0026; every scale stayed ~0.01 below no augmentation.
`docs/augmentation-scale.md`

### 7. Then what is it? (v6.8 duplication control)

Scale 0.0 makes the "generated" view a bit-exact copy: no noise, no new
information, only a doubled training set.

**Finding, and the session's most consequential one: the penalty is a schedule
artefact.** Exact duplicates cost −0.0114 against −0.0131 for real augmentation.
Augmented stages hold 890 rows against 445, so at a fixed 100 epochs they take
twice the optimizer and scheduler steps. Every A1-versus-B0 comparison in this
project — including the canonical −0.0376 — had been measuring augmentation
*plus* a doubled schedule. `docs/duplication-control.md`

### 8. Aligning the schedule (v6.9)

Aligned stages draw one source-sized subset of their pool per epoch, so they
take the same number of steps as an un-augmented stage. Epochs, batch size,
optimizer, scheduler, seeds and folds are untouched.

**Finding: alignment recovers about a third of the penalty (+0.0043), not all of
it.** The two are consistent once per-sample exposure is separated from step
count: with twice the data at a fixed epoch budget you cannot match both, and
the aligned arm sees each row ~50 times instead of 100. Under alignment the
CDCNN components stop being negative, and the seed spread falls from 0.0364 to
0.0136 — the most stable stage trained here. `docs/epoch-alignment.md`

This run also hit an **Xid 31 GPU fault** that poisoned CUDA machine-wide. Cause
was most likely torch 2.8.0+cu126 against a CUDA 12.2 driver; training now runs
in a container pinned to torch 2.5.1+cu121. `docs/gpu-fault-20260921.md`

### 9. The last structural difference (v6.10)

Fig. 2 applies the contrastive loss to the pre-FC128 latent with no learned
head; this project used a learned head after FC128.

**Finding: the placement was not the explanation.** At the paper's placement the
term is worth +0.0002; at this project's, +0.0025. Five independent measurements
across v6.3, v6.5, v6.6, v6.9 and v6.10 now put the contrastive effect between
−0.005 and +0.003. `docs/contrastive-placement.md`

---

## Part 3 — Where it ended

| | Target mean |
|---|---:|
| Canonical A3 (start) | 0.4600 |
| **Best stage now (`B0-stab-PS`)** | **0.5613** |
| Paper ResNet baseline | 0.6344 |
| Paper CDCNN | 0.7230 |

| Change | Worth |
|---|---:|
| Per-sample input normalization | +0.109 |
| LayerNorm instead of BatchNorm | +0.085 |
| Epoch alignment | +0.004 |
| Feature generation, paper's branch | +0.001 to +0.014 |
| Contrastive loss | −0.005 to +0.003 |
| Augmentation (any noise scale) | −0.009 to −0.013 |

**The conclusion of the reproduction so far**: under this project's strict
source-only protocol, normalization accounts for essentially all of the
improvement, and the three CDCNN components together are worth approximately
zero. On Batches 2 and 3 the plain normalized backbone already matches or beats
the paper's CDCNN; the remaining deficit is concentrated in Batches 6 and 8.

---

## Part 4 — Method notes

What kept the results trustworthy, and what went wrong.

- **Every stage change was verified not to disturb existing stages.** Before each
  launch, all pre-existing stages were trained under both the old and new code
  and required to produce bit-identical losses and predictions — 25 stages by
  the end. This caught nothing, which is the point: it made "only the intended
  thing changed" an observation rather than a claim.
- **Configurations are immutable artefacts.** Each pins the stage lists it was
  written with as frozen literals, so adding a stage later cannot silently change
  what an older configuration must contain. The validator twice refused an edit
  that would have altered a config a completed run depended on.
- **Determinism holds within one environment**, exactly, across separate runs and
  container launches. Across torch versions, source CV folds still match but
  target means can move in the fourth decimal, so cross-stack differences below
  about 0.004 are environment, not effect.
- **Failures that cost time**: a controller killed at 23/30 when a session ended
  (resumed from frozen checkpoints); a report generator that assumed a config
  block and crashed after 30 checkpoints had been evaluated (fixed, regression
  test added for every config); an unquoted heredoc that let the shell mangle a
  documentation edit; and the GPU fault above.
- **Protocol boundaries were never crossed.** Every run loaded Batch 1 only until
  all checkpoints were frozen, opened Batches 2–10 once afterwards, and passed
  its own leakage audit. No target metric influenced any hyperparameter,
  augmentation setting, checkpoint or stage selection; where selection was
  needed, Batch 1 CV chose.

---

## Part 5 — Open questions

1. **Batches 6 and 8** account for most of the remaining gap (0.47 against 0.93,
   0.27 against 0.72), and are where the paper scores unusually well. Their
   confusion matrices are the next thing to look at, not more global tuning.
2. **Contrastive hyperparameters were never tuned.** τ = 0.07, λ = 0.5 and the
   128-d head are all project choices the paper never specifies; they can be
   selected on Batch 1 CV alone.
3. **The epoch protocol needs a decision.** Unaligned holds per-sample exposure
   constant and doubles compute; aligned holds compute constant and halves
   exposure. Neither is neutral, and the specification currently fixes the
   unaligned form.
4. **Small numerical fidelity items remain**: Gaussian sigma sampling instead of
   log-normal, dividing by the variance rather than the standard deviation, and
   block 5's 512-wide inner convolution.
