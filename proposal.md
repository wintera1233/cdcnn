# Proposal: reconstruct the paper's ResNet baseline

Branch `exp/v7-redesign`. Written 2026-09-23.

## 1. Why the baseline first

The paper's Table 3 reports a plain ResNet — "the backbone of CDCNN", no data
augmentation, no feature generation, no contrastive loss — at a target mean of
0.6344. The previous attempt (`exp/a3-confound-ablation`) never reached that
number with any configuration, its best being 0.5613 with a stack of additions
the paper does not describe. Every conclusion that project drew about the three
CDCNN components was therefore drawn on top of a backbone that was already
0.07-0.23 short, which makes those conclusions uninterpretable.

This proposal covers one thing: rebuild the backbone until it reproduces 0.6344,
or establish why it cannot. Nothing else is trained until that is settled.

## 2. What the paper actually specifies

Verified against `docs/paper/1-s2.0-S0924424724003078-main.pdf` (Fig. 2,
Section 5.2, Table 3) and `...-mmc1.docx` (S1). Equations in the supplement are
embedded WMF images; they were read by rendering the document to PDF.

### 2.1 Five blocks — confirmed

Section 5.2: "To get the core features of the gas signal, we use a neural
network with five ResNet blocks. The kernel's shape of each ResNet is 3 x 1, and
there are two convolution layers and a shortcut function with 1x1 kernel in each
block, and the channels of convolution increase from 1 to 128 step by step. Due
to the simple net, to get enough features to forecast data labels, this network
doesn't apply the pooling layer."

Section 2.4 adds that the feature generation block sits "in the middle of the
third and the fourth ResNet block", and Section 2.2 defines the figure's
notation: "3 Conv 128 means the kernel size is 3, and the number of filters is
128. The activation function chooses RULE" (i.e. ReLU).

So each block is `Conv1d(k=3) -> ReLU -> Conv1d(k=3)` summed with a
`Conv1d(k=1)` shortcut, then ReLU. No pooling anywhere in the backbone; the
length stays 128 throughout, which requires `padding=1` on the 3-kernels.

### 2.2 Kernel counts — read from Fig. 2

| Block | shortcut | conv 1 | conv 2 | in -> out | params |
|---|---|---|---|---|---:|
| Resnet1 | `1 Conv, 32` | `3 Conv 2` | `3 Conv 32` | 1 -> 32 | 296 |
| Resnet2 | `1 Conv, 64` | `3 Conv 64` | `3 Conv 64` | 32 -> 64 | 20,672 |
| Resnet3 | `1 Conv, 128` | `3 Conv 128` | `3 Conv 128` | 64 -> 128 | 82,304 |
| Resnet4 | `1 Conv, 256` | `3 Conv 256` | `3 Conv 256` | 128 -> 256 | 328,448 |
| Resnet5 | `1 Conv, 128` | `3 Conv 512` | `3 Conv 128` | 256 -> 128 | 623,360 |

Two entries deserve attention.

**`3 Conv 2` in Resnet1 is almost certainly a typo.** The block would map
1 channel -> 2 channels -> 32 channels while its shortcut maps 1 -> 32 directly.
A 2-channel bottleneck at the first layer discards nearly all of the input before
the network starts, and it is the only place in the figure where the main path is
narrower than the shortcut. The intended value is probably 32. Both readings are
cheap to run, so both are in the ladder below rather than being guessed at.

**`3 Conv 512` in Resnet5 contradicts the prose.** Section 5.2 says channels
"increase from 1 to 128 step by step"; the figure reaches 256 in Resnet4 and 512
inside Resnet5. The figure is the more specific statement and is followed here.

### 2.3 Input and output

Input is one measurement: a 128-vector, fed as `[N, 1, 128]`. Section 5.1
describes it as "a 128-dimensional (8 x 16) time series vector" — 8 features
extracted per sensor across 16 sensors. Per `CLAUDE.md` section 5 this is not
reshaped into a 2D image.

The backbone leaves `[N, 128, 128]`, 128 channels at unchanged length. The head
is `FC 128 -> Batch Normal -> FC 6` (Fig. 2). With no pooling, `FC 128` consumes
the flattened 16,384 values.

Output is 6 logits, one per gas.

The paper's "the classifier of the CDCNN algorithm requires only 0.77 K
parameters" refers to `FC 6` alone: 128 x 6 + 6 = 774. It is a comparison against
TDACNN's 54 K classifier and says nothing about the rest of the network.

There is also a **`Normal` block before augmentation** in Fig. 2. Its definition
appears nowhere in the paper or supplement. This is the single largest piece of
free choice in the reconstruction; see 4.2.

### 2.4 The loss — `L_ce` only, and its exact form

Main text Eq. (4) gives the full CDCNN objective as
`min(L_ce + lambda_MSE * L_MSE + lambda_con * L_con)`, and states that "the
cross-entropy loss `L_ce` is the primary loss", while `L_MSE` and `L_con` are
"auxiliary loss functions".

`L_MSE` (Eq. S4) is defined between the original and the generated feature and
requires the feature generation block. `L_con` (Eq. S5) is defined over the pair
`(z_f, z_bar_f)` and requires the same block plus the contrastive head. **Neither
term is computable without the components the ResNet baseline omits, so the
baseline trains on `L_ce` alone.** Confirmed by Fig. 2, where only the `P1` path
carries `cross_entropy loss` while `L2 loss` needs both `P1` and `P2`.

Supplement S1 gives the general form (S1) and the task-specific form (S2):

```
(S1)    -sum Y log(p1)

(S2)    L_ce = -(1/B) sum_{i=1..B} (1/n) sum_{j=1..n}
                   Y_j^i * log( S( FC( Phi( X_j^i ) ) ) )
```

with, verbatim from S1: `p1 = FC(z_f)`; `z_f` "is the feature of input, which is
extracted from input data by the feature extractor"; the extractor `Phi` "is
composed of `R_s` and `R_f`" (the five blocks, split at the feature-generation
point); `S` "indicates the softmax function, B represents the number of batches,
and n means the size of each batch".

`FC` here is the whole classifier head, `FC128 -> BatchNorm -> FC6`, since
Fig. 2 routes `z_f` through all three before `P1`.

This is ordinary multiclass cross-entropy with mean reduction, i.e.
`nn.CrossEntropyLoss()` on the logits, with one wrinkle: S2 averages within a
batch and then across batches. That equals a global mean only when every batch
has the same size `n`. With 445 source rows and any batch size that does not
divide 445, the final partial batch is up-weighted under S2. The reconstruction
will follow S2 literally (mean per batch, then mean over batches) and record the
difference from a global mean; it is expected to be negligible but it is free to
get right.

### 2.5 Training protocol from the paper

- 100 epochs (Fig. S1 caption, explicit).
- "standard Stochastic gradient descent (SGD) and learning rate damping"
  (Section 5.2). No learning rate, momentum, weight decay, or schedule given.
- Batch size is swept over 16 / 32 / 64 / 128 / 256 in Fig. S2; which value
  produced Table 3 is not stated.
- Batch 1 is the only source domain; the other batches "cannot be visited"
  (Section 5.2, Fig. S7).
- Control methods including ResNet were run "in the same setting".

## 3. The reference numbers

Table 3, row `ResNet`, "This work":

| B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | printed avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.7692 | 0.6618 | 0.6424 | 0.7160 | 0.7253 | 0.5024 | 0.6602 | 0.6035 | 0.4302 | 0.6344 |

Recomputing the unweighted mean of those nine values gives 0.6346; the printed
0.6344 is a rounding error in the paper. Either is the target.

## 4. What is not specified, and must be chosen

### 4.1 Capacity

The literal architecture, built and counted:

```
Resnet1      296        Resnet4      328,448
Resnet2   20,672        Resnet5      623,360
Resnet3   82,304        backbone   1,055,080

FC128  2,097,280   BatchNorm 256   FC6 774   head  2,098,310
                                             TOTAL 3,153,390
```

**3.15 M parameters trained on 445 samples, 7,086 parameters per sample**, of
which two thirds are the single `Flatten -> Linear(16384, 128)`. The previous
project measured Batch 1 cross-validation accuracy of 0.978-0.991 for every
configuration it tried while target accuracy ranged from 0.37 to 0.56: the source
task is saturated and carries no signal about generalization. A head this wide is
the most likely place for the missing 0.23 to be hiding.

Global average pooling before `FC 128` would cut the model to 1.07 M parameters
and is the standard choice for a ResNet, but it contradicts "this network doesn't
apply the pooling layer". Both are run.

### 4.2 The `Normal` block

Undefined in the paper, so any choice is compatible with it. The previous project
measured **+0.109 target mean for per-sample normalization** over a Batch-1-fitted
`StandardScaler`, its single largest effect, and **+0.085 for LayerNorm over
BatchNorm** inside the blocks. Both are legitimate readings of an undefined
"Normal" block and both are run.

### 4.3 Optimizer constants

"Learning rate damping" is not a schedule. The previous project's
`SGD(lr=0.001, momentum=0.9, weight_decay=1e-4)` with `StepLR(25, 0.5)` is
carried over unchanged, not because it is right but because changing it at the
same time as the architecture would confound the comparison. Batch size 64, in
the middle of the paper's sweep. These are held fixed across the whole ladder and
revisited only if the ladder fails.

## 5. The open question Fig. S1 raises

Fig. S1(a) plots accuracy over 100 epochs "in a training set" for Resnet, CDWC
and CDCNN. The three curves plateau at roughly **0.58, 0.65 and 0.70**, which are
close to those models' *target* means in Table 3 (0.6344, 0.6705, 0.7230) and
nowhere near the 0.98 Batch 1 accuracy the previous project reached.

Two readings, both consequential:

1. **It is training accuracy.** Then the paper's ResNet never fits its source
   domain — 0.58 on 445 samples with 3.15 M parameters — and its whole training
   regime is far more constrained than anything tried here. Our baseline would be
   overfitting Batch 1 by a wide margin, and that is the defect to fix.
2. **It is target accuracy.** Then the paper monitored target-domain accuracy
   during training, and the reported numbers are not strictly source-only.

Fig. S1(b) does not settle it: the ResNet loss plateaus near 0.55, which for six
classes is more consistent with high accuracy than with 0.58.

This is testable without violating the protocol. Each run saves a checkpoint at
every epoch; **after all checkpoints are frozen**, both curves are computed — Batch
1 accuracy per epoch and target accuracy per epoch — and overlaid on Fig. S1. If
our training curve sits at 0.98 where theirs sits at 0.58, reading 1 holds and
capacity is the problem. If our target curve traces theirs, reading 2 holds.

## 6. Proposed experiment

One ladder, five variants, `L_ce` only throughout, five seeds each, 25
checkpoints. Every variant is declared before any run starts; none is selected on
target data.

| Variant | Change from the literal reading | Isolates |
|---|---|---|
| `R-lit` | none: Fig. 2 exactly, `3 Conv 2`, flatten head, `StandardScaler` | the paper as printed |
| `R-w32` | `R-lit` with Resnet1 conv1 widened 2 -> 32 | the suspected figure typo |
| `R-gap` | `R-w32` with global average pooling before `FC 128` | head capacity, 3.15 M -> 1.07 M |
| `R-ps` | `R-w32` with per-sample input normalization as `Normal` | the undefined block, known +0.109 |
| `R-ln` | `R-w32` with LayerNorm in place of BatchNorm | known +0.085 |

Read as: `R-lit -> R-w32` is the typo; `R-w32 -> R-gap` is capacity;
`R-w32 -> R-ps` and `R-w32 -> R-ln` re-measure the previous project's two real
effects on a correct backbone. `R-ps` and `R-ln` are kept separate so their
interaction is not assumed; if both help, the combination is a second, declared
run rather than a post-hoc pick.

Cost: 25 runs x 100 epochs on 445 rows. The previous project's comparable ladders
finished in well under an hour of GPU time each.

### Success criteria

- **Reproduced**: some variant reaches target mean 0.63 +- 0.02 with the per-batch
  vector broadly tracking the paper's, in particular B8 >= 0.60 and B6 >= 0.70,
  the two batches where the previous project fell short by 0.44.
- **Explained**: no variant reaches it, but the epoch curves of section 5 identify
  which reading of Fig. S1 holds and therefore what the remaining gap is made of.
- **Failed**: neither. Then the optimizer constants inherited in 4.3 become the
  next suspect and are swept before any CDCNN component is built.

## 7. Protocol

Unchanged from `CLAUDE.md` and non-negotiable: Batch 1 only for training,
normalization fitting, and any cross-validation; Batches 2-10 opened once, after
all 25 final checkpoints and all per-epoch checkpoints are written and hashed;
a leakage audit in every run directory; one worker on CUDA; a passing GPU smoke
artifact before the full run; unique `runs/<timestamp>_<name>/` directories.

Batch 1 CV is reported but is **not** used to select between variants. The
previous project showed it saturates at 0.96-0.99 across configurations whose
target means differ by 0.19, so it cannot discriminate. All five variants are
reported; any later choice among them is declared as target-informed.

Environment: rebuild the pinned container at torch 2.5.1+cu121. The `.venv`
currently on this machine holds torch 2.8.0+cu126, the build implicated in the
Xid 31 MMU fault on 2026-09-21.

## 8. What this proposal does not do

No data augmentation, no feature generation, no contrastive loss, no projection
head, no numerical-stability package. Those are only worth rebuilding once the
backbone they sit on is known to work.
