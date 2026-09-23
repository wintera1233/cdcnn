# Proposal: reconstruct the paper's ResNet baseline

Branch `exp/v7-redesign`. Written 2026-09-23.

## 0. Work plan

Thirty items in five stages. Two stopping points, marked STOP, where the work
pauses for review before continuing. Nothing is trained before the first one.

### Stage 0 — environment (1 item)

1. Restore `docker/Dockerfile` and `requirements-cu121.txt` from
   `exp/a3-confound-ablation`, rebuild the container at torch 2.5.1+cu121, verify
   CUDA. The `.venv` on this machine holds 2.8.0+cu126, the build implicated in
   the Xid 31 MMU fault of 2026-09-21.

### Stage 1 — code (11 items)

2. `src/protocol.py` — `ProtocolError`, the target-access recorder.
3. `src/data.py` — LIBSVM loader, Batch 1 gate, file hashing.
4. `src/normalize.py` — the `Normal` block: Batch-1 `StandardScaler` and
   per-sample normalization.
5. `src/model.py` — residual block, the channel specification per variant,
   flatten and GAP heads.
6. `src/loss.py` — `L_ce` exactly as Eq. (S2): mean within a batch, then mean
   across batches.
7. `src/train.py` — 100 epochs, SGD, `StepLR`, fixed seeds, a checkpoint at every
   epoch for the seed named by `per_epoch_checkpoint_seed`, and a final
   checkpoint for every seed. Saving every epoch of all twelve runs would cost 6.2 GB
   against 28 GB free; the Fig. S1 question in section 5 needs one seed per
   variant, so only seed 1042 saves per epoch, for 2.1 GB.
8. `src/evaluate.py` — post-freeze target evaluation, per-batch accuracy, target
   mean, confusion matrices.
9. `src/audit.py` — leakage audit and the run manifest: code hash, config hash,
   library and driver versions.
10. `src/config.py` — configuration loading, and validation that pins the
    ladder's declared variants, seeds and protocol constants.
11. `scripts/run_baseline.py` — the single entry point: `smoke`, `gpu-smoke`,
    `launch`, `evaluate`.
12. `configs/baseline_ladder.json` — the declared experiment.

### Stage 2 — tests (8 items)

13. The loader reproduces 445 rows and the 90 / 98 / 83 / 30 / 70 / 74 class
    histogram.
14. `L_ce` equals `CrossEntropyLoss` when the batch size divides the row count,
    and differs as predicted when it does not.
15. Parameter counts match section 6 exactly for every variant.
16. The backbone output is `[N, 128, 128]`; no pooling anywhere inside it.
17. The audit raises if a target file is opened before the last checkpoint is
    frozen.
18. The same seed reproduces bit-identical losses.
19. Configuration validation rejects an undeclared variant.
20. A run refuses to write into an existing run directory.

**STOP.** Test output is reviewed before any GPU time is spent.

### Stage 3 — gates (2 items)

21. CPU smoke: one seed, few epochs, end to end.
22. GPU smoke artifact on Batch 1 only, as `CLAUDE.md` section 6 requires.

### Stage 4 — training and evaluation (4 items)

23. 12 training runs: 4 variants x 3 seeds x 100 epochs.
24. Freeze and hash all 12 final checkpoints and the 400 epoch checkpoints of
    seed 1042.
25. One target evaluation pass: Batches 2-10 opened once, per-batch accuracy and
    target mean.
26. Compute the Batch 1 and target accuracy curves per epoch from the frozen
    checkpoints and overlay them on Fig. S1, to settle the question in section 5.

### Stage 5 — record (4 items)

27. `baseline.md` — the final design and every cell against the paper's 0.6344.
28. The Fig. S1 overlay figure.
29. `docs/change-log.md` — a fresh log for this branch.
30. `docs/run-cleanup-20260923.md` — append the runs this stage produces.

**STOP.** Results are reviewed before deciding whether to rebuild any CDCNN
component.

### Out of scope

No data augmentation, no feature generation, no contrastive loss, no projection
head, no numerical-stability package. See section 8.

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
narrower than the shortcut. The intended value is probably 32. It is read as 32;
section 4.4 explains why this is not worth a run of its own.

**`3 Conv 512` in Resnet5, and `256` in Resnet4, contradict the prose.**
Section 5.2 says the channels "increase from 1 to 128 step by step"; the figure
reaches 256 in Resnet4 and 512 inside Resnet5, then comes back down. The two
statements cannot both hold. Three readings are possible:

| Reading | Blocks | Backbone | Invents |
|---|---|---:|---|
| the figure as printed | 32, 64, 128, 256, 128 | 1,055,080 | nothing |
| the text, capped at 128 | 32, 64, 128, 128, 128 | 336,416 | nothing |
| the text, strictly increasing | 8, 16, 32, 64, 128 | 109,768 | the values 8 and 16 |

The middle reading is taken as the default for the ladder. It keeps every width
the figure actually prints for Resnet1-3, which the prose does not contradict
(32 -> 64 -> 128 *is* an increase from 1 to 128 step by step), and overrides the
figure only where the two sources conflict. The strictly increasing reading has
to invent two widths that appear nowhere in the paper, and section 4.4 shows the
figure reading has in effect already been measured, so neither alternative is
given a run here.

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

### 4.1 Capacity, and where the paper's lightweighting claim points

The paper repeatedly frames CDCNN as cheap:

- Section 2.4: previous work "transferred the gas signals into gray images and
  used two-dimensional convolution [18]. Instead, this work has designed the
  single-dimensional convolution [19] for less cost". Reference [19] is the
  authors' own *Lightweight neural network for gas identification*.
- Section 5.2: "A deep network to extract these features is not theoretically
  feasible. Therefore, we replaced it with a simple net."
- Section 5.3: "The classifier of the TDACNN algorithm contains 54 K parameters,
  while the classifier of the CDCNN algorithm requires only 0.77 K parameters."
- Conclusion (4): "CDCNN consumes fewer resources during testing."

None of these constrains the backbone widths. The saving claimed in 2.4 is 1D
convolution instead of 2D; the saving in the conclusion is not visiting the
target domain and not needing TDACNN's four-classifier ensemble.

Counted, the choice of widths barely matters and the choice of head decides
everything:

| Backbone | with flatten head | with GAP head |
|---|---:|---:|
| figure literal, 1,055,080 | **3,153,390** | 1,072,622 |
| text capped at 128, 336,416 | **2,434,726** | 353,958 |
| text strictly increasing, 109,768 | **2,208,078** | 127,310 |

`Flatten -> Linear(16384, 128)` costs **2,098,310** parameters under every channel
reading. Taking the strictly increasing reading removes 90 % of the backbone and
only 30 % of the model. Replacing the flatten with global average pooling takes
that same head from 2,098,310 to 17,542.

**This inverts the 0.77 K claim into evidence.** Boasting that the classifier is
774 parameters against TDACNN's 54 K is only meaningful if there is not a 2.1 M
linear layer immediately before it. Either the paper's accounting is misleading,
or its implementation does not flatten 128 channels x 128 length into the head at
all. The second possibility is consistent with Fig. S1 (section 5), where the
ResNet accuracy curve plateaus at 0.58 rather than saturating as a 3.15 M model
on 445 rows would.

So the capacity question is real, but it lives in the head, not in the channel
widths. The literal architecture is **3.15 M parameters on 445 samples, 7,086 per
sample**, and the previous project measured Batch 1 cross-validation accuracy of
0.978-0.991 for every configuration it tried while target accuracy ranged from
0.37 to 0.56: the source task is saturated and carries no signal about
generalization. Global average pooling contradicts "this network doesn't apply
the pooling layer", so it is run as a declared variant against the faithful build,
not adopted.

### 4.2 The `Normal` block

Undefined in the paper, so any choice is compatible with it. The previous project
measured **+0.109 target mean for per-sample normalization** over a Batch-1-fitted
`StandardScaler`, its single largest effect, and **+0.085 for LayerNorm over
BatchNorm** inside the blocks. Per-sample normalization is carried into the ladder
as the third rung. LayerNorm is not: it is a second per-sample normalization
acting at a different place, so running both would measure one mechanism twice.

### 4.3 Optimizer constants

"Learning rate damping" is not a schedule. The previous project's
`SGD(lr=0.001, momentum=0.9, weight_decay=1e-4)` with `StepLR(25, 0.5)` is
carried over unchanged, not because it is right but because changing it at the
same time as the architecture would confound the comparison. Batch size 64, in
the middle of the paper's sweep. These are held fixed across the whole ladder and
revisited only if the ladder fails.

### 4.4 What the previous project has already measured

`exp/a3-confound-ablation:src/cdcnn_ablation.py:501` built its backbone as

```python
channels = (32, 64, 128, 256, 128)
```

with `ResidualBlock1D(incoming, outgoing)` — two 3-kernels at the block's output
width, a 1-kernel shortcut, no pooling — followed by
`Flatten -> Linear(16384, 128) -> BatchNorm1d(128) -> Linear(128, 6)`. That is the
figure's channel reading with a flatten head, differing from `R-fig32` only in
Resnet5's inner width. Its measured target means:

| Previous stage | Equivalent here | Target mean |
|---|---|---:|
| `B0` | figure channels, flatten head, `StandardScaler` | 0.4097 |
| `B0-LN` | the same with LayerNorm | 0.4922 |
| `B0-PS` | the same with per-sample inputs | 0.5164 |

So the figure's channel reading combined with a flatten head has already been
measured three times, and tops out at 0.52 against the paper's 0.6344. Re-running
`R-fig`, `R-fig32` and `R-mono` would spend fifteen runs re-confirming a known
result; the widths are not where the missing 0.23 is.

**The head has never been varied.** `Flatten -> Linear(16384, 128)` is present in
every configuration either project has ever trained. That is what the ladder in
section 6 tests.

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

A 2x2 factorial on one backbone, `L_ce` only throughout, three seeds each, 12
checkpoints. Every variant is declared before any run starts; none is selected on
target data.

The two factors are the only two places the paper leaves free that the previous
project's evidence points at:

| | `Normal` = `StandardScaler` | `Normal` = per-sample |
|---|---|---|
| **flatten** head, 2,434,726 params | `R-txt` | `R-txt-ps` |
| **GAP** head, 353,958 params | `R-lite` | `R-lite-ps` |

All four share the backbone `32, 64, 128, 128, 128` at 336,416 parameters and
differ only in the head's reduction and the input normalisation. Both main
effects and their interaction are therefore estimable.

`R-txt` is the faithful build and doubles as the correctness check: section 4.4
predicts it lands near 0.41, the previous project's `B0`. If it does not land in
roughly 0.40-0.42, the reconstruction has a bug and nothing else is worth
reading.

**The head factor.** `Flatten -> Linear(16384, 128)` costs 2,097,280 parameters
and keeps every position of the length axis separately addressable. Global
average pooling collapses each channel to its mean, costing 16,384 parameters and
discarding where along the axis a response sat. That axis is not time: position
`8(s-1)+k` is statistic `k` of sensor `s`, so a flatten head can learn "sensor
11's second statistic should be about this much" — exactly the kind of rule that
sensor ageing invalidates. Measured on an untrained model, drifting 2 of the 16
sensors by 20 % perturbs the flattened representation 5.9 times more than the
pooled one. This factor has **never been varied**: every configuration either
project has trained carried the flatten head.

The risk runs the other way too. Position 1 of each sensor group is a
steady-state resistance of order 1e4 while the six transient features are of
order 1; averaging them together lets the large ones dominate, and gas identity
may live precisely in the cross-sensor ratios that the average destroys. The
paper's "this network doesn't apply the pooling layer" may be load-bearing.

**The normalisation factor.** `StandardScaler` applies Batch 1's per-feature mean
and variance to a target batch recorded up to three years later. Per-sample
normalisation standardises each measurement against its own 128 values, fits
nothing, and was worth +0.109 to the previous project.

Both factors do the same kind of thing — refusing to carry Batch 1's coordinates
into a drifted batch — one at the input and one at the feature map, so they may
well overlap. The full 2x2 measures that overlap instead of assuming it away; a
three-rung ladder could not.

Cost: 4 variants x 3 seeds = 12 runs x 100 epochs on 445 rows. The previous
project's comparable ladders finished in well under an hour of GPU time each.

### Held back

`R-fig` and `R-fig32` (the figure's 256 / 512 widths, and the `3 Conv 2` typo),
`R-mono` (8, 16, 32, 64, 128) and `R-ln` (LayerNorm) were in an earlier draft and
are dropped: the first three are answered indirectly by section 4.4, and
LayerNorm is a third instance of the mechanism both factors above already test.
They are reinstated only if the four cells fail to reach 0.63.

### Seeds, and when to add more

Seeds `1042`, `2024`, `3407` — the first three of the five the previous project
used, fixed in advance so the choice cannot be revisited after seeing results.

Three seeds is enough to size a large effect and not enough to resolve a small
one. The previous project measured seed-to-seed standard deviations of the target
mean between **0.0186 and 0.0415**; at the upper end, three seeds give a standard
error of 0.024, so a rung-to-rung difference below about 0.05 would not be
separable from noise.

The results report the standard deviation for every cell, and the work stops
to ask before drawing a conclusion if either:

- any variant's target-mean standard deviation exceeds **0.02**, or
- a rung-to-rung difference is smaller than **twice the pooled standard error**.

Seeds `42` and `123` are held in reserve for that case; adding them is a second
declared run, not a re-run.

The effect this ladder is looking for is large — `R-txt -> R-lite` changes the
model by a factor of seven and the hoped-for gain is above +0.10 — so three seeds
should settle it. The escalation rule exists because the previous project's
spreads make triggering it a real possibility, not a remote one.

### Held back

`R-fig` and `R-fig32` (the figure's 256 / 512 widths, and the `3 Conv 2` typo),
`R-mono` (8, 16, 32, 64, 128) and `R-ln` (LayerNorm) were in an earlier draft of
this ladder and are dropped: the first three are answered indirectly by section
4.4, and LayerNorm duplicates the mechanism `R-lite-ps` already tests. They are
reinstated only if the three rungs above fail to reach 0.63.

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
all 12 final checkpoints and all per-epoch checkpoints are written and hashed;
a leakage audit in every run directory; one worker on CUDA; a passing GPU smoke
artifact before the full run; unique `runs/<timestamp>_<name>/` directories.

Batch 1 CV is reported but is **not** used to select between variants. The
previous project showed it saturates at 0.96-0.99 across configurations whose
target means differ by 0.19, so it cannot discriminate. All four variants are
reported; any later choice among them is declared as target-informed.

Environment: rebuild the pinned container at torch 2.5.1+cu121. The `.venv`
currently on this machine holds torch 2.8.0+cu126, the build implicated in the
Xid 31 MMU fault on 2026-09-21.

## 8. What this proposal does not do

No data augmentation, no feature generation, no contrastive loss, no projection
head, no numerical-stability package. Those are only worth rebuilding once the
backbone they sit on is known to work.
