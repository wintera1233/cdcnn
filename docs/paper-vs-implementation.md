# Paper versus implementation

What `Sensors and Actuators: A. Physical 372 (2024) 115314` actually specifies,
where this project's code differs, and which differences are worth testing.

Sources in `docs/paper/`: `1-s2.0-S0924424724003078-main.pdf` (main text),
`1-s2.0-S0924424724003078-mmc1.docx` (supplement S1-S5), `fig2_cdcnn.png`
(framework figure). Text was read with `pdftotext -layout`; the supplement is a
zipped OOXML document whose `word/document.xml` carries the prose. Equation
images do not extract, so the equation numbers below come from the surrounding
text.

## Metric comparability

Supplement Table S1's "batch 1" training row is identical to the CDCNN row of
main-text Table 3, and its average is the unweighted mean of the nine per-batch
accuracies. That is the same statistic this project calls **target mean**, so the
numbers can be compared directly.

## Headline comparison

Paper rows are Table 3 ("This work"); ours are the five-seed means of
`20260915T025701880293Z_cdcnn_v6_3_a3_confound_full`.

| Model | B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 | Target mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper ResNet | 0.769 | 0.662 | 0.642 | 0.716 | 0.725 | 0.502 | 0.660 | 0.604 | 0.430 | **0.6346** |
| paper CDWC | 0.760 | 0.772 | 0.705 | 0.789 | 0.851 | 0.578 | 0.582 | 0.568 | 0.429 | **0.6705** |
| paper CDCNN | 0.827 | 0.768 | 0.724 | 0.789 | 0.932 | 0.618 | 0.715 | 0.566 | 0.566 | **0.7230** |
| ours B0 | 0.738 | 0.507 | 0.494 | 0.451 | 0.478 | 0.309 | 0.195 | 0.205 | 0.291 | **0.4077** |
| ours A2-semantic | 0.767 | 0.509 | 0.493 | 0.339 | 0.414 | 0.281 | 0.173 | 0.253 | 0.309 | **0.3931** |
| ours A3 | 0.894 | 0.683 | 0.576 | 0.349 | 0.414 | 0.330 | 0.172 | 0.304 | 0.419 | **0.4602** |
| ours B0-LN | 0.873 | 0.692 | 0.512 | 0.458 | 0.487 | 0.374 | 0.193 | 0.344 | 0.497 | **0.4922** |

Recomputing each paper row from its own nine values reproduces 0.6705 and
0.7230 exactly; the ResNet row gives 0.6346 against a printed 0.6344, a rounding
discrepancy in the source table.

The decisive observation is that the **paper's plain ResNet baseline (0.6344)
beats every stage this project has trained**, including the best one
(0.4922). The shortfall is therefore not located in the contrastive loss,
the feature generation block, or the augmentation: a backbone with none of them
already scores 0.23 higher in the paper. Largest per-batch gaps against our best
stage: B8 (+0.450), B6 (+0.444), B5 (+0.331), B7 (+0.234).

## What matches the paper

- Five ResNet blocks, `3x1` kernels, two convolutions plus a `1x1` shortcut per
  block, and no pooling layer in the backbone (Section 5.2, Fig. S4a).
- Feature generation placed between the third and fourth blocks (Section 2.4).
- `FC128 -> BatchNorm -> FC6` classifier head (Fig. 2). The paper's "0.77 K
  parameter classifier" equals `128x6 + 6 = 774`, i.e. the final layer only.
- Originals concatenated with augmented samples (Fig. 2; algorithm step 1).
- Augmentation by mixing per-sample mean and variance and sampling Gaussian
  noise added to the anchor, Eqs. (5)-(7).
- MaxPool plus nearest-neighbour upsampling for the decomposition, Eqs. (8)-(9).
- Loss `L_ce + lambda_MSE * L_MSE + lambda_con * L_con`, Eq. (4); MSE between the two
  branch outputs (Fig. 2 "L2 loss" between P1 and P2, Eq. S4).
- Dot-product similarity on unit-sphere features with the anchor's true label,
  Eq. (S5). No pseudo-labels anywhere.
- SGD with learning-rate damping (Section 5.2), 100 epochs (S2, Fig. S1).
- Batch 1 as the only source domain, targets never visited during training
  (Section 5.2, Fig. S7).
- Dataset composition per batch, Table 2, matches the loaded files exactly.

## Differences

| Item | Paper | This project | Status |
|---|---|---|---|
| Contrastive input | Fig. 2 feeds `z_f` and `z̄_f`, the ResNet5 outputs **before** FC128, into the contrast loss; S1 calls `P` a function mapping features to the unit sphere | learned `Linear(128 -> 128)` applied **after** FC128 | undeclared difference; test |
| Restyled branch | Eq. (15) `z̃_s = z^H_s + z̃^L_s`, so the **residual** (their L) is restyled | canonical A2-semantic and A3 restyle the **pooled** branch | declared adaptation; the faithful variant `A2-paper-literal` has never been run beyond smoke tests and has no A3 counterpart |
| Style statistics | Eqs. (10)-(11) sum over both C and H, giving one scalar per sample; the following sentence instead says `μ^L_s, δ^L_s ∈ R^{B×CH}` | per-channel, `[B, 128, 1]` | declared adaptation; the paper is self-contradictory here |
| Sigma sampling | Eq. (14) `δ'_s ~ N(μ̃, δ̃)`, an ordinary Gaussian that can return a negative scale | log-normal rule guaranteeing positivity | declared adaptation |
| Normalization denominator | Eq. (16) divides by `δ^L_s`, which Eq. (11) defines as a **variance** | divides by `sqrt(var + eps)`, a standard deviation | undeclared; the paper's own formula is dimensionally odd for an AdaIN-style transfer |
| ResNet5 width | Fig. 2 shows `3 Conv 512 -> 3 Conv 128` inside block 5 | both convolutions are 128 wide | undeclared difference |
| Input handling | Fig. 2 has a "Normal" block before augmentation; its definition appears nowhere | `StandardScaler` fitted on Batch 1 | project-controlled; unconstrained by the paper |
| A3 numerical safeguards | none mentioned | LayerNorm, `±20` clamps, sigma bounds, gradient clipping at 1.0, anchor-mean reduction | declared v6.3 deviations |
| Contrastive reduction | Eq. (S5) sums over anchors | mean over anchors | declared v6.3 deviation |
| Optimizer constants | only "SGD with learning rate damping"; batch size shown only as a trend in Fig. S2 | lr 0.001, momentum 0.9, weight decay 1e-4, `StepLR(25, 0.5)`, batch 64 | project-controlled, unconstrained |
| Temperature, head width, loss weights | not given | 0.07, 128, 0.5 / 0.5 | project-controlled, unconstrained |

## Ambiguities in the source material

1. **Statistics shape.** Eqs. (10)-(11) reduce over channel and length; the next
   sentence claims the result lives in `R^{B×CH}`. These cannot both hold.
2. **Variance versus standard deviation.** `δ` is called a variance in Eq. (6)
   and Eq. (11) but is used where AdaIN uses a standard deviation, Eq. (16).
3. **Projection function.** S1's `P` "maps the original feature to the unit
   sphere", which may mean L2 normalization alone or a learned head followed by
   normalization. Fig. 2 shows no head.
4. **The "Normal" block.** Never defined; it could be a fitted scaler, a
   per-sample normalization, or a normalization layer.
5. **H/L naming.** Eq. (8) calls the pooled, upsampled branch H and Eq. (9)
   calls the residual L, which is the opposite of the usual signal-processing
   reading. This project's `A2-semantic` renames them; `A2-paper-literal`
   preserves the paper's assignment.

## The paper's own per-class result

Fig. S3's CDCNN confusion matrix has one class with a zero diagonal. Matching
both readings of its labels against the paper's per-batch accuracies favours the
reading in which that class is **Ethylene** — the same class this project never
predicts, and which the SVM baselines also miss. Under that reading the paper
holds Ammonia at 0.94 and Toluene at 0.92 where this project loses both after
Batch 5, and its CDCNN-over-CDWC gain is concentrated in Toluene (0.27 to 0.92).
See `docs/per-class-failure.md`.

## Improvement points, ranked

Each is source-only and fits the existing ladder.

1. ~~**Input normalization.**~~ **Done (v6.4, v6.5).** Per-sample input
   normalization is worth +0.1087 target mean on 5/5 seeds and +0.1492 combined
   with LayerNorm, taking the project best from 0.4922 to 0.5569; signed-log and
   clipping do nothing, so the mechanism is removing each sample's offset and
   gain rather than bounding extreme values. Repeating the ablation ladder on
   normalized inputs (v6.5) showed the CDCNN components remain
   neutral-to-negative, so poor input conditioning was not what held them back.
   See `docs/input-normalization.md` and `docs/normalized-input-ladder.md`.
2. ~~**Run the paper-literal decomposition at five seeds.**~~ **Done (v6.6).**
   Restyling the paper's residual branch beats this project's semantic pooled
   branch on 5/5 seeds: +0.0137 target mean for A2 and +0.0201 for A3. Feature
   generation also turns mildly positive (+0.0044) once the paper's branch is
   used. The same run isolated A1 augmentation for the first time and found it
   is the component that hurts (−0.0131, 1/5 seeds).
   See `docs/paper-literal-ladder.md`.
3. ~~**Move the contrastive loss to `z_f` and drop the learned head**~~
   **Done (v6.10).** At the paper's placement the term is worth +0.0002 (2/5
   seeds) against +0.0025 for this project's placement, and the two differ by
   −0.0023. The placement was not the explanation; the contrastive effect is
   indistinguishable from zero under either.
   See `docs/contrastive-placement.md`.
4. **Test the statistics axis**: scalar per sample (the literal equations)
   against the current per-channel form. Resolves ambiguity 1 empirically.
5. **Smaller faithfulness items**: Gaussian sigma sampling with a positivity
   guard instead of the log-normal rule; dividing by the variance rather than
   the standard deviation; the 512-wide inner convolution in block 5.

**Blocking issue found while testing item 5's neighbourhood (v6.7, v6.8).** The
augmentation penalty this project has reported since v6.3 is a schedule
artefact: augmented stages take twice the optimizer steps at a fixed 100 epochs,
and a zero-noise duplication control reproduces the full penalty. Until the
protocol resolves that, no statement about what the paper's augmentation
contributes can be made from this project's numbers. See
`docs/duplication-control.md`.

Items 2-5 change the executable protocol, so each needs a new implementation
version, synchronized documentation, and a fresh GPU smoke artifact before any
launch.
