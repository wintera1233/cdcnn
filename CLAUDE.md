# Project Instructions for AI Agents

This branch (`exp/v12-drift-projection`, from `exp/v7-redesign`) carries the
whole ladder: the baseline is settled (`baseline.md`), directed augmentation,
feature generation and a signed feature generation are all measured (section 4),
the contrastive loss `L_con` is implemented but, by decision, not run, and
v12.0 - projecting Batch 1's own drift axis out of the input - is measured
(`proposal-v12.md`, `docs/v12-drift-projection.md`). The rules below are protocol, not
method: they say nothing about which architecture, optimizer, or training
schedule to use, because those are what the redesign is for. Section 4 records
what has already been measured - do not spend GPU time re-deriving any of it.

## 1. Data

The UCI Gas Sensor Array Drift dataset, `Dataset/batch{1..10}.dat`, LIBSVM
format, 13,910 measurements, 128 features per measurement, six gas classes.

| Batch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rows | 445 | 1244 | 1586 | 161 | 197 | 2300 | 3613 | 294 | 470 | 3600 |

Batch 1 class counts by label: 90, 98, 83, **30**, 70, 74.

Label mapping (`docs/label-mapping.md`), from the dataset's own documentation
with labels 2 and 3 exchanged on the evidence of Batch 1's principal-component
structure against the paper's Fig. 4(a):

`1 = Ethanol, 2 = Ammonia, 3 = Ethylene, 4 = Acetaldehyde, 5 = Acetone, 6 = Toluene`

The paper's Table 2 is the UCI count table copied verbatim and matches this file
in every batch under a column permutation (total mismatch 9 over 60 cells, two
typos); what it cannot settle is which name goes with which label, because the
UCI documentation's text encoding and its own count table disagree. The counts
per label are the fact and the names are the interpretation; the paper's Fig. S3
is consistent with the text encoding, which the adopted mapping follows up to the
2/3 exchange. Changing the mapping changes no measurement, because only integer
labels reach the model. See `docs/label-mapping.md`.

Rules:

- **Immutable.** Never modify, rename, move, delete, or clip anything under
  `Dataset/`. Never introduce automatic outlier removal or value imputation.
- **No metadata as features.** Do not assume concentration labels, timestamps, or
  device IDs exist. If any are derived, keep them out of model inputs.

## 2. Source-only protocol

This is the point of the whole study and is not negotiable.

- **Batch 1 is the only source.** All training, scaler or transform fitting,
  hyperparameter tuning, architecture selection, early-stopping decisions, and
  cross-validation use Batch 1 and nothing else.
- **Batches 2–10 are the target and stay closed** until every checkpoint for the
  comparison is written to disk and hashed. Opening a target file earlier
  invalidates the run, including "just to check the shape".
- **Audit every run.** A run records when each target file was first opened and
  when each checkpoint was frozen, and fails if any target access precedes the
  last freeze.
- **Selection may not consult the target.** If a choice was made after seeing
  target numbers, it is a target-informed choice and the run reports it as such.

## 3. The comparison statistic

**Target mean** = the unweighted mean of the nine per-batch accuracies on Batches
2–10. This is the statistic the paper reports and the only one used for headline
comparisons. Report the pooled accuracy and the per-batch vector alongside it,
never instead of it. Report seed-to-seed spread with every mean; differences
below roughly 0.005 are not effects.

Reference values from the paper's Table 3:

| Model | Target mean |
|---|---:|
| paper ResNet (no CDCNN components) | 0.6346 |
| paper CDWC | 0.6705 |
| paper CDCNN | 0.7230 |

## 4. What is already established

`baseline.md` holds the settled baseline design. Do not spend GPU time
re-deriving any of the following.

### From this branch (v7.0 to v12.0, `docs/`)

#### The baseline (v7.0 to v7.3)

- **The baseline stands at target mean 0.5556** over five seeds against the
  paper's 0.6344. This is the settled source-only number and nothing since has
  replaced it. `docs/all-results.md` lists every cell trained.
- **Per-sample input normalisation is worth +0.112**; it and a LayerNorm head are
  substitutes with an interaction of -0.141, so use exactly one, at the input.
- **A signed-log before per-sample is worth a further +0.032**, separable at five
  seeds, improving Ethanol and Ethylene and leaving Acetone and Toluene alone.
  Grouping the normalisation by statistic across sensors costs -0.097; averaging
  seeds' softmax is worth +0.004.
- **Drift is a sensor phenomenon, not a gas one.** Ethanol, Acetaldehyde, Acetone
  and Toluene drift along one shared direction, pairwise cosine 0.87 to 0.985,
  and Batch 1's own two-session block offset estimates that direction at cosine
  0.850 against Acetaldehyde's real drift - from Batch 1 alone. See
  `docs/batch1-internal-drift.md`.
- **Global average pooling costs -0.078.** The paper's "this network doesn't
  apply the pooling layer" is load-bearing.
- **Memorisation is not the defect.** The flatten head reaches 1.0000 on Batch 1
  and still beats the pooled head on target.
- **No source-only stopping rule exists.** Held-out accuracy and held-out loss
  are anti-correlated with target accuracy across the window where it collapses;
  see `docs/early-stopping.md`. The epoch count must be fixed by fiat.
- **Acetaldehyde does not transfer.** Source recall 1.000, target recall 0.001,
  and every target batch's Acetaldehyde centroid lands off its own, eight times
  of nine on Batch 1's Ethanol. Class weighting cannot help it. **The paper's own
  CDCNN also scores 0.00 on Acetaldehyde** and sends 100% of it to Ethanol, so
  the gap to the paper lies in the other five classes; see `baseline.md`
  section 5 and `docs/why-acetaldehyde.md`.

#### The drift geometry (`docs/drift-geometry.md`)

Measured on the 51 class-by-batch centroid displacements (6 gases x 9 batches,
less the three where Toluene is absent), by uncentred SVD.

- **Drift is roughly four-dimensional.** PC1 carries 70.6%, and 90% of the energy
  needs four components. A single shared direction - what v8.0 and v8.1 use -
  captures 70.6% of it. The spectrum falls smoothly, so "four" is a cut through a
  continuous curve, and 51 vectors in 128 dimensions make the tail noisy.
- **Ethylene is the exception**: k=1 covers only 23.9% of its drift and Batch 1's
  own block-offset subspace only 24.9%, against 76% to 81% for the other five.
  This is why Ethylene is the one class every directed method damages.
- **The geometry is the same at every depth**, so analysing it in the normalised
  input space is representative: PC1 is 70.6% at the input, 65.4% after block 3
  and 64.5% at `z_f`, and 90% needs 4, 5 and 5 components respectively.
- **The backbone does not learn drift invariance.** Drift over within-class radius
  falls from 1.798 at the input to 1.662 at `z_f` - five ResNet blocks remove
  **7.5%** of it. Directed augmentation makes this worse, not better: `R-aug-t2`
  gains +0.0214 in accuracy while its compression drops to 5.7%. **Augmentation
  therefore does not work by inducing invariance** - no loss term asks for it. It
  moves the decision boundary: Ethanol's over-prediction falls from 2.46x to
  1.83x and the freed quota goes to Acetone (0.57x to 1.05x), paid for by
  Ethylene.
- **Only the contrastive loss has an invariance mechanism.** `L_con` (Eq. S5) and
  `L_MSE` (Eq. S4) are the only terms that tie the original and generated
  branches together, and `L_con` needs feature generation to exist first. Testing
  the paper's central claim requires reaching that step.

#### v8.0 directed augmentation (`docs/v8-augmentation.md`)

Run `runs/20260923T033747817275Z_augmentation_full`, four cells x five seeds,
pooled SD 0.0177, threshold 0.0224.

| variant | augmentation | target mean | SD | vs baseline | separable |
|---|---|---:|---:|---:|---|
| `R-aug-ethd` | directed only, T=18 | 0.5670 | 0.0096 | +0.0115 | no |
| `R-fig-logps` | none (baseline) | 0.5556 | 0.0137 | - | - |
| `R-aug-eth` | directed + isotropic | 0.5462 | 0.0190 | -0.0093 | no |
| `R-aug-paper` | the paper's isotropic | 0.5340 | 0.0249 | -0.0216 | no |

- **The pre-registered coverage prediction was directionally right**: the ordering
  of the four cells matched the predicted coverage exactly, and only the cell
  predicted to move did. No cell was separable, so this is agreement in trend.
- **The mechanism runs, on the wrong classes.** Acetone +0.250 and Toluene +0.124;
  Ethylene **-0.326**, and Ethylene is precisely the class whose drift is least
  aligned with the shared direction (cosine 0.451 against 0.77 to 0.94).
- **Acetaldehyde stays dead** (0.000 to 0.001). It drifts almost alongside Ethanol
  - cosine 0.761, magnitude ratio 1.24 - so a single shared direction translates
  both by the same amount and leaves them on top of each other. Undoing that needs
  a **per-class** displacement magnitude, and that ratio is only knowable from the
  target. This is the structural limit of shared-direction augmentation.

#### v8.1 the displacement sweep (`docs/v8-augmentation.md`)

Run `runs/20260923T045843638020Z_displacement_full`, four cells x five seeds,
pooled SD 0.0109, threshold 0.0137. Two changes from v8.0: T from 18 to 2/3/4,
and isotropic noise removed entirely.

| variant | T | target mean | SD | vs baseline | separable |
|---|---:|---:|---:|---:|---|
| **`R-aug-t2`** | 2 | **0.5770** | 0.0078 | **+0.0214** | **yes** |
| `R-aug-t3` | 3 | 0.5731 | 0.0104 | +0.0175 | **yes** |
| `R-aug-t4` | 4 | 0.5675 | 0.0108 | +0.0119 | no |
| `R-fig-logps` | - | 0.5556 | 0.0137 | - | - |

- **This is the project's first separable CDCNN-component gain.**
- **v8.0's T=18 was six times too large.** Drift does not accumulate linearly with
  time: measured in units of Batch 1's own block offset, every real class-by-batch
  drift lands between **0.83 and 3.74**, and B10 (36 months) is smaller than B8
  (22.5 months). The 18 came from extrapolating 36/2.
- **Isotropic noise was swept and then cut.** At sigma >= 0.25 Acetaldehyde's
  coverage falls back to the unaugmented 38.4%; the paper's noise displaces by
  11.3 with a directional component of about 3.
- **The whole gain is Acetone (+0.255), paid for by Ethylene (-0.272), and the
  dead class stays dead** (0.000 to 0.003). Larger T trades more Ethylene for more
  Acetone and Toluene.
- **Coverage is a crude predictor only.** It ranked T=3 first when T=2 won, and for
  Toluene it points the wrong way (T=18: coverage 5.3%, recall 0.477; T=2:
  coverage 32.5%, recall 0.362). It sees proximity to the target region, not how
  the classifier then reallocates its prediction quota - and the quota is what
  drives the Ethylene loss.
- **`R-aug-t2`'s 0.5770 is not a new baseline.** The direction, T, and the decision
  to drop isotropic noise were all made against target-derived metrics, so it is
  **target-informed** - an upper bound on what directed augmentation can do. The
  source-only number remains 0.5556.

#### v9.0 feature generation and `L_MSE` (`docs/v9-feature-generation.md`)

Run `runs/20260923T071856507630Z_baseline_ladder_full`, four cells x five seeds,
pooled SD 0.0062 to 0.0078, threshold 0.0089 to 0.0091. The paper's Eqs. (8)-(16)
block sits between ResNet blocks 3 and 4 (section 2.4: "Rs(.) means the first
three convolutional layers"), with Eq. (S4)'s `L_MSE` added to the objective. All
four cells carry v8.1's `R-aug-t2` augmentation, so the block itself is the only
factor under test and `R-aug-t2` is the control row.

| variant | lambda_MSE | `L_ce` on | target mean | SD | vs `R-aug-t2` |
|---|---:|---|---:|---:|---:|
| `R-gen-m10` | 1.0 | original branch | 0.5776 | 0.0062 | +0.0006 |
| `R-gen` | 0.5 | original branch | 0.5774 | 0.0065 | +0.0004 |
| `R-aug-t2` | - | - | 0.5770 | 0.0078 | - |
| `R-gen-ce2` | 0.5 | both branches | 0.5749 | 0.0062 | -0.0022 |

- **Feature generation is worth +0.0004** - an order of magnitude below section 3's
  0.005 floor and a twentieth of the separability threshold. Every per-batch
  accuracy agrees within 0.01 across the four cells except `R-gen-ce2`'s B9.
- **The block is not inert**, on three independent measurements. It moves `z_f` by
  **0.0448**, inside the 0.047 to 0.105 range of a real two-session drift within
  Batch 1 (per-channel 0.0269, per-scalar 0.0114 - the per-position style axis was
  chosen because it spans 30.1% of the real drift against 15.3% and 1.2%, itself a
  target-informed reading). `L_MSE` falls **2 to 5 times** over 100 epochs. And
  raising lambda_MSE to 1.0, the ceiling Eq. (4) allows, moves the mean by 0.0002.
  So the network does learn the invariance the loss asks for, and that invariance
  buys zero target accuracy.
- **It does not compress the real drift.** Drift over within-class radius: 1.952 at
  the input, 1.824 to 1.841 at `z_f` - the block improves compression from 5.7% to
  6.3-6.5%, leaving **94% of the real drift passing through untouched**.
- **The reason is the sign, not the axis.** Measured at block 3 (16,384 dimensions,
  random-direction cosine 0.0078): the perturbation's |cos| against real drift is
  0.0243 to 0.0360 (**3.1-4.6x** the null) and **7.6%** of its energy lies in the
  51-dimensional drift span against a 0.3% null (**25x**). Its magnitude is 0.950
  source radii against real drift's 1.908 - only **2x** short. But the signed
  component averages **-7.5%** of its typical magnitude (per class +3%, -34%, -45%,
  +23%, -4%, +12%). **Eq. (14)'s Gaussian displaces symmetrically along the drift
  axis**; teaching a classifier to be invariant over a symmetric +/-epsilon
  neighbourhood inflates the decision region evenly and never moves the boundary
  toward where the drifted data actually is. An unsigned perturbation cannot
  correct a signed displacement however well-aimed its axis.
- **This is geometrically the same failure as v8.0's isotropic noise**, and v8.1
  proved it from the other side: undirected -0.0216, the same displacement given a
  direction +0.0115 to +0.0214. Direction is the entire effect, and Eq. (14)
  discards it.
- **The pre-registered prediction hit**: the config predicted a sub-threshold gain
  "because the block moves the drift component the backbone was not removing
  anyway". The recorded falsifier did not trigger, but it compares two quantities
  that are both noise, so it establishes nothing.
- **Independent replication.** `exp/a3-confound-ablation` measured the same
  component at +0.001 to +0.014 on a different backbone; v9.0 narrows it to
  +0.0004 at five seeds with a smaller SD.
- **Confounds.** All four cells carry target-informed augmentation, so the whole
  ladder is an upper bound, not a source-only result. Gradient steps and
  per-sample exposure are matched; only wall-clock differs. Eq. (14)'s draw is
  left untruncated as the paper writes it - the negative-scale fraction is 0.000
  per-position and 0.015 per-channel, so it costs nothing at this depth. The block
  is inactive at inference (Fig. 2), and `src/evaluate.py` scores every frozen
  checkpoint through the plain five-block path.

#### v9.0 post-hoc: the generated cloud never leaves Batch 1 (`docs/v9-feature-generation.md`)

Measured 2026-09-29 on the frozen `R-gen` seed 1042 checkpoint at block 3, with
the artificial cloud generated **from Batch 1** the way training generates it.

- **Along each (gas, batch) drift axis** (source centroid 0, target centroid 1),
  over 51 cells: histogram overlap between artificial and target has median
  **0.00**, the artificial cloud's width is **1.01x** Batch 1's own, and the
  block's paired displacement along the axis has median **0.03** of the drift.
  The six cells with overlap above 0.10 are ones where the target itself spreads
  back over the source. `scripts/make_generation_drift_projection_figure.py`.
- **The "magnitude only 2x short" reading was the full-space norm.** Along any
  single drift axis the shortfall is a median 30x (7x to 140x).
- **The paper's Fig. 5 is circular**: it applies the block to batch 2 and compares
  with batch 2. Our reproduction of that figure matches its shape
  (`reports/figures/feature_generation.png`); the non-circular version
  (`generation_drift_aligned_2d.png`, x = drift direction, y = leading orthogonal
  direction) shows the orange cloud on the grey one and the blue cloud apart.

#### v10.0 a signed Eq. (14) (`docs/v10-signed-generation.md`)

Run `runs/20260929T031927475103Z_baseline_ladder_full`, four cells x five seeds,
pooled SD 0.0069 to 0.0115, threshold 0.0087 to 0.0120. The direction is Batch
1's Ethanol acquisition-block offset measured in the block's style space
(per-position mean and std of the block-3 residual), refreshed every epoch,
never from a target file. All cells carry `R-aug-t2`, lambda_MSE 0.5.

| variant | Eq. (14) | target mean | SD | vs `R-gen` | separable |
|---|---|---:|---:|---:|---|
| `R-gen-shift` | paper's noise + 2 block offsets along the direction | 0.5796 | 0.0077 | +0.0023 | no |
| `R-gen` | paper's symmetric draw | 0.5773 | 0.0069 | - | - |
| `R-aug-t2` | no block | 0.5770 | 0.0078 | -0.0003 | no |
| `R-gen-sign` | half-normal along the direction's sign | 0.5767 | 0.0115 | -0.0006 | no |

- **The sign was right and the magnitude was right, and it still did nothing.**
  Post-hoc, in style space: the direction's cosine with the real drift is **+0.593**
  over 51 cells (null 0.088), positive for all six classes (+0.31 to +0.77), and
  the shift reaches **0.73x to 1.41x** the real drift's magnitude. The recorded
  risk that the direction might be anti-correlated did not materialise.
- **The pre-registered falsifier triggered**: `R-gen-shift` is not separable
  from `R-gen`. Every per-batch and per-class difference is within 0.012, where
  the same directed displacement at the input (v8.1) moved Acetone and Ethylene
  by 0.25 or more.
- **This closes the feature generation block as a mechanism under this protocol,
  regardless of the sign of Eq. (14).** The per-position style space spans 30.1%
  of the real drift and the decision boundary does not respond to displacements
  inside it.
- `R-aug-t2` and `R-gen` re-ran within 0.0003 of their v9.0 values.
- **Confound**: all cells already carry the input-space directed augmentation, so
  the shift is measured on top of a same-direction displacement; the cleaner
  `R-gen-shift` without augmentation against `R-fig-logps` was not run.

#### v12.0 projecting the drift axis out of the input (`docs/v12-drift-projection.md`)

Run `runs/20261005T145146298993Z_baseline_ladder_full`, four cells x five seeds,
pooled SD 0.0088 to 0.0164, threshold 0.0145 to 0.0191. `x' = x - U U^T x` after
the Normal block, `U` from Batch 1's two acquisition sessions, stored in the
checkpoint and applied at inference. No augmentation, no block: the reference is
the settled baseline and the numbers are source-only except `R-proj-eth`.

| variant | removed | k | target mean | SD | vs baseline | separable |
|---|---|---:|---:|---:|---:|---|
| `R-proj-eth` | Ethanol's session offset | 1 | 0.5747 | 0.0119 | +0.0191 | yes, target-informed |
| `R-proj-axis` | the line the three offsets share | 1 | 0.5684 | 0.0088 | +0.0128 | no (threshold 0.0145) |
| `R-proj-sub3` | the span of the three offsets | 3 | 0.5610 | 0.0164 | +0.0055 | no |
| `R-fig-logps` | nothing | 0 | 0.5556 | 0.0137 | - | - |

- **Batch 1's session offset is a usable direction estimate but is not the
  drift.** Its common axis carries 29% of the three-year drift's energy and 42%
  of Batch 1's within-class radius (the within-class streaks run along it); the
  3-dimensional span carries 73% of the drift energy and 76% of the
  between-class variance. In this input space drift and class information share
  two or three effective dimensions and a projection cannot separate them.
- **Both pre-registered falsifiers triggered.** The headline is 0.0017 under its
  threshold, and Acetaldehyde stays at 0.000: its target centroid still lands on
  Batch 1's Ethanol in 7 to 9 batches of 9 under every projection, even `sub3`,
  because after the projection the two are 1.01 radii apart in Batch 1. The dead
  class is not only a shared-axis translation; `docs/why-acetaldehyde.md` is
  incomplete on this point.
- **Removing the axis trades exactly as augmenting along it did**: Acetone +0.24
  to +0.27, Ethylene -0.28 to -0.29, the v8.1 pattern. The proposal's claim that
  projection would not reallocate the prediction quota was wrong. Two unrelated
  mechanisms giving the same ceiling (+0.019 to +0.021, target-informed) and the
  same per-class shape says the ceiling belongs to the data's geometry.
- Source separability is untouched: the step 1 gate (Batch 1 CV with the network,
  `runs/20261005T140110659659Z_drift_projection_cv`) gave 0.9709 against 0.9680.
- **The source-only settled number remains 0.5556.**

#### Standing methodological facts

- **A fixed epoch budget confounds augmentation**, because doubling the rows at a
  fixed epoch count doubles the gradient steps. Any comparison between augmented
  and unaugmented training must state which of step count and per-sample exposure
  it matches; it cannot match both.
- **Two of the paper's three components now measure at about zero under this
  protocol**: augmentation as the paper specifies it (isotropic) at -0.0216, and
  feature generation at +0.0004 as written and +0.0023 with a correctly signed,
  drift-sized displacement (v10.0). **The contrastive loss `L_con` (Eq. S5) is
  implemented (v11.0, `src/loss.py:supervised_contrastive`, variants `R-con`,
  `R-con-shift`, `R-con-t5`, `configs/contrastive.json` with a pre-registered
  prediction) and was deliberately not run**, by the user's decision on
  2026-09-29: the invariance it asks for is to `z_bar_f`, and v9.0 and v10.0
  measured that `z_bar_f` never leaves Batch 1 along the drift axis, so the
  premise is already gone. The previous branch measured it five times at -0.005
  to +0.003. The run is one `gpu-smoke` plus one `launch`, about fifteen
  minutes, if it is ever wanted.
- **A StandardScaler fitted on all ten batches does not explain the paper's
  0.6346.** `docs/leak-diagnostic-scaler.md`, run
  `20261006T025338707996Z_leak_diagnostic_scaler`, deliberately target-informed
  and excluded from every source-only table: `R-fig` goes from 0.4114 (scaler on
  Batch 1) to 0.4351 (scaler on Batches 1-10, values only), +0.024, five seeds
  all under 0.46. The 0.08 gap to the paper's ResNet remains unexplained; the
  explanations left all need more target information than a scaler, or a
  different Batch 1.
- **A drift-over-radius ratio can rise when drift is removed**, if the removal
  shrinks the within-class radius more than the drift (v12.0: 1.80 to 2.46 under
  `R-proj-axis`). Report the numerator and denominator separately.
- **Coverage in a high-dimensional feature space has to be measured along the
  drift axis.** A 2D-PCA hull is blind to the other axes, a full-dimensional
  radius test saturates, and a centroid comparison ignores extent; the three
  disagreed on the same data. `scripts/summarise_run.py` prints a finished run's
  table, per-batch and per-class recall with the separability threshold.

### From the previous branch

`exp/a3-confound-ablation` at commit `95115a2` holds the v6 code, 21 documents
and a review deck; `git checkout exp/a3-confound-ablation -- <path>` retrieves
any of it. Its run outputs are quarantined in `.trash/run-cleanup-20260923/`.
Its best configuration of any kind reached 0.5613, and its three CDCNN
components were each worth approximately zero under this protocol: contrastive
loss -0.005 to +0.003, feature generation +0.001 to +0.014, augmentation -0.013
to -0.009 at every noise scale tested.

## 5. Architecture constraint

The input is a 128-vector. Models consume it as `[N, 1, 128]` with Conv1d, or as
a flat 128-vector. **Never reshape 128 features into a 16x8 or 8x16 image and
apply Conv2d to it.** The 128 features are 8 statistics extracted from each of 16
sensors; the axes are not spatial and a 2D convolution over them mixes unrelated
quantities. Grouping features by sensor with an explicit, documented layout is a
different thing and is allowed, but it must be justified in the run's
documentation and must not be an image reshape.

## 6. Execution safeguards

- **Single worker on CUDA.** GPU launchers enforce one worker at a time; parallel
  workers collide in VRAM.
- **GPU pre-flight gate.** A launcher rejects a full run unless a passing GPU
  smoke artifact exists from the current code.
- **Environment.** Pin torch to a build matching the installed driver. A
  torch 2.8.0+cu126 build against the CUDA 12.2 driver on this machine produced an
  Xid 31 MMU fault that poisoned CUDA machine-wide and required a reboot;
  2.5.1+cu121 is known good. `exp/a3-confound-ablation` carries the Dockerfile.
- **Determinism.** Fix and record every seed. A run records the code hash, config
  hash, library versions, and driver version in its manifest.

## 7. Run hygiene

- Every run writes a unique `runs/<timestamp>_<name>/` directory. **Never
  overwrite or write into an existing run directory.** `runs/` is gitignored, so
  its contents exist in exactly one place.
- **Never delete a run directory.** Move it to `.trash/run-cleanup-<date>/` and
  record what moved and why in `docs/run-cleanup-<date>.md`.
- A run directory is self-describing: config, manifest with input hashes,
  per-seed metrics, the leakage audit, and any failure records stay together.

## 8. Reporting

- State what was measured, on how many seeds, and the spread. A single-seed pilot
  is labelled a pilot.
- Where a result contradicts the paper, say so plainly and cite the run directory.
- Interrupted, failed, and superseded runs are recorded, not quietly dropped.
