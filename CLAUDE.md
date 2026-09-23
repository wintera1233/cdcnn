# Project Instructions for AI Agents

This branch (`exp/v7-redesign`) restarts the study from the baseline. It holds the
raw data, the source paper, and this file; no model code exists yet. The rules
below constrain how any new code must behave. They are protocol, not method: they
say nothing about which architecture, optimizer, or training schedule to use,
because those are what the redesign is for.

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

The paper's Table 2 per-gas counts match this file in only one column under that
mapping. The conflict is recorded, not resolved; the counts per label are the
fact and the names are the interpretation. Changing the mapping changes no
measurement, because only integer labels reach the model.

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

### From this branch (v7.0 to v7.3, `docs/`)

- **The baseline stands at target mean 0.5222** against the paper's 0.6344, with
  six of the nine batches at or above the paper and the whole deficit in B5, B8
  and B9. `docs/all-results.md` lists every cell trained.
- **Per-sample input normalisation is worth +0.112**; it and a LayerNorm head are
  substitutes with an interaction of -0.141, so use exactly one, at the input.
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
- **A fixed epoch budget confounds augmentation**, because doubling the rows at a
  fixed epoch count doubles the gradient steps. Any comparison between augmented
  and unaugmented training must state which of step count and per-sample exposure
  it matches; it cannot match both.

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
