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

Batch 1 class counts: Acetone 90, Acetaldehyde 98, Ethanol 83, Ethylene 30,
Ammonia 70, Toluene 74.

Label mapping, verified against the paper's Table 2 in 8 of 10 batches (the two
mismatches are errors in the paper, not in the loader):

`1 = Acetone, 2 = Acetaldehyde, 3 = Ethanol, 4 = Ethylene, 5 = Ammonia, 6 = Toluene`

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

## 4. What the previous attempt established

Branch `exp/a3-confound-ablation` at commit `95115a2` holds the v6 code, 21
documents, and the review deck; `git checkout exp/a3-confound-ablation -- <path>`
retrieves any of it. Its run outputs are quarantined in
`.trash/run-cleanup-20260923/`. Do not spend GPU time re-deriving these:

- **The deficit is in the baseline.** The paper's plain ResNet baseline (0.6346)
  beats every stage that project ever trained, best 0.5613. No CDCNN component
  explains the gap.
- **Per-sample input normalization was worth +0.109**, LayerNorm instead of
  BatchNorm +0.085, the two together +0.149. These were the only real gains.
- **The three CDCNN components were worth approximately zero** under this
  protocol: supervised contrastive loss −0.005 to +0.003, feature generation
  +0.001 to +0.014, augmentation −0.013 to −0.009 at every noise scale tested.
- **Ethylene was never predicted**, by any stage or by either SVM baseline. It is
  the smallest class in Batch 1 at 30 rows. The paper's own Fig. S3 confusion
  matrix has a class with a zero diagonal.
- **A fixed epoch budget confounds augmentation**, because doubling the rows at a
  fixed epoch count doubles the gradient steps. Any comparison between augmented
  and unaugmented training must state which of step count and per-sample exposure
  it matches; it cannot match both.

Closing the baseline gap comes before adding any domain-generalization component.

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
