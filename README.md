# Exploratory PCA for the UCI Gas Sensor Array Drift Dataset

The repository covers exploratory PCA and dataset validation, two retained historical SVM baselines, and the CDCNN v6.3 drift experiments. A completed five-seed v6.3 four-stage run is retained (`20260911T092326995583Z_cdcnn_v6_3_full`), and a completed A3 confound ablation (`20260915T025701880293Z_cdcnn_v6_3_a3_confound_full`) shows that A3's target-accuracy gain comes from its LayerNorm swap rather than from contrastive learning. Retained results are not authorization to train or tune further; each new training run needs an explicit request. Baseline settings are project settings, not verified settings from the paper.

## Environment

Python 3.11.2 and a project-local virtual environment are in use:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The direct dependencies are pinned in `requirements.txt`; the complete installed environment is recorded in `docs/environment.txt`.

## Dataset and configuration

The immutable, already-extracted raw files remain in `Dataset/`. Analysis settings and the project-relative dataset path are in `configs/pca.json`. Initial format and structure findings, including input hashes, are in `docs/dataset-inspection.md`.

## Layout

- `src/`: reusable parsing, validation, preprocessing, and plotting logic
- `scripts/`: runnable entry points
- `configs/`: dataset paths and analysis settings
- `docs/`: dataset and environment documentation
- `reports/`: curated summaries
- `runs/<unique_run_id>/`: uniquely named run outputs; retained runs are never overwritten
- `data/derived/`: derived datasets
- `data/cache/`: extracted or temporary dataset cache

## Retained run inventory

The following table is the complete inventory of retained directories under `runs/`. "Historical" means the artifact remains interpretable and may be used as frozen provenance, but it is outside the active validation/PCA phase. Smoke tests are implementation checks, not result-bearing experiments.

| Run directory | Status and role | Why it is retained |
|---|---|---|
| `20260906T075606Z_pca` | Complete; primary dataset validation, global PCA, and Batch-1-fitted source-only PCA | Canonical exploratory PCA result and upstream source for the visualization revision |
| `20260906T080508Z_pca_visual_revision` | Complete; visualization-only PCA derivative | Curated full/zoom PCA views and batch×gas shift summaries; depends on the primary PCA run |
| `20260906T085010Z_pca_svm` | Complete; historical PCA–SVM baseline | Supplies the frozen Batch 1 fold assignments referenced by later historical configurations |
| `20260906T090350Z_svm_no_pca` | Complete; historical no-PCA SVM comparison | Direct, saved-fold comparison with the PCA–SVM baseline |
| `20260910T195333901824Z_cdcnn_v6_batch1_smoke` | Passed; Batch-1-only v6 mode smoke test | Confirms all five executable modes run without opening target batches; not an experiment result |
| `20260910T204313901041Z_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Pre-v6.3 launcher GPU placement evidence; not an experiment result |
| `20260911T075932355204Z_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate for the v6 one-seed pilot |
| `20260911T075952177964Z_cdcnn_v6_one_seed_pilot` | Completed; one-seed pilot | Pre-v6.3 pilot; not a five-seed result |
| `20260911T085949697200Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | First v6.3 gate; superseded by the 0903 gate |
| `20260911T090046807325Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only v6.3 mode smoke | Five-mode implementation check; not an experiment result |
| `20260911T090335466807Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate used by the completed v6.3 full run |
| `20260911T090400010314Z_cdcnn_v6_3_one_seed_pilot` | Completed; one-seed v6.3 pilot | Seed-42 pilot preceding the full run; not a five-seed result |
| `20260911T092326995583Z_cdcnn_v6_3_full` | **Completed; canonical five-seed v6.3 four-stage experiment** | Primary CDCNN result: B0, A1, A2-semantic, A3 over seeds 1042/2024/3407/42/123 |
| `20260911T100408906245Z_cdcnn_v6_3_result_plots` | Complete; plotting derivative | Figures for the full run |
| `20260911T103411776649Z_GAS4_predict` | Completed; inference only | Frozen-checkpoint inference diagnostic on gas 4 |
| `20260912T103502820938Z_cdcnn_v6_3_accuracy_bars` | Complete; plotting derivative | Accuracy bar figures for the full run |
| `20260915T025623543707Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only mode smoke | Eight-mode check including the confound stages |
| `20260915T025641199413Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate for the A3 confound ablation |
| `20260915T025701880293Z_cdcnn_v6_3_a3_confound_full` | **Completed; six-stage A3 confound ablation** | Attributes A3's gain: LayerNorm +0.0845, contrastive loss +0.0029 target mean |
| `GAS4_predict` | Completed; inference only | Non-timestamped legacy inference directory |

### Run cleanup record

On 2026-09-10, three non-retained directories were removed from `runs/` after checking their manifests, logs, implementation hashes, dependencies, and live processes. They were moved to the recoverable local quarantine `.trash/run-cleanup-20260910/` rather than permanently erased:

- `20260909T075721362542Z_b0_a1_a2_value_report`: failed during manifest creation (`NameError`) and was superseded by the completed `20260909T075808886497Z_b0_a1_a2_value_report`.
- `20260910T200804526033Z_cdcnn_v6_full`: incomplete, no longer running, had no run manifest, used an obsolete configuration, and launched four concurrent CUDA workers instead of the current sequential protocol.
- `20260910T204237549960Z_b0_batch1_gpu_smoke`: passed but was immediately superseded by `20260910T204313901041Z_b0_batch1_gpu_smoke`, whose recorded launcher hash matches the current launcher.

This cleanup did not modify `Dataset/` or any retained run. Future analyses must write a new unique run directory and must not overwrite a retained run. Failed or superseded runs should only be removed after confirming that no retained artifact depends on them and recording the removal here.

On 2026-09-11, twelve legacy ResNet/A1/A2 directories were removed from
`runs/` as an obsolete pre-v6 settings chain, repeated result, or derivative of
that chain. They were moved to the recoverable quarantine
`.trash/run-cleanup-20260911/`. The complete directory-by-directory rationale,
duplicate evidence, retained inventory, and raw-data verification are recorded
in [`docs/run-cleanup-20260911.md`](docs/run-cleanup-20260911.md).

## PCA fit scopes

Global PCA is exploratory: one `StandardScaler` and one PCA are fit on the combined Batch 1–10 feature matrix. Source-only PCA fits both the scaler and PCA on Batch 1 and uses those fitted objects to transform every batch. It must never fit a separate scaler on a target batch. Labels and other metadata do not enter either feature matrix.

## Historical PCA–SVM baseline

The following command documents how the retained baseline was produced. Do not run it during the current validation/PCA phase unless classifier work is explicitly requested.

Run the leakage-safe Batch 1 baseline with:

```bash
source .venv/bin/activate
python scripts/run_pca_svm.py --config configs/pca_svm.json
```

The command searches a `StandardScaler → PCA → RBF-SVM` pipeline using duplicate-group-aware, five-fold stratified CV on Batch 1, refits the selected pipeline on all Batch 1 samples, and evaluates Batches 2–10 without using them for tuning. Each invocation writes a new `runs/<timestamp>_pca_svm/` directory and does not reuse or overwrite PCA outputs.

## Historical no-PCA SVM baseline

The following command is retained for provenance and reproduction outside the current phase.

Run the 128-feature baseline with:

```bash
source .venv/bin/activate
python scripts/run_svm.py --config configs/svm.json
```

This reuses the saved Batch 1 fold assignments from the PCA–SVM run, searches a `StandardScaler → RBF-SVM` pipeline, and writes a new `runs/<timestamp>_svm_no_pca/` directory with target predictions and a direct PCA–SVM comparison.

## CDCNN v6.3 four-stage protocol

Smoke and gate directories are implementation checks, not results. The
completed five-seed four-stage experiment is
`20260911T092326995583Z_cdcnn_v6_3_full`; its measured stage means are recorded
in the specification's final comparison table. The v6.3 A3 stabilization changes
and selected hard bounds are documented in
[`docs/cdcnn-v6.3-numerical-stabilization.md`](docs/cdcnn-v6.3-numerical-stabilization.md),
and the deviations from the specification's printed A3 definition are listed in
its "v6.3 Implementation Status and Deviations" section.
Do not launch this protocol unless training is explicitly approved.

The canonical B0, A1, A2-semantic, and A3 implementations now share one code
path and one locked configuration: SGD (learning rate 0.001, momentum 0.9,
weight decay 1e-4), `StepLR(step_size=25, gamma=0.5)`, batch size 64, and exactly
100 epochs without early stopping. Training is locked to `cuda:0`; model,
input, label, first-batch, and `nvidia-smi` PID evidence is written for every
source-training context. A2-semantic restyles the pooled/upsampled
low-frequency-like branch. The separately named A2-paper-literal diagnostic
restyles the residual branch printed as L in the paper; it is never substituted
silently for canonical A2.

In v6.3, A3 alone uses LayerNorm without BatchNorm running state, bounded
Log-Normal sigma/residual/contrastive values, anchor-mean contrastive loss, and
gradient clipping at max norm 1.0. B0, A1, and A2 behavior and all shared
training settings remain unchanged.

The stage-specific wrappers are not permitted full-training interfaces. After
explicit training approval and before launching the complete four-stage run,
execute a new focused B0 CUDA gate through the canonical launcher (older smoke
hashes are invalid after the v6.3 code/config update):

```bash
source .venv/bin/activate
python scripts/run_cdcnn_v6_full.py gpu-smoke --config configs/cdcnn_v6.json
```

After that command reports a passing run directory, pass it explicitly to the
full launcher. CUDA source jobs are intentionally sequential fresh Python
subprocesses; `--max-workers` must be 1, and no fork-based CUDA multiprocessing
is used.

```bash
python scripts/run_cdcnn_v6_full.py launch \
  --config configs/cdcnn_v6.json \
  --max-workers 1 \
  --gpu-smoke-run runs/<passing_b0_batch1_gpu_smoke>
```

### A3 confound ablation

Because v6.3 gave A3 LayerNorm, hard bounds, and gradient clipping alongside the
contrastive loss, an A3-minus-A2 difference cannot be credited to contrastive
learning by itself. The confound ablation adds the diagnostic stages `B0-LN`,
`B0-stab`, and `A2-stab` (A3 without the contrastive loss) and re-runs B0,
A2-semantic, and A3 as same-environment references. It uses its own
configuration and the same launcher, and is documented in
[`docs/a3-confound-ablation.md`](docs/a3-confound-ablation.md):

```bash
python scripts/run_cdcnn_v6_full.py gpu-smoke --config configs/cdcnn_v6_3_a3_confound.json
python scripts/run_cdcnn_v6_full.py launch \
  --config configs/cdcnn_v6_3_a3_confound.json \
  --max-workers 1 \
  --gpu-smoke-run runs/<passing_b0_batch1_gpu_smoke>
```

The completed ablation attributes A3's target-mean gain to the LayerNorm swap
(+0.0845, 5/5 seeds), not to the contrastive loss (+0.0029). `B0-LN` is the
strongest stage on both Batch 1 CV and target accuracy. Full tables are in
[`docs/a3-confound-ablation.md`](docs/a3-confound-ablation.md).

Runs reproduce only within one software environment: the 2026-09-11 run used
torch 2.5.1+cu121 and the 2026-09-15 ablation uses torch 2.8.0+cu126, which
reproduced four of five B0 seeds exactly and differed by one validation sample in
the fifth. Compare stages only within a single run.

The older `src/resnet_1d_baseline.py`, `src/a1_formal.py`, and
`src/a2_formal.py` modules are retained as historical pre-v6 implementation
evidence. Their public wrapper scripts route to the unified v6 implementation.
The obsolete result directories themselves were quarantined on 2026-09-11 as recorded in
[`docs/run-cleanup-20260911.md`](docs/run-cleanup-20260911.md).

## Banned legacy two-dimensional framework

The pre-v6 one-dimensional B0 compared `StepLR(step_size=10, gamma=0.1)`
against a no-scheduler control using Batch-1-only CV. The earlier Conv2d
baseline reshaped inputs to 16×8. Both result families used obsolete settings
and have been removed from `runs/`; their recoverable copies are under
`.trash/run-cleanup-20260911/`.

The legacy 16×8 model implementation has been removed from active source. Its
old launcher and configuration are disabled tombstones with no training
settings, and invoking the launcher raises an error. A repository test forbids
two-dimensional convolution constructors in active Python source. The only
supported full-training entry point is `scripts/run_cdcnn_v6_full.py` with
`configs/cdcnn_v6.json`, or `configs/cdcnn_v6_3_a3_confound.json` for the
confound ablation; it must not be launched unless training is explicitly
requested.
