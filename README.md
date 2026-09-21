# Exploratory PCA for the UCI Gas Sensor Array Drift Dataset

The repository covers exploratory PCA and dataset validation, two retained historical SVM baselines, and the CDCNN v6.3 drift experiments. A completed five-seed v6.3 four-stage run is retained (`20260911T092326995583Z_cdcnn_v6_3_full`), a completed A3 confound ablation (`20260915T025701880293Z_cdcnn_v6_3_a3_confound_full`) showing that A3's target-accuracy gain comes from its LayerNorm swap rather than from contrastive learning, and a completed input-normalization ladder whose best stage `B0-LN-PS` reaches 0.5569 target mean with no CDCNN component at all. The narrative of how the reproduction went is in
[`docs/development-journal.md`](docs/development-journal.md), the consolidated
Batches 2-10 results for every approach in
[`docs/all-results.md`](docs/all-results.md), and a running record of changes in
[`docs/change-log.md`](docs/change-log.md). Retained results are not authorization to train or tune further; each new training run needs an explicit request. Baseline settings are project settings, not verified settings from the paper.

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
| `20260920T165723700086Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only mode smoke | Twelve-mode check after the A3 code split |
| `20260920T172614291493Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only mode smoke | Twelve-mode check including the v6.4 input stages |
| `20260920T172700916776Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate for the v6.4 input-normalization run |
| `20260920T172721243924Z_cdcnn_v6_4_input_norm_full` | Source phase complete; target phase failed | All 30 checkpoints frozen and evaluated; report generation raised `KeyError` and the attempt is retained as evidence |
| `20260920T181354711407Z_cdcnn_v6_4_input_norm_eval` | **Completed; v6.4 input-normalization evaluation** | Best result in the project: `B0-LN-PS` 0.5569 target mean, 0.9933 Batch 1 CV |
| `20260920T182017231402Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only mode smoke | Fifteen-mode check including the v6.5 ladder stages |
| `20260920T182044094409Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate for the v6.5 normalized-input ladder |
| `20260920T182057522943Z_cdcnn_v6_5_normalized_ladder_full` | **Completed; four-stage normalized-input ladder** | CDCNN components remain neutral-to-negative once inputs are normalized |
| `20260920T191600735756Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only mode smoke | Eighteen-mode check including the v6.6 stages |
| `20260920T191622505529Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate for the v6.6 paper-literal ladder |
| `20260920T191634939156Z_cdcnn_v6_6_paper_literal_full` | **Completed; paper-literal and augmentation ladder** | Augmentation alone −0.0131; the paper's restyled branch beats the project's semantic branch on 5/5 seeds |
| `20260920T201453177056Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only mode smoke | Twenty-one-mode check including the v6.7 scale stages |
| `20260920T201510387823Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate for the v6.7 augmentation-scale sweep |
| `20260920T201524271480Z_cdcnn_v6_7_augmentation_scale_full` | **Completed; augmentation-scale sweep** | Noise magnitude barely matters: scales 0.05–0.5 all stay ~0.01 below no augmentation |
| `20260920T204952623150Z_cdcnn_v6_3_batch1_smoke` | Passed; Batch-1-only mode smoke | Twenty-two-mode check including the duplication control |
| `20260920T205025054689Z_v6_3_b0_batch1_gpu_smoke` | Passed; Batch-1-only CUDA gate | Gate for the v6.8 duplication control |
| `20260920T205038327586Z_cdcnn_v6_8_duplication_control_full` | **Completed; duplication control** | Exact duplicates cost the same as augmentation: the A1 penalty is a doubled-schedule artefact |
| `20260921T014945430161Z_cdcnn_v6_9_epoch_aligned_full` | Failed; GPU fault | Five anchor checkpoints completed, then an Xid 31 MMU fault killed every aligned task; retained as evidence ([`docs/gpu-fault-20260921.md`](docs/gpu-fault-20260921.md)) |
| `20260921T145852844886Z_cdcnn_v6_9_epoch_aligned_full` | **Completed; epoch-aligned ladder (container)** | Step alignment recovers about a third of the augmentation penalty; CDCNN components become mildly positive and seed variance drops |
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

### Input normalization (v6.4)

Per-sample input normalization, applied after the Batch-1 scaler, is the
strongest single change measured here: +0.1087 target mean on 5/5 seeds, and
+0.1492 when combined with LayerNorm. Bounding extreme values instead
(signed-log, clip) does nothing. Configuration
`configs/cdcnn_v6_4_input_norm.json`; details and tables in
[`docs/input-normalization.md`](docs/input-normalization.md).

### Contrastive placement (v6.10)

Applying the contrastive loss where Fig. 2 puts it — on the pre-FC128 latent
with no learned head — leaves it worth +0.0002, against +0.0025 for this
project's post-FC128 learned head. Neither is distinguishable from zero, and
the placement was not the explanation. Configuration
`configs/cdcnn_v6_10_contrastive_placement.json`; details in
[`docs/contrastive-placement.md`](docs/contrastive-placement.md).

### Epoch alignment (v6.9)

Aligned stages draw one source-sized subset of the augmented pool per epoch, so
they take the same optimizer and scheduler steps as an un-augmented stage.
Matching steps recovers about a third of the augmentation penalty; the remainder
reflects halved per-sample exposure, which cannot be held constant at the same
time. Under alignment the CDCNN components stop being negative and seed variance
drops sharply. Not adopted as canonical; see
[`docs/epoch-alignment.md`](docs/epoch-alignment.md). Training now runs in the
pinned container (`docker/Dockerfile`); see
[`docs/gpu-fault-20260921.md`](docs/gpu-fault-20260921.md).

### Duplication control (v6.8) — protocol issue

Exact duplicates (`perturbation_scale = 0.0`) cost −0.0114, indistinguishable
from canonical augmentation's −0.0131. Augmented stages train on 890 rows rather
than 445 under the fixed 100-epoch schedule, so they take twice the optimizer
steps; that, not the generated data, explains the A1 penalty. **Every A1-versus-B0
comparison here is confounded by schedule length.** Resolving it changes the
training protocol, so nothing was changed; options are in
[`docs/duplication-control.md`](docs/duplication-control.md).

### Augmentation scale sensitivity (v6.7)

Reducing the A1 noise magnitude twentyfold changes target accuracy by +0.0026 and
leaves augmentation ~0.01 below the un-augmented backbone, so the penalty is not
about how much noise is added. Configuration
`configs/cdcnn_v6_7_augmentation_scale.json`; details in
[`docs/augmentation-scale.md`](docs/augmentation-scale.md).

### Paper-literal and augmentation ladder (v6.6)

Isolating A1 augmentation shows it is the component that hurts (−0.0131), and
restyling the branch the paper actually restyles — the residual, its L — beats
this project's semantic pooled-branch choice on 5/5 seeds (+0.0137 for A2,
+0.0201 for A3). Configuration `configs/cdcnn_v6_6_paper_literal.json`; details
in [`docs/paper-literal-ladder.md`](docs/paper-literal-ladder.md).

### Normalized-input ladder (v6.5)

Repeating each v6.3 ablation step on per-sample-normalized inputs shows the
CDCNN components are still neutral-to-negative: augmentation plus feature
generation −0.0224, contrastive loss −0.0049. Input conditioning was not what
held them back. Configuration `configs/cdcnn_v6_5_normalized_ladder.json`;
details in [`docs/normalized-input-ladder.md`](docs/normalized-input-ladder.md).

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
