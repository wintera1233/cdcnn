# Project map and authority

Use this map to avoid treating every retained file as a current interface.

## Authority and recency

1. `AGENTS.md` is the hard project policy.
2. The user's scoped request determines what actions are authorized; training, cleanup, and other mutations require explicit scope.
3. For the current CDCNN executable contract, read together:
   - `configs/cdcnn_v6.json`
   - `src/cdcnn_ablation.py`, especially `validate_config` and `DataAccessGuard`
   - `scripts/run_cdcnn_v6_full.py`
   - `tests/test_cdcnn_v6.py`
   - `docs/cdcnn-v6.3-numerical-stabilization.md`
4. `CDCNN_four_experiment_spec_v6.md` is the scientific protocol baseline. The v6.3 document, current config, code, and tests supersede its older A3 numerical details where they differ.
5. Run-local configuration, manifests, audits, selections, predictions, and reports describe that exact run and take precedence over a top-level status summary.

When evidence still conflicts, stop and present the conflict instead of silently choosing a scientifically material interpretation.

## Dataset and shared utilities

- `Dataset/batch{1..10}.dat`: immutable LIBSVM-like raw files; 13,910 rows total, labels 1–6, 128 indexed features.
- `configs/pca.json`: dataset path, gas mapping, and PCA settings.
- `src/pca_analysis.py`: canonical parser plus exploratory global and Batch-1-fitted PCA workflow.
- `src/cv_folds.py`: validates saved Batch 1 folds, sample identities, labels, line numbers, full coverage, and duplicate-group isolation.
- `runs/20260906T085010Z_pca_svm/cv_fold_assignments.csv`: persisted source-fold dependency used by v6.3. Verify it through `load_folds`; do not regenerate casually.

Gas mapping: 1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene.

## Current CDCNN surfaces

- `configs/cdcnn_v6.json`: only canonical CDCNN configuration.
- `src/cdcnn_ablation.py`: unified B0, A1, A2-semantic, A2-paper-literal diagnostic, and A3 implementation.
- `scripts/run_cdcnn_v6_full.py`: only permitted full-training launcher. Its public user-facing commands are `gpu-smoke`, `pilot-launch`, and `launch`; internal controller/worker/evaluate commands belong to orchestration.
- `tests/test_cdcnn_v6.py`: executable contract for config locks, target access, 1D architecture, stage semantics, A3 stabilization, and scheduler behavior.
- `scripts/run_cdcnn_smoke.py` and `scripts/run_cdcnn_ablation.py --smoke`: Batch-1-only implementation smoke paths, not experiments.

Stage wrappers such as `scripts/run_resnet_1d.py`, `scripts/run_a1_formal.py`, and `scripts/run_a2_formal.py` route into unified code, but they are not permitted full-training interfaces under `AGENTS.md`.

## Current analysis and derivative surfaces

- `scripts/run_pca.py`: validated exploratory PCA; writes a new run.
- `scripts/revise_pca_visualizations.py`: read-only reuse of a PCA run, writing a new derivative run.
- `scripts/plot_cdcnn_v6_full_results.py`: v6.3 full-run confusion matrices and accuracy-by-batch plots.
- `scripts/plot_cdcnn_v6_3_accuracy_bars.py`: v6.3 five-seed accuracy bars.
- `scripts/predict_a3_v6_3_batch2_gas4.py`: narrow frozen-checkpoint inference analysis. Inspect its input checks and exact scope before reuse or generalization.

## Historical, stale, or disabled material

- `configs/resnet.json`, `scripts/run_resnet.py`, and `src/resnet_baseline.py` are disabled tombstones for the banned 2D path.
- `configs/resnet_1d.json`, `configs/a1_formal*.json`, and `configs/a2_formal.json` contain pre-v6 settings such as 30 epochs, early stopping, CPU runs, old schedulers, or scale 0.5. Keep for provenance; never use them for a canonical run.
- `src/resnet_1d_baseline.py` supplies active `ResidualBlock1D`/reshape utilities but its historical standalone training workflow is not canonical. `src/a1_formal.py` and `src/a2_formal.py` are historical implementations.
- `docs/new-experiment-settings-review.md` is a pre-v6 design review with obsolete 2D/Adam assumptions and unresolved questions later settled by v6/v6.3.
- `implementation_audit_v6.md` and `leakage_target_access_audit_v6.md` are useful historical audits, not live status. In particular, v6's contrastive sum was changed to a mean-over-anchors reduction in v6.3.
- `README.md` and `docs/cdcnn-v6.3-numerical-stabilization.md` contain dated status statements. The filesystem now includes later smoke, pilot, full-run, plotting, and inference artifacts. Verify live state rather than repeating "not started" claims.
- `scripts/compare_cdcnn_ablation.py` and `scripts/generate_b0_a1_a2_value_report.py` hard-code obsolete or quarantined pre-v6 runs. Do not run them for current v6.3 conclusions.
- `scripts/audit_prediction_collapse.py` is disabled through its tombstone implementation.
- `scripts/evaluate_a1_frozen_b0.py` and `scripts/run_augmentation_sanity.py` reference missing modules/configs in the current tree. Treat them as orphaned until deliberately repaired under a separate request.
- Generic `scripts/plot_batch_confusion_matrices.py` expects an older artifact schema (`predictions.csv.gz` and `configuration.json`); use the v6.3 plotting scripts for a v6.3 full run.

## Inspecting live state

Do not choose a run by lexicographic recency alone. Enumerate candidate directories, then inspect at least their `controller_result.json` or `run_manifest.json`, resolved configuration, audit status, input/implementation hashes, and failure records. A later derivative run is not a newer model run.

The repository currently contains a completed canonical full-run example at `runs/20260911T092326995583Z_cdcnn_v6_3_full`, but verify its manifests and audits before citing it and do not assume it remains the preferred run forever.
