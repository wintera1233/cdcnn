# Analysis, artifacts, and reporting

## Exploratory PCA

`scripts/run_pca.py --config configs/pca.json` creates two explicitly different views:

- Global exploratory PCA fits one scaler and PCA on all Batches 1–10. It describes the supplied dataset and is not a leakage-safe predictive pipeline.
- Source-only PCA fits both objects on Batch 1 and applies them unchanged to Batches 2–10. Never fit separate target scalers.

Keep labels, batch IDs, line numbers, and sample IDs outside the feature matrix. Retain duplicates and extremes. Report nonfinite or zero-variance validation failures rather than repairing raw inputs automatically. A global all-batch scaler is acceptable only in this clearly labeled descriptive PCA path, never for predictive model selection or CDCNN preprocessing.

Interpret PCA as evidence of multivariate distribution shift, not proof that sensor drift is the unique cause. Gas composition is a confounder, two PCs omit later variance, and unequal batch sizes influence an ordinary global fit. Do not infer unavailable concentration-matched behavior.

## Result metrics

Keep these distinct:

- Source CV accuracy: mean over the persisted Batch 1 folds, normally summarized per seed and across seeds.
- Target mean accuracy: unweighted mean of the nine Batch 2–10 accuracies; this is the main cross-batch summary.
- Target pooled accuracy: sample-weighted accuracy across all target rows; report separately because batch sizes differ sharply.
- Per-batch accuracy and confusion matrices: necessary to expose heterogeneous drift/generalization behavior.

When comparing stages, prefer paired seed-wise deltas because the same five seeds and folds are used. Do not select a stage from target performance and then present that selection as source-only model choice. Paper numbers are reference benchmarks, not expected reproductions.

## Audit an existing run before interpreting it

For a canonical full run, review in this order:

1. `controller_result.json`: completed status and source/target phase outcome.
2. `orchestration_config.json` and `launch_manifest.json`: stages, seeds, implementation version, config hash, worker count, smoke dependency, command, and PID.
3. `global_freeze_manifest.json` and `source_run_selections.json`: exact checkpoint coverage and hashes.
4. `leakage_target_access_audit.json`: every check must pass.
5. `dataset_validation.json` and `data_access_log.json`: expected dimensions, no exclusions/repairs, Batch 1-only pre-freeze, and each target loaded post-freeze.
6. `failure_records.json` and `source_failure_records.json`: distinguish zero failures, recovered attempts, and unresolved failures.
7. `source_cv_predictions.csv.gz` and `target_predictions.csv.gz`: uniqueness and expected coverage.
8. Recompute `source_cv_metrics.csv`, `target_batch_metrics.csv`, confusion counts, target mean, and pooled accuracy from sample-level predictions before trusting summaries.
9. `stage_summary.csv`, `per_seed_summary.csv`, `per_batch_summary.csv`, and the report only after the underlying checks pass.

For a source worker, additionally inspect `model_freeze_manifest.json`, `scaler_fit_scopes.csv`, training histories, augmentation provenance, feature geometry/style statistics, device placement evidence, and the checkpoint metadata.

Never edit an upstream run to fix a report. Write a new timestamped derivative run, record upstream paths and hashes, state that no fitting occurred, and include a manifest/inventory.

## Plotting and narrow inference

Use `scripts/plot_cdcnn_v6_full_results.py <full-run>` for current full-run confusion and accuracy figures, or `scripts/plot_cdcnn_v6_3_accuracy_bars.py <full-run>` for five-seed bars. Both should read upstream artifacts and write new unique output directories.

Before using a plotting or inference helper:

- inspect its expected filenames and schema;
- verify that the upstream run type matches;
- verify predictions/metrics first;
- preserve stage and gas-label order;
- label aggregation over seeds and error-bar convention (`ddof=1` for reported across-seed sample SD);
- record upstream hashes and avoid in-place writes.

Treat `scripts/predict_a3_v6_3_batch2_gas4.py` as a narrowly scoped frozen-checkpoint analysis, not a general evaluation entry point. Generalize it only under an explicit implementation request with preserved freeze and provenance checks.

## Scientific writing

In every report, distinguish:

- paper-supported architecture/objective concepts;
- project-controlled numerical choices;
- physical-semantic adaptations such as low/high branch naming;
- observed results from a named validated run;
- limitations and unresolved ambiguity.

Avoid causal language unsupported by the dataset. Say that behavior is consistent with distribution shift, not that PCA or accuracy changes prove a unique physical drift mechanism. Always identify the exact run, experiment kind, seed count, fit scope, and whether target data were available at the decision point.
