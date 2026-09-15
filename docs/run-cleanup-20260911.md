# Run cleanup record — 2026-09-11

## Scope and disposition

The contents of `runs/` were audited against `AGENTS.md`,
`CDCNN_four_experiment_spec_v6.md`, `configs/cdcnn_v6.json`, and
`implementation_audit_v6.md`. Configuration files, run manifests, failure
records, input dependencies, result hashes, and live processes were checked.

Twelve directories were removed from `runs/`. They were moved to the
recoverable quarantine `.trash/run-cleanup-20260911/`; their contents were not
permanently erased. This keeps obsolete results out of the active run inventory
while preserving evidence in case it is needed for forensic review.

The canonical v6 training settings used for this decision are SGD, learning
rate 0.001, momentum 0.9, weight decay 0.0001, `StepLR(step_size=25,
gamma=0.5)`, batch size 64, exactly 100 epochs without early stopping, and the
five fixed seeds 1042, 2024, 3407, 42, and 123. Canonical A1 uses perturbation
scale 1.0 and variance-correct sampling. Canonical A2 adds the specified
probability-MSE term and corrected style distribution.

## Removed directories

| Directory removed from `runs/` | Classification | Reason |
|---|---|---|
| `20260906T091956Z_resnet` | Wrong settings and inconsistent reporting | Legacy Conv2d diagnostic using an Adam configuration rather than the canonical 1D v6 protocol. Its standalone `summary.json` disagreed with its checkpoint-derived predictions and other metric artifacts. |
| `20260907T184942Z_resnet_validation` | Invalid-chain derivative | Read-only validation derived solely from the removed legacy Conv2d run. It documented the reporting inconsistency but was not an independent experiment. |
| `20260908T132734Z_canonical_1d_resnet_b0` | Wrong settings | Historical Adam B0 used at most 30 epochs, patience-based early stopping, and no locked v6 StepLR schedule; v6 requires the shared fixed SGD/100-epoch protocol. |
| `20260908T133331Z_1d_resnet_confusion_matrices` | Invalid-chain derivative | Figures derived solely from the removed historical Adam B0 predictions. |
| `20260908T163610Z_canonical_1d_resnet_b0_sgd` | Wrong settings | Although it used SGD, it used a 30-epoch maximum with patience 8 and lacked the fixed v6 StepLR schedule and five-seed protocol. |
| `20260908T163948Z_1d_resnet_confusion_matrices` | Invalid-chain derivative | Figures derived solely from the removed pre-v6 SGD B0 predictions. |
| `20260908T165848Z_canonical_1d_resnet_b0_sgd` | Wrong settings and repeated result | Compared no scheduler with `StepLR(step_size=10, gamma=0.1)` under a 30-epoch early-stopping protocol and selected no scheduler, all contrary to v6. Its selected-run batch metrics, confusion matrices, and decompressed target predictions duplicate the earlier SGD B0 result. |
| `20260908T174237Z_formal_a1` | Wrong settings | Pre-v6 A1 used the old training protocol and the A1 implementation that mixed standard deviations directly; the v6 audit identifies the required variance mixing followed by a square root as a blocking correction. |
| `20260908T180205Z_formal_a1_scale_0p5` | Wrong settings | Repeated the pre-v6 A1 workflow with noncanonical perturbation scale 0.5, retained the old sampling semantics, and used 30-epoch early stopping without the required scheduler. |
| `20260908T183933746912Z_formal_a2` | Failed repeated result and wrong settings | Failed its post-run `report_csv_summary_consistency` gate. It was rerun unchanged as the next A2 directory: configuration, model, summary, core metric files, and decompressed CSV payloads are identical. It also used the obsolete pre-v6 A2 method. |
| `20260908T184845032203Z_formal_a2` | Wrong settings | Corrected only the overly strict reporting audit; its own fix log says the method and configuration did not change. The computation still used perturbation scale 0.5, 30-epoch early stopping, no scheduler, the old A2 style rule, and no probability-MSE loss required by v6. |
| `20260909T075808886497Z_b0_a1_a2_value_report` | Invalid-chain derivative | Frozen-artifact comparison whose only experiment inputs were the removed pre-v6 B0, A1, and A2 results; retaining it would present obsolete settings as a usable comparison. |

The two A2 attempts were not merely similar runs. Twenty-three same-named
artifacts had identical SHA-256 hashes, including `configuration.json`,
`model.pt`, `summary.json`, metric tables, histories, and the figure. The four
gzip files that differed at the container-byte level had identical decompressed
payloads. The replacement run's `implementation_fix_log.json` explicitly says
that neither the method nor configuration changed.

## Retained directories

| Directory | Reason retained |
|---|---|
| `20260906T075606Z_pca` | Canonical validation plus global and Batch-1-fitted source-only PCA. It records 13,910 rows, batches 1–10, six labels, 128 features, no exclusions, no nonfinite values, and the required fit scopes. |
| `20260906T080508Z_pca_visual_revision` | Visualization-only derivative of the valid PCA run; it performs no new fitting or sample exclusion. |
| `20260906T085010Z_pca_svm` | Independent, completed historical SVM baseline and the provenance source for the saved Batch 1 fold assignments still used by the current v6 smoke gates. It is outside the active phase but is not invalidated by the v6 CDCNN settings. |
| `20260906T090350Z_svm_no_pca` | Independent, completed historical no-PCA SVM comparison; outside the active phase but not a duplicate and not governed by the v6 CDCNN protocol. |
| `20260910T195333901824Z_cdcnn_v6_batch1_smoke` | Passed target-inaccessible implementation smoke test under the current v6 settings; explicitly not an experiment result. |
| `20260910T204313901041Z_b0_batch1_gpu_smoke` | Passed current-launcher CUDA/device-placement gate; explicitly not an experiment result. |

## Verification and preservation

- `runs/` contains exactly the six retained directories listed above.
- The quarantine contains exactly the twelve removed directories.
- No matching training or report-generation process was running before the
  move.
- Four A1/A2 source directories had read-only directory mode. Owner write
  permission was added only to those directory entries so they could be moved;
  artifact file contents were not modified.
- `Dataset/` was not moved, renamed, or written. All ten current raw-file
  SHA-256 values match the primary PCA run's recorded `input_manifest.json`.
- No retained run directory was overwritten or modified.

## Follow-up GPU-smoke invalidation

During the one-seed pilot launch preparation, GPU smoke run
`20260911T075630147230Z_b0_batch1_gpu_smoke` passed against the configuration
but was invalidated when `scripts/run_cdcnn_v6_full.py` gained the gated
one-seed pilot orchestration path. Because the launcher implementation hash no
longer matched, the run was moved from `runs/` to the recoverable quarantine
`.trash/run-cleanup-20260911/`. It was not deleted or overwritten.

The replacement `20260911T075932355204Z_b0_batch1_gpu_smoke` was generated
after that change and passed against the final launcher and unchanged
configuration. It is the smoke artifact referenced by the completed
`20260911T075952177964Z_cdcnn_v6_one_seed_pilot` run.
