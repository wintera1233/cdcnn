---
name: analyzing-e-nose-drift
description: Reviews, analyzes, modifies, validates, runs, and reports on this UCI Gas Sensor Array Drift e-nose repository, including PCA, CDCNN v6.3, experiment configs, run artifacts, and scientific documentation. Use for e-nose data inspection, experiment design or execution, leakage audits, model changes, results interpretation, plotting, or project documentation; not for unrelated ML repositories.
---

# E-nose drift project

Work from the repository root. This skill supplies project workflow; it never grants permission to train a model, read target batches early, or mutate protected artifacts.

## Begin every task

1. Read `AGENTS.md` completely. Its constraints are mandatory even if another file or this skill disagrees.
2. Read the relevant portion of `README.md`, then inspect the actual files and manifests involved in the request. Status prose and run inventories can lag behind the filesystem.
3. Check `git status --short` before editing. Preserve unrelated user changes and never modify generated or raw artifacts in place.
4. Classify the task as read-only review/reporting, implementation change, exploratory PCA, plotting/derived analysis, smoke/pilot training, full training, or run cleanup. Do not broaden its authorization.

## Non-negotiable safeguards

- Treat `Dataset/batch1.dat` through `Dataset/batch10.dat` as immutable. Never move, rename, edit, delete, clip, impute, deduplicate, or silently exclude their rows.
- Treat each record as one gas label plus 128 extracted features. Do not invent concentration, timestamp, humidity, device-ID, or other metadata. Never include labels or provenance columns in feature matrices.
- For predictive CDCNN work, Batch 1 is the sole pre-freeze source for scaler fitting, CV, tuning, training, augmentation statistics, feature-style statistics, and checkpoint decisions. Batches 2–10 are final evaluation only after all applicable source checkpoints are frozen.
- Do not use target metrics, PCA appearance, or target-batch behavior to choose preprocessing, hyperparameters, checkpoints, exclusions, or retries.
- Never restore or introduce `Conv2d` gas classification or a 128-to-16x8 reshape. The canonical tensor layout is `[N, 1, 128]` with `Conv1d`.
- Never overwrite a run. New computation or derived figures go to a unique `runs/<UTC timestamp>_<name>/` directory with provenance.
- Never delete obsolete or failed runs. Move them to `.trash/run-cleanup-<date>/` only after dependency/process checks, and document the rationale in `docs/run-cleanup-<date>.md`.
- Do not launch any training unless the user explicitly requests it. Full training is allowed only through `scripts/run_cdcnn_v6_full.py launch` with `--max-workers 1`, a current passing GPU-smoke artifact, and either `configs/cdcnn_v6.json` (canonical four stages) or `configs/cdcnn_v6_3_a3_confound.json` (A3 confound ablation; see `docs/a3-confound-ablation.md`).

## Route to the right reference

- Read [references/project-map.md](references/project-map.md) before choosing among current, historical, stale, disabled, or one-off files.
- Read [references/cdcnn-v6-3.md](references/cdcnn-v6-3.md) for CDCNN implementation, configuration, loss, augmentation, leakage, or test work.
- Read [references/analysis-and-reporting.md](references/analysis-and-reporting.md) for PCA, metrics, plots, result comparison, scientific claims, or artifact review.
- Read [references/run-lifecycle.md](references/run-lifecycle.md) before any smoke, pilot, full run, monitoring, retry, postprocessing run, or cleanup operation.

Read only the references relevant to the task, but always apply the safeguards above.

## Review and change discipline

- Ground findings in exact files and line numbers. Separate current defects from historical or quarantined behavior.
- For a code/config change, trace the invariant through `configs/cdcnn_v6.json`, `validate_config`, the launcher, model/training code, tests, and affected documentation. Do not weaken a guard merely to make a proposed config pass.
- Preserve distinctions among paper-supported facts, project-controlled choices, semantic adaptations, and empirical findings.
- Prefer existing parsers, fold validators, audit functions, and provenance patterns over parallel implementations.
- Run the narrowest relevant checks first. For CDCNN protocol changes, run `.venv/bin/python -m unittest -q tests.test_cdcnn_v6`; use broader checks only when the changed surface requires them.
- Report what was inspected, changed, validated, and not run. Never describe a smoke test, one-seed pilot, historical baseline, or paper benchmark as a canonical five-seed result.
