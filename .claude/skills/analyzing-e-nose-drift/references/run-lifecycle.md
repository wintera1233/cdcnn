# Run lifecycle and operational safety

Read this before creating, monitoring, retrying, deriving from, or quarantining a run.

## Authorization boundary

- Read-only inspection, tests that do not train, config validation, and artifact recomputation do not authorize training.
- A request to review or modify code does not authorize a smoke, pilot, full run, or target evaluation.
- Run a Batch-1 smoke, GPU smoke, pilot, or full experiment only when the user explicitly requests that class of execution. If scope is ambiguous, ask.
- Never invoke internal `source-worker`, `controller`, `pilot-controller`, or `evaluate` commands manually as a shortcut.

## Preflight for any approved CDCNN training

1. Re-read `AGENTS.md`, `configs/cdcnn_v6.json`, and the current launcher help/code.
2. Confirm CUDA availability and that no conflicting training process owns the intended GPU.
3. Run `.venv/bin/python -m unittest -q tests.test_cdcnn_v6`.
4. Confirm the saved Batch 1 fold file exists and passes identity/duplicate checks through the canonical loader.
5. Confirm the raw hashes in the config match without writing to `Dataset/`.
6. Confirm `git status --short`; record code/config identities and do not mix unexplained edits into provenance.
7. Ensure the new output path does not exist. Let canonical scripts create it with exclusive creation.

## Approved canonical sequence

After explicit training approval, create a current B0 CUDA gate:

```bash
source .venv/bin/activate
python scripts/run_cdcnn_v6_full.py gpu-smoke --config configs/cdcnn_v6.json
```

The artifact is usable only if it passed, opened Batch 1 only, proves CUDA/PID placement, matches the current config hash, and matches current hashes of `src/cdcnn_ablation.py` and `scripts/run_cdcnn_v6_full.py`. Any relevant edit invalidates it.

For an explicitly requested one-seed pilot:

```bash
python scripts/run_cdcnn_v6_full.py pilot-launch \
  --config configs/cdcnn_v6.json \
  --seed <one-of-the-five-canonical-seeds> \
  --max-workers 1 \
  --gpu-smoke-run runs/<passing-current-gpu-smoke>
```

For an explicitly requested canonical full experiment:

```bash
python scripts/run_cdcnn_v6_full.py launch \
  --config configs/cdcnn_v6.json \
  --max-workers 1 \
  --gpu-smoke-run runs/<passing-current-gpu-smoke>
```

Do not substitute stage wrappers, `src` module entry points, historical configs, altered seed lists, or parallel CUDA workers. A pilot is not a canonical five-seed result and cannot replace it silently.

## Monitoring detached orchestration

The launch command creates a run directory, starts a detached controller, and records its PID/log. Monitor without modifying artifacts:

- inspect `launch_manifest.json` for PID and log path;
- check that exact PID before assuming the process is alive;
- tail the recorded controller log at reasonable intervals;
- inspect `controller_result.json`, failure artifacts, and source selections as they appear;
- do not infer failure merely from temporary lack of log output;
- do not start a competing retry while the controller is alive.

The controller owns bounded retries. If it exhausts attempts, targets must remain unopened. Any manual retry or recovery requires a fresh explicit request and a new run directory unless the canonical controller itself is continuing its recorded attempt sequence.

## Completion criteria

Do not call a run successful until:

- the controller status is completed;
- all expected source checkpoints exist and hash-verify;
- the leakage/target-access audit passes;
- dataset validation passes with no exclusions, clipping, deduplication, or imputation;
- source and target sample-level prediction coverage is complete and unique;
- metrics and confusion matrices recompute exactly from predictions;
- failures are accounted for;
- reports, configuration, software/code identities, access log, and output inventory are present.

If an audit fails, describe the run as failed or invalid even if model checkpoints exist.

## Derived analyses

Plots, report corrections, and frozen inference also write a new timestamped run. They must hash their upstream inputs, identify the upstream run, state that raw data/upstream artifacts were not modified, record any fitting scope, and save a manifest plus output inventory. Never patch an old CSV, JSON, model, or report in place.

## Cleanup and quarantine

Before moving a run out of `runs/`:

1. Resolve the exact directory; never use a broad glob as the destructive target.
2. Check controller/worker PIDs and relevant live processes.
3. Inspect manifests, completion/failure state, hashes, and upstream/downstream dependencies.
4. Decide whether the run is invalid, obsolete, superseded, duplicate, or still needed as provenance.
5. Move it intact to `.trash/run-cleanup-<date>/` without overwriting another entry.
6. Add a directory-specific rationale and preservation checks to `docs/run-cleanup-<date>.md`.
7. Reverify raw dataset hashes and retained-run integrity when the cleanup is material.

Never permanently delete a run or modify `Dataset/` during cleanup.
