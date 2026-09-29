# Running the baseline ladder

All training runs inside the pinned container. `--user` matters: without it the
container writes run directories as `root`, which is how eleven directories from
2026-09-21 became unmovable without elevated privileges.

```bash
DOCKER="docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $PWD:/workspace -w /workspace cdcnn:cu121"

# tests
$DOCKER python -m unittest discover -s tests

# CPU smoke (omit --gpus all so torch sees no device)
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/workspace \
  -w /workspace cdcnn:cu121 \
  python scripts/run_baseline.py smoke --config configs/baseline_ladder.json --epochs 3

# GPU smoke: the pre-flight gate artifact
$DOCKER python scripts/run_baseline.py gpu-smoke \
  --config configs/baseline_ladder.json --epochs 3

# the full ladder
$DOCKER python scripts/run_baseline.py launch \
  --config configs/baseline_ladder.json --max-workers 1 \
  --gpu-smoke-run runs/<passing_gpu_smoke>
```

`launch` refuses to start unless the named GPU smoke passed on CUDA **and** its
manifest's code hashes match the current files. Editing any module in
`src/audit.py:CODE_FILES` invalidates the gate, so re-run `gpu-smoke` after a
code change.

## v9.0, the feature generation block

```bash
$DOCKER python scripts/run_baseline.py gpu-smoke \
  --config configs/feature_generation.json --epochs 3

$DOCKER python scripts/run_baseline.py launch \
  --config configs/feature_generation.json --max-workers 1 \
  --gpu-smoke-run runs/<passing_gpu_smoke>
```

`src/augment.py` and `src/generate.py` joined `src/audit.py:CODE_FILES` with
this version, so the gate now invalidates on a change to either. `augment.py`
had been outside it since v8.0, which means the v8 gates did not cover the one
module those runs were testing.

## v10.0, the signed Eq. (14)

```bash
$DOCKER python scripts/run_baseline.py gpu-smoke \
  --config configs/signed_generation.json --epochs 3

$DOCKER python scripts/run_baseline.py launch \
  --config configs/signed_generation.json --max-workers 1 \
  --gpu-smoke-run runs/<passing_gpu_smoke>

# after the run
.venv/bin/python scripts/summarise_run.py runs/<run> --reference R-gen
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/measure_signed_direction.py
```

A full four-cell, five-seed run takes about twelve minutes on this GPU. Post-hoc
scripts run in `.venv` on the CPU only: its torch 2.8.0+cu126 must not touch CUDA
on this driver (section 6 of CLAUDE.md).
