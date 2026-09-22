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
