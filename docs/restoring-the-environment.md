# Restoring the environment

The project is paused at v12.0 (2026-10-06). Nothing in the repository depends on
the local virtual environment or the container image being present; both can be
rebuilt from the files below.

## Two environments, two jobs

| Environment | Used for | Pinned by |
|---|---|---|
| Container `cdcnn:cu121` | every training run, smoke and cross-validation on the GPU | `docker/Dockerfile`, `docker/requirements-cu121.txt` |
| `.venv` (Python 3.11) | post-hoc analysis, figures, decks, unit tests, on the CPU only | `requirements-venv-lock.txt` |

The `.venv` carries torch 2.8.0+cu126. On this machine's CUDA 12.2 driver that
build produced an Xid 31 fault that needed a reboot (CLAUDE.md section 6), so
every command that uses it sets `CUDA_VISIBLE_DEVICES=` and never touches the GPU.

## Rebuild

```bash
# the training container
docker build -t cdcnn:cu121 -f docker/Dockerfile .

# the analysis environment
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-venv-lock.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

`docs/run-commands.md` lists the commands for every version from v7.0 to v12.0.

## What a run needs that is not in git

- `Dataset/batch{1..10}.dat`: gitignored, immutable, 23 MB. Keep it.
- `runs/`: gitignored. Each run directory holds its manifest (config, code hashes,
  git revision, library versions), per-seed `history.json`, `target_results.json`,
  `target_summary.json`, the leakage audit and `checkpoint_digests.json`. These
  records are the trace behind every number in `docs/`. **The checkpoint files
  were removed on 2026-10-06** to shrink the folder from 23 GB to 264 MB
  (`docs/checkpoint-cleanup-20261006.json`). They are regenerable: a run is
  deterministic given its manifest, and a four-cell, five-seed run takes about
  twelve minutes on this GPU. Post-hoc scripts that load a `final.pt`
  (the generation-coverage figures, `measure_signed_direction.py`, the drift
  compression measurements) need the run they name re-trained first.
