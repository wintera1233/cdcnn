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

- `Dataset/batch{1..10}.dat`: gitignored, immutable, 23 MB. **Removed on
  2026-10-06 at the user's decision**, together with the container image. See
  "Restoring the dataset" below; no run, smoke or post-hoc script works until
  it is back.
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

## Restoring the dataset

The ten files are the UCI Machine Learning Repository's *Gas Sensor Array Drift
Dataset* (Vergara et al., 2012), ten batches in LIBSVM format, 13,910 rows in
all. Download it from the repository, place the files as
`Dataset/batch1.dat` ... `Dataset/batch10.dat`, and check every one against the
table below before running anything. The same hashes are in the
`dataset_sha256` field of every run's `manifest.json`; a file that does not
match is not the data these results were measured on.

| file | bytes | rows | sha256 |
|---|---:|---:|---|
| `batch1.dat` | 752,726 | 445 | `f346beee8e0c5e31ac5961845b6d96a70dc1ccf799481592fb2f0d96a81952e6` |
| `batch2.dat` | 2,119,601 | 1,244 | `07f7e94a9bf4377240f9b230d20c750932fc785d4766b6a4b7fc370a035c825a` |
| `batch3.dat` | 2,706,554 | 1,586 | `a4c7a1a6744df32f0ca139c75c3b4dade7cc99f57a7785d28cb0f62577dd2061` |
| `batch4.dat` | 276,473 | 161 | `bd76f86be34ffe89f46a27c6fc2b5b8c6ebfcf984dff4b3e5befc76d98034b7f` |
| `batch5.dat` | 339,035 | 197 | `3f95cb6e4a39a94bbacd2f1984f1754c9fb5eb3221812975ad70edbcea7abaec` |
| `batch6.dat` | 3,902,170 | 2,300 | `83348c504105a5aa5264d1209f2a5d7ba7d9c8bcae60290395f201ac4708ff95` |
| `batch7.dat` | 6,117,334 | 3,613 | `3168cb56d5c9bc29c36184e2e73c9f3474adb3c4a893b1ce0d88b6c47698cfdb` |
| `batch8.dat` | 491,904 | 294 | `296346b932893ea18c513e23ac5b660f35e1c499fde8254d4804a2ce8a1b4ca7` |
| `batch9.dat` | 785,870 | 470 | `e019f11f4fa8336ab1f41f7503eb456b3679da0a84eb8032fe6cef90f0547826` |
| `batch10.dat` | 6,045,679 | 3,600 | `30011067c7c05c2b01038f84c73af0c6f1820b622fa2b0ad89111cb4c649f791` |

```bash
sha256sum Dataset/batch*.dat      # compare with the table
python -m unittest discover -s tests   # the data tests assert the row and label counts
```

Once restored, section 1 of `CLAUDE.md` applies again in full: nothing under
`Dataset/` is modified, renamed, moved, deleted or clipped.

## Restoring the container

`docker build -t cdcnn:cu121 -f docker/Dockerfile .` as above. The image removed
on 2026-10-06 was `b5a6be388758`, 5.86 GB; a rebuild resolves the same pinned
versions from `docker/requirements-cu121.txt`.
