#!/usr/bin/env python3
"""Free the space checkpoints take under runs/, keeping every run's records.

The project is paused at v12.0. `runs/` holds about 17 GB of checkpoint files
and 17 MB of records. This removes the checkpoint files only. Every run
directory stays, with its manifest, config, training summaries, target results,
leakage audit, per-seed history.json and checkpoint_digests.json, so each
number in docs/ still traces to its run and every removed file's sha256 stays
on record. No checkpoint is kept, by the user's decision on 2026-10-06; a
post-hoc figure or measurement that needs a model needs its run re-trained.

**Dry run by default.** It prints what it would remove and writes nothing.
Pass `--execute` to hash the files that have no recorded digest, write
`docs/checkpoint-cleanup-20261006.json`, and delete. Standard library only, so
it runs without the project's virtual environment.

This deviates from CLAUDE.md section 7 in the same way
`docs/checkpoint-cleanup-20260923.json` did: files inside retained runs are
removed, no run directory is. Moving them to `.trash/` would free nothing on
the same filesystem.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

# The user chose on 2026-10-06 to remove every checkpoint, including the eight
# seed-1042 final.pt the post-hoc scripts load; those scripts need the run
# re-trained first.
KEEP: list[str] = []
RECORD = "docs/checkpoint-cleanup-20261006.json"


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--execute", action="store_true",
                        help="actually delete; without it nothing is changed")
    args = parser.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    missing = [k for k in KEEP if not os.path.exists(k)]
    if missing:
        print("refusing: a file on the keep list is missing:", *missing, sep="\n  ")
        return 2
    runs = sorted(d for d in glob.glob("runs/*") if os.path.isdir(d))
    plan, per_run = [], {}
    for run in runs:
        files = [p for p in sorted(glob.glob(run + "/**/*.pt", recursive=True))
                 if p not in KEEP]
        if not files:
            continue
        has_digests = os.path.exists(run + "/checkpoint_digests.json")
        per_run[run] = {"files_removed": len(files),
                        "bytes_freed": sum(os.path.getsize(p) for p in files),
                        "sha256_recorded_in": ("checkpoint_digests.json" if has_digests
                                               else RECORD + ", sha256_of_files_without_digests"),
                        "_files": files, "_hash_here": not has_digests}
        plan += files
    total = sum(entry["bytes_freed"] for entry in per_run.values())

    print(f"{'run':<58}{'files':>7}{'GB':>8}")
    for run, entry in per_run.items():
        print(f"{run:<58}{entry['files_removed']:>7}{entry['bytes_freed'] / 1e9:>8.2f}")
    print(f"\n{len(plan)} checkpoint files, {total / 1e9:.2f} GB, in {len(per_run)} of "
          f"{len(runs)} run directories; {len(KEEP)} files kept.")
    if not args.execute:
        print("\nDry run: nothing was changed. Re-run with --execute to delete.")
        return 0

    hashed = {}
    for entry in per_run.values():
        if entry["_hash_here"]:
            for path in entry["_files"]:
                hashed[path] = sha256(path)
    record = {
        "removed_at": "2026-10-06",
        "what": "every checkpoint under runs/; none kept, by the user's explicit choice",
        "why": "the project is paused at v12.0 and the folder was 23 GB, 17 GB of it "
               "checkpoints; the user asked to keep the code trace and the reports only. "
               "Moving the files to .trash would not free space on the same filesystem.",
        "kept": KEEP, "files_removed": len(plan), "bytes_freed": total,
        "deviation": "CLAUDE.md section 7 forbids deleting a run directory. This removes "
                     "checkpoint files inside retained runs, as "
                     "docs/checkpoint-cleanup-20260923.json did. Every run directory and "
                     "all of its records are untouched.",
        "how_to_regenerate": "each run is deterministic given its manifest's config, seeds "
                             "and code hashes: rebuild the container from docker/Dockerfile, "
                             "check out the manifest's git revision and re-run the config "
                             "with scripts/run_baseline.py gpu-smoke then launch; about "
                             "twelve minutes per four-cell run.",
        "per_run": {run: {k: v for k, v in entry.items() if not k.startswith("_")}
                    for run, entry in per_run.items()},
        "sha256_of_files_without_digests": hashed,
    }
    with open(RECORD, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
        handle.write("\n")
    for path in plan:
        os.remove(path)
    left = sorted(glob.glob("runs/**/*.pt", recursive=True))
    if left != sorted(KEEP):
        print("unexpected checkpoints remain:", *left, sep="\n  ")
        return 1
    if sorted(d for d in glob.glob("runs/*") if os.path.isdir(d)) != runs:
        print("a run directory disappeared; this script never removes one")
        return 1
    print(f"\nremoved {len(plan)} files, {total / 1e9:.2f} GB. Record: {RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
