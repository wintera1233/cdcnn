#!/usr/bin/env python
"""Leak diagnostic: does a scaler fitted on all ten batches give the paper's 0.63?

The paper's "Normal" block is undefined. A common way to get a number like its
ResNet's 0.6346 without meaning to is to fit a StandardScaler on the whole
dataset - every batch, source and target - before splitting. This run tests
that reading directly, and is therefore **target-informed by construction**:
the target files are opened before any model is trained, the scaler sees
their values (not their labels), and the run cannot pass the leakage audit.
It does not call `audit.freeze` or `audit.leakage_audit`; it writes its own
`leak_diagnostic.json` saying what was leaked and why.

Two cells, five seeds each, the Fig. 2 widths and the settled schedule:

    R-fig, scaler on Batch 1      the source-only control (v7.2 measured 0.4105)
    R-fig, scaler on all batches  the leak under test

Nothing from this run may enter a source-only table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import audit, normalize  # noqa: E402
from src.config import ALL_SEEDS, LADDER_SCHEDULER, LADDER_TRAINING  # noqa: E402
from src.data import TARGET_BATCHES, load_source, load_target  # noqa: E402
from src.evaluate import aggregate, evaluate_checkpoint  # noqa: E402
from src.protocol import TargetAccessLog, utc_now  # noqa: E402
from src.train import train_one  # noqa: E402

VARIANT = "R-fig"          # Fig. 2 widths, flatten head, StandardScaler Normal
LEARNING_RATE = 0.0003


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ALL_SEEDS))
    parser.add_argument("--epochs", type=int, default=LADDER_TRAINING["epochs"])
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise SystemExit("CUDA required")

    config = {
        "implementation_version": "CDCNN_leak_diagnostic_scaler",
        "description": __doc__.strip(),
        "leak_by_design": True,
        "variants": [VARIANT], "learning_rates": [LEARNING_RATE],
        "seeds": args.seeds,
        "training": {**LADDER_TRAINING, "epochs": args.epochs},
        "optimizer": {"lr": LEARNING_RATE, "momentum": 0.9, "weight_decay": 0.0001},
        "scheduler": LADDER_SCHEDULER,
        "cells": {"R-fig@batch1-scaler": "StandardScaler fitted on Batch 1 (source-only control)",
                  "R-fig@all-batch-scaler": "StandardScaler fitted on Batches 1-10, values only"},
        "reference": {"paper ResNet, Table 3": 0.6346,
                      "R-fig@lr0.0003 v7.2 (3 seeds, Batch 1 scaler)": 0.4105,
                      "R-fig-logps@lr0.0003 v7.5 (settled source-only)": 0.5556},
    }
    run_dir = audit.new_run_dir("leak_diagnostic_scaler")
    audit.write_manifest(run_dir, config)

    # The leak, done openly: every target file is opened before training and the
    # access is logged. Labels are never read into the scaler.
    access_log = TargetAccessLog()
    source_x, source_y = load_source()
    targets = [load_target(b, access_log)[0] for b in TARGET_BATCHES]
    everything = np.concatenate([source_x] + targets, axis=0)
    opened_at = access_log.accesses
    scaler_batch1 = normalize.fit("standard_scaler", source_x)
    scaler_all = normalize.fit("standard_scaler", everything)

    results = []
    cells = [("batch1-scaler", scaler_batch1), ("all-batch-scaler", scaler_all)]
    for tag, params in cells:
        for seed in args.seeds:
            name = f"{VARIANT}_{tag}_seed{seed}"
            target = run_dir / "checkpoints" / name
            print(f"{VARIANT} {tag} seed {seed}", flush=True)
            summary = train_one(VARIANT, seed, config, source_x, source_y, target, device,
                                save_every_epoch=False, learning_rate=LEARNING_RATE,
                                normalizer_params=params)
            print(f"    train acc {summary['final_train_accuracy']:.4f}", flush=True)
            result = evaluate_checkpoint(target / "final.pt", access_log, device)
            result["cell"] = f"{VARIANT}@{tag}"
            results.append(result)
            print(f"    target mean {result['target_mean']:.4f}", flush=True)

    summary = aggregate(results)
    (run_dir / "target_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (run_dir / "target_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "leak_diagnostic.json").write_text(json.dumps({
        "status": "leak_by_design",
        "what_leaked": "the StandardScaler's per-feature mean and standard deviation were "
                       "fitted on the values of Batches 1-10 before training; no target label "
                       "was used anywhere",
        "why": "to test whether the paper's undefined Normal block, fitted on the whole "
               "dataset, accounts for its ResNet's 0.6346",
        "target_files_opened_at": opened_at,
        "training_started_after_target_access": True,
        "may_enter_source_only_tables": False,
        "finished_at": utc_now(),
    }, indent=2) + "\n")

    print(f"\n{'cell':<26}{'source':>9}{'target mean':>13}{'sd':>8}")
    for cell, entry in summary.items():
        print(f"{cell:<26}{entry['source_accuracy']:>9.4f}{entry['target_mean']:>13.4f}"
              f"{entry['target_mean_sd']:>8.4f}")
    print(f"\npaper ResNet 0.6346; settled source-only 0.5556\n{run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
