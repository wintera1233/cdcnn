#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdcnn_ablation import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment", nargs="?", default="B0",
        choices=["B0", "A1", "A2", "A2-semantic", "A2-paper-literal", "A3"],
    )
    parser.add_argument("--config", default="configs/cdcnn_v6.json")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run all modes on Batch 1 for one truncated validation epoch; never load targets.",
    )
    args = parser.parse_args()
    print(main(args.experiment, args.config, smoke=args.smoke))
