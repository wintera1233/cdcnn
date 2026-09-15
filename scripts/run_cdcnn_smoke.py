#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdcnn_ablation import run_smoke_suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cdcnn_v6.json")
    args = parser.parse_args()
    print(run_smoke_suite(args.config))
