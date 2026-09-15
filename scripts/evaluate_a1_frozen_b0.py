#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.a1_frozen_b0_evaluation import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/a1_frozen_b0_evaluation.json")
    args = parser.parse_args()
    print(main(args.config))
