#!/usr/bin/env python3
"""Run validated global and Batch-1-source PCA analyses."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pca_analysis import main

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/pca.json")
args = parser.parse_args()
output = main(args.config)
print(output)
