#!/usr/bin/env python3
"""Run the Batch-1-tuned no-PCA RBF-SVM baseline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.svm_baseline import main

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/svm.json")
args = parser.parse_args()
print(main(args.config))
