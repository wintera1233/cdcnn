#!/usr/bin/env python3
"""Audit frozen B0/A1 predictions, confidence, entropy, and FC128 norms."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prediction_collapse_audit import main


if __name__ == "__main__":
    print(main())
