#!/usr/bin/env python3
"""Disabled launcher retained only to reject the banned legacy 2D path."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.resnet_baseline import Legacy2DFrameworkDisabled, main

if __name__ == "__main__":
    try:
        main()
    except Legacy2DFrameworkDisabled as exc:
        raise SystemExit(f"ERROR: {exc}") from None
