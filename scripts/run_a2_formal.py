#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cdcnn_ablation import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cdcnn_v6.json")
    parser.add_argument(
        "--paper-literal", action="store_true",
        help="Run the predeclared A2 paper-literal diagnostic instead of canonical A2-semantic.",
    )
    args = parser.parse_args()
    print(main("A2-paper-literal" if args.paper_literal else "A2-semantic", args.config))
