"""Configuration loading and validation.

A configuration names the variants it declares in advance. Validation refuses a
variant the code does not implement and refuses a configuration whose optimizer
constants drift from the ones `proposal.md` section 4.3 fixes for the ladder,
because changing them mid-ladder would confound every comparison in it.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.model import VARIANTS
from src.protocol import ProtocolError

LADDER_VERSION = "CDCNN_v7.0_baseline_ladder"

# Frozen literals: the declared contents of the ladder. Adding a variant later
# must not silently change what this version means.
LADDER_VARIANTS = ("R-txt", "R-lite", "R-lite-ps")
LADDER_SEEDS = (1042, 2024, 3407)
LADDER_OPTIMIZER = {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001}
LADDER_SCHEDULER = {"step_size": 25, "gamma": 0.5}
LADDER_TRAINING = {"epochs": 100, "batch_size": 64}

REQUIRED = ("implementation_version", "variants", "seeds", "training", "optimizer",
            "scheduler")


def validate(config: dict) -> dict:
    missing = [key for key in REQUIRED if key not in config]
    if missing:
        raise ProtocolError(f"configuration is missing {missing}")

    unknown = [name for name in config["variants"] if name not in VARIANTS]
    if unknown:
        raise ProtocolError(
            f"undeclared variant(s) {unknown}; implemented: {sorted(VARIANTS)}")
    if len(set(config["variants"])) != len(config["variants"]):
        raise ProtocolError("a variant is listed twice")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ProtocolError("a seed is listed twice")

    if config["implementation_version"] == LADDER_VERSION:
        for name, expected, actual in (
                ("variants", list(LADDER_VARIANTS), list(config["variants"])),
                ("seeds", list(LADDER_SEEDS), list(config["seeds"])),
                ("training", LADDER_TRAINING, config["training"]),
                ("optimizer", LADDER_OPTIMIZER, config["optimizer"]),
                ("scheduler", LADDER_SCHEDULER, config["scheduler"])):
            if expected != actual:
                raise ProtocolError(
                    f"{LADDER_VERSION} fixes {name} as {expected}, got {actual}")
    return config


def load(path: Path) -> dict:
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))
