"""Configuration loading and validation.

A configuration names the variants it declares in advance. Validation refuses a
variant the code does not implement and refuses a configuration whose optimizer
constants drift from the ones `proposal.md` section 4.3 fixes for the ladder,
because changing them mid-ladder would confound every comparison in it.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.model import RETIRED_VARIANTS, VARIANTS
from src.protocol import ProtocolError

LADDER_VERSION = "CDCNN_v7.0_baseline_ladder"
CHANNEL_VERSION = "CDCNN_v7.1_channel_restore"
HEAD_NORM_VERSION = "CDCNN_v7.2_head_normalisation"

# Frozen literals: the declared contents of the ladder. Adding a variant later
# must not silently change what this version means.
LADDER_VARIANTS = ("R-txt", "R-txt-ps", "R-lite", "R-lite-ps")
LADDER_SEEDS = (1042, 2024, 3407)
LADDER_OPTIMIZER = {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0001}
LADDER_SCHEDULER = {"step_size": 25, "gamma": 0.5}
LADDER_TRAINING = {"epochs": 100, "batch_size": 64}

# v7.1 keeps every protocol constant and changes only the backbone widths, so
# the new cells compare directly against R-txt and R-txt-ps of the v7.0 run.
# Normal is held at per-sample: v7.0 measured StandardScaler as -0.093 under
# both heads, so keeping it would spend half the runs re-confirming a loss.
CHANNEL_VARIANTS = ("R-txt-ps", "R-fig-ps")
CHANNEL_LEARNING_RATES = (0.001, 0.0003, 0.0001)

# v7.2 fixes Fig. 2's widths and lr 0.0003 and crosses the head's normalisation
# with the input's. Parameter counts are identical across all four cells.
HEAD_NORM_VARIANTS = ("R-fig", "R-fig-ln", "R-fig-ps", "R-fig-ps-ln")
HEAD_NORM_LEARNING_RATES = (0.0003,)

REQUIRED = ("implementation_version", "variants", "seeds", "training", "optimizer",
            "scheduler")


def validate(config: dict) -> dict:
    missing = [key for key in REQUIRED if key not in config]
    if missing:
        raise ProtocolError(f"configuration is missing {missing}")

    retired = [name for name in config["variants"] if name in RETIRED_VARIANTS]
    if retired and config["implementation_version"] != LADDER_VERSION:
        raise ProtocolError(
            f"retired variant(s) {retired}: global average pooling was measured at "
            "-0.078 target mean and withdrawn; see docs/baseline-ladder.md")

    unknown = [name for name in config["variants"] if name not in VARIANTS]
    if unknown:
        raise ProtocolError(
            f"undeclared variant(s) {unknown}; implemented: {sorted(VARIANTS)}")
    if len(set(config["variants"])) != len(config["variants"]):
        raise ProtocolError("a variant is listed twice")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ProtocolError("a seed is listed twice")

    if config["implementation_version"] == HEAD_NORM_VERSION:
        for name, expected, actual in (
                ("variants", list(HEAD_NORM_VARIANTS), list(config["variants"])),
                ("learning_rates", list(HEAD_NORM_LEARNING_RATES),
                 [float(v) for v in config.get("learning_rates", [])]),
                ("seeds", list(LADDER_SEEDS), list(config["seeds"])),
                ("training", LADDER_TRAINING, config["training"]),
                ("scheduler", LADDER_SCHEDULER, config["scheduler"])):
            if expected != actual:
                raise ProtocolError(
                    f"{HEAD_NORM_VERSION} fixes {name} as {expected}, got {actual}")

    if config["implementation_version"] == CHANNEL_VERSION:
        for name, expected, actual in (
                ("variants", list(CHANNEL_VARIANTS), list(config["variants"])),
                ("learning_rates", list(CHANNEL_LEARNING_RATES),
                 [float(v) for v in config.get("learning_rates", [])]),
                ("seeds", list(LADDER_SEEDS), list(config["seeds"])),
                ("training", LADDER_TRAINING, config["training"]),
                ("optimizer", LADDER_OPTIMIZER, config["optimizer"]),
                ("scheduler", LADDER_SCHEDULER, config["scheduler"])):
            if expected != actual:
                raise ProtocolError(
                    f"{CHANNEL_VERSION} fixes {name} as {expected}, got {actual}")

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
