"""Disabled tombstone for the banned legacy two-dimensional framework."""

from __future__ import annotations


class Legacy2DFrameworkDisabled(RuntimeError):
    """Raised whenever the removed legacy training path is invoked."""


def main(*_args, **_kwargs):
    raise Legacy2DFrameworkDisabled(
        "The legacy 16x8 two-dimensional ResNet is banned. "
        "Use scripts/run_cdcnn_v6_full.py with configs/cdcnn_v6.json."
    )
