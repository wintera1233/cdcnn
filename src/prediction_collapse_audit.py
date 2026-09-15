"""Disabled tombstone for an audit that depended on the banned 2D checkpoint."""

from __future__ import annotations

from src.resnet_baseline import Legacy2DFrameworkDisabled


def main(*_args, **_kwargs):
    raise Legacy2DFrameworkDisabled(
        "This audit depended on the banned legacy 16x8 checkpoint and is disabled."
    )
