"""Protocol errors and the target-access recorder.

Kept free of other project imports so every module can raise ProtocolError
without creating an import cycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ProtocolError(RuntimeError):
    """A source-only protocol rule was violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class TargetAccessLog:
    """Records the first time each target file is opened.

    `CLAUDE.md` section 2 requires that no target batch is read until every
    checkpoint of a run is written and frozen. This class records the evidence;
    `src.audit` checks it. A run that never constructs one cannot read a target
    batch at all, because `src.data.load_target` demands one.
    """

    def __init__(self) -> None:
        self._first_access: dict[str, str] = {}
        self._freeze_events: list[dict[str, str]] = []

    def record_access(self, path: Path) -> None:
        key = str(path)
        self._first_access.setdefault(key, utc_now())

    def record_freeze(self, checkpoint: Path, digest: str) -> None:
        if self._first_access:
            raise ProtocolError(
                f"cannot freeze {checkpoint} after target files were opened: "
                f"{sorted(self._first_access)}")
        self._freeze_events.append(
            {"checkpoint": str(checkpoint), "sha256": digest, "frozen_at": utc_now()})

    @property
    def accesses(self) -> dict[str, str]:
        return dict(self._first_access)

    @property
    def freezes(self) -> list[dict[str, str]]:
        return [dict(event) for event in self._freeze_events]

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"target_first_access": self.accesses, "checkpoint_freezes": self.freezes},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")
