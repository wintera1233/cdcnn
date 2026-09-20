"""Shared protocol exception for the CDCNN v6 pipeline.

`ProtocolError` lives in its own module so that stage modules and the shared
pipeline can raise the same exception type without importing each other.
"""

from __future__ import annotations


class ProtocolError(RuntimeError):
    """Raised when a v6 invariant or access boundary is violated."""
