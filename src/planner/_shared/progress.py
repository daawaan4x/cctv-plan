"""Small structural protocol for live progress writers."""

from __future__ import annotations

from typing import Protocol


class ProgressWriter(Protocol):
    """Minimal text-stream surface needed by planner progress reporting."""

    def write(self, message: str, /) -> int: ...

    def flush(self) -> None: ...


__all__ = ["ProgressWriter"]
