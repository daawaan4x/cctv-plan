from __future__ import annotations

from typing import Final

PHASE_NAME = "optimization"
PHASE_ARTIFACT_STEM = "04_solution"
PHASE_PRECOMPUTE_ARTIFACT_STEM = "04_precompute"

_DORI_LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4)
_LARGE_THRESHOLD_MEMBERSHIP_WARNING: Final[int] = 100_000_000
