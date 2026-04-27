from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PHASE_NAME = "visibility"
PHASE_ARTIFACT_STEM = "02_visibility"


@dataclass(frozen=True, slots=True)
class VisibilityArtifacts:
    los_pairs: NDArray[np.int32]
    diagonal_blocked_pairs: NDArray[np.int32]


__all__ = [
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "VisibilityArtifacts",
]
