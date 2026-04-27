"""Phase 04 optimization artifact containers for solver outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PHASE_NAME = "optimization"
PHASE_ARTIFACT_STEM = "04_solution"


@dataclass(frozen=True, slots=True)
class OptimizationArtifacts:
    """Selected camera configurations and final per-open-cell DORI scores."""

    selected_configuration_indices: NDArray[np.int32]
    final_open_cell_scores: NDArray[np.int8]


__all__ = [
    "OptimizationArtifacts",
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
]
