from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PHASE_NAME = "candidate_generation"
PHASE_ARTIFACT_STEM = "01_candidates"


@dataclass(frozen=True, slots=True)
class CandidateGenerationArtifacts:
    open_cell_indices: NDArray[np.int32]
    candidate_cell_indices: NDArray[np.int32]


__all__ = [
    "CandidateGenerationArtifacts",
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
]
