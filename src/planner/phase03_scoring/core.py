from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PHASE_NAME = "scoring"
PHASE_ARTIFACT_STEM = "03_sparse_scores"


@dataclass(frozen=True, slots=True)
class SparseScoreArtifacts:
    configuration_indices: NDArray[np.int32]
    target_indices: NDArray[np.int32]
    dori_scores: NDArray[np.int8]


__all__ = [
    "PHASE_ARTIFACT_STEM",
    "PHASE_NAME",
    "SparseScoreArtifacts",
]
