from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VisibilityArtifacts:
    """Sparse candidate-major LOS relationships derived from the phase-01 ordinals."""

    grid_shape: tuple[int, int]
    candidate_count: int
    open_cell_count: int
    los_candidate_offsets: NDArray[np.int32]
    los_target_ordinals: NDArray[np.int32]
    diagonal_candidate_offsets: NDArray[np.int32]
    diagonal_target_ordinals: NDArray[np.int32]
