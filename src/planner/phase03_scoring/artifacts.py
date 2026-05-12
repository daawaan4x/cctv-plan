"""Phase-03 sparse score artifact containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SparseScoreArtifacts:
    """Sparse configuration-major DORI score arrays for valid non-zero pairs."""

    grid_shape: tuple[int, int]
    candidate_count: int
    open_cell_count: int
    orientation_angles_deg: NDArray[np.float32]
    configuration_candidate_ordinals: NDArray[np.int32]
    configuration_angle_ordinals: NDArray[np.int16]
    score_configuration_offsets: NDArray[np.int32]
    score_target_ordinals: NDArray[np.int32]
    score_values: NDArray[np.int8]


__all__ = ["SparseScoreArtifacts"]
