from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CandidateGenerationArtifacts:
    """Deterministic phase-01 outputs derived from a tri-state floor-plan grid."""

    grid_shape: tuple[int, int]
    open_cell_indices: NDArray[np.int32]
    open_cell_coords_rc: NDArray[np.int32]
    eligible_candidate_cell_indices: NDArray[np.int32]
    eligible_candidate_cell_coords_rc: NDArray[np.int32]
    eligible_candidate_boundary_flags: NDArray[np.uint8]
    candidate_cell_indices: NDArray[np.int32]
    candidate_cell_coords_rc: NDArray[np.int32]
    candidate_boundary_flags: NDArray[np.uint8]
    candidate_exception_flags: NDArray[np.uint8]

@dataclass(frozen=True, slots=True)
class _DirectionalMasks:
    """Boolean directional masks reused across solid-adjacency wall-run logic."""

    open_mask: NDArray[np.bool_]
    north_solid: NDArray[np.bool_]
    east_solid: NDArray[np.bool_]
    south_solid: NDArray[np.bool_]
    west_solid: NDArray[np.bool_]
