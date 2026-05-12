from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OptimizationArtifacts:
    """Selected configurations plus final best-per-target DORI scores."""

    grid_shape: tuple[int, int]
    open_cell_count: int
    candidate_count: int
    configuration_count: int
    solved_k: int
    solver_name: str
    solver_status: str
    objective_value: float
    selected_configuration_ordinals: NDArray[np.int32]
    selected_candidate_ordinals: NDArray[np.int32]
    selected_angle_ordinals: NDArray[np.int16]
    selected_angles_deg: NDArray[np.float32]
    final_open_cell_scores: NDArray[np.int8]
    best_configuration_ordinals: NDArray[np.int32]

@dataclass(frozen=True, slots=True)
class OptimizationPrecomputeArtifacts:
    """Persisted reusable threshold-index artifacts for repeated phase-04 solves."""

    grid_shape: tuple[int, int]
    open_cell_count: int
    candidate_count: int
    configuration_count: int
    candidate_configuration_offsets: NDArray[np.int32]
    level_offsets: tuple[
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
    ]
    level_configuration_ordinals: tuple[
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
        NDArray[np.int32],
    ]
