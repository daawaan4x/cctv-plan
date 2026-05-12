from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    """Summary metrics over open cells only for one optimization result."""

    total_dori_score: float
    detection_plus_pct: float
    observation_plus_pct: float
    recognition_plus_pct: float
    identification_pct: float
    blind_spot_pct: float

@dataclass(frozen=True, slots=True)
class VisualizationArtifacts:
    """Deterministic phase-05 arrays and metrics for one solved camera budget."""

    grid_shape: tuple[int, int]
    open_cell_count: int
    candidate_count: int
    configuration_count: int
    solved_k: int
    solver_name: str
    solver_status: str
    selected_camera_count: int
    metrics: CoverageMetrics
    final_open_cell_scores: NDArray[np.int8]
    final_score_grid: NDArray[np.int8]
    blind_spot_mask: NDArray[np.bool_]
    selected_configuration_ordinals: NDArray[np.int32]
    selected_candidate_ordinals: NDArray[np.int32]
    selected_candidate_coords_rc: NDArray[np.int32]
    selected_angle_ordinals: NDArray[np.int16]
    selected_angles_deg: NDArray[np.float32]
    best_configuration_ordinals: NDArray[np.int32]
