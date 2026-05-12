from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.common.floorplan import FloorPlanInput
from src.planner.phase01_candidate_generation import CandidateGenerationArtifacts
from src.planner.phase04_optimization import OptimizationArtifacts

from .artifacts import CoverageMetrics, VisualizationArtifacts
from .metrics import _metrics_equal, compute_coverage_metrics
from .score_validation import _validate_final_open_cell_scores
from .transforms import (
    _decode_selected_candidate_coords,
    _reconstruct_final_score_grid,
)


def validate_visualization_artifacts(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
    artifacts: VisualizationArtifacts,
) -> None:
    """Validate structural invariants and phase-05 reconstruction semantics."""

    _validate_phase_dependencies(floorplan, phase01_artifacts, phase04_artifacts)
    _validate_visualization_artifact_structure(artifacts)

    if artifacts.grid_shape != floorplan.shape:
        raise ValueError("Visualization grid_shape does not match floorplan.shape.")
    if artifacts.grid_shape != phase01_artifacts.grid_shape:
        raise ValueError("Visualization grid_shape does not match phase-01 grid_shape.")
    if artifacts.grid_shape != phase04_artifacts.grid_shape:
        raise ValueError("Visualization grid_shape does not match phase-04 grid_shape.")
    if artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Visualization open_cell_count does not match phase-01 open-cell count."
        )
    if artifacts.open_cell_count != phase04_artifacts.open_cell_count:
        raise ValueError(
            "Visualization open_cell_count does not match phase-04 open-cell count."
        )
    if artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Visualization candidate_count does not match phase-01 candidate count."
        )
    if artifacts.candidate_count != phase04_artifacts.candidate_count:
        raise ValueError(
            "Visualization candidate_count does not match phase-04 candidate count."
        )
    if artifacts.configuration_count != phase04_artifacts.configuration_count:
        raise ValueError(
            "Visualization configuration_count does not match phase-04."
        )
    if artifacts.solved_k != phase04_artifacts.solved_k:
        raise ValueError("Visualization solved_k does not match phase-04.")
    if artifacts.solver_name != phase04_artifacts.solver_name:
        raise ValueError("Visualization solver_name does not match phase-04.")
    if artifacts.solver_status != phase04_artifacts.solver_status:
        raise ValueError("Visualization solver_status does not match phase-04.")

    expected_selected_camera_count = len(artifacts.selected_configuration_ordinals)
    if artifacts.selected_camera_count != expected_selected_camera_count:
        raise ValueError(
            "selected_camera_count must equal the selected configuration count."
        )
    if artifacts.selected_camera_count != len(artifacts.selected_candidate_ordinals):
        raise ValueError(
            "selected_camera_count must equal the selected candidate count."
        )
    if artifacts.selected_camera_count != len(artifacts.selected_angle_ordinals):
        raise ValueError(
            "selected_camera_count must equal the selected angle-ordinal count."
        )
    if artifacts.selected_camera_count != len(artifacts.selected_angles_deg):
        raise ValueError("selected_camera_count must equal the selected angle count.")
    if artifacts.selected_candidate_coords_rc.shape[0] != artifacts.selected_camera_count:
        raise ValueError(
            "selected_candidate_coords_rc length must equal selected_camera_count."
        )

    if not np.array_equal(
        artifacts.selected_configuration_ordinals,
        phase04_artifacts.selected_configuration_ordinals,
    ):
        raise ValueError(
            "selected_configuration_ordinals does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.selected_candidate_ordinals,
        phase04_artifacts.selected_candidate_ordinals,
    ):
        raise ValueError(
            "selected_candidate_ordinals does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.selected_angle_ordinals,
        phase04_artifacts.selected_angle_ordinals,
    ):
        raise ValueError(
            "selected_angle_ordinals does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.selected_angles_deg,
        phase04_artifacts.selected_angles_deg,
    ):
        raise ValueError(
            "selected_angles_deg does not match phase-04 selection output."
        )
    if not np.array_equal(
        artifacts.final_open_cell_scores,
        phase04_artifacts.final_open_cell_scores,
    ):
        raise ValueError(
            "final_open_cell_scores does not match the phase-04 final score vector."
        )
    if not np.array_equal(
        artifacts.best_configuration_ordinals,
        phase04_artifacts.best_configuration_ordinals,
    ):
        raise ValueError(
            "best_configuration_ordinals does not match the phase-04 best-selection map."
        )

    expected_score_grid = _reconstruct_final_score_grid(
        floorplan.shape,
        phase01_artifacts.open_cell_coords_rc,
        phase04_artifacts.final_open_cell_scores,
    )
    if not np.array_equal(artifacts.final_score_grid, expected_score_grid):
        raise ValueError(
            "final_score_grid does not match the expected open-cell scatter result."
        )
    if np.any(artifacts.final_score_grid[~floorplan.open_mask] != -1):
        raise ValueError(
            "Non-open cells in final_score_grid must keep the sentinel value -1."
        )

    expected_blind_spot_mask = floorplan.open_mask & (artifacts.final_score_grid == 0)
    if not np.array_equal(artifacts.blind_spot_mask, expected_blind_spot_mask):
        raise ValueError(
            "blind_spot_mask must equal the open-cell score-zero mask exactly."
        )
    if np.any(artifacts.blind_spot_mask[floorplan.null_mask | floorplan.solid_mask]):
        raise ValueError("Null or solid cells must never appear in blind_spot_mask.")

    expected_selected_coords = _decode_selected_candidate_coords(
        phase01_artifacts.candidate_cell_coords_rc,
        phase04_artifacts.selected_candidate_ordinals,
    )
    if not np.array_equal(artifacts.selected_candidate_coords_rc, expected_selected_coords):
        raise ValueError(
            "selected_candidate_coords_rc does not decode from the selected "
            "candidate ordinals."
        )

    expected_metrics = compute_coverage_metrics(artifacts.final_open_cell_scores)
    if not _metrics_equal(artifacts.metrics, expected_metrics):
        raise ValueError("Visualization metrics do not match the final score vector.")
    if not (
        artifacts.metrics.identification_pct
        <= artifacts.metrics.recognition_plus_pct
        <= artifacts.metrics.observation_plus_pct
        <= artifacts.metrics.detection_plus_pct
    ):
        raise ValueError("Coverage percentage metrics must be monotone by DORI level.")
    if not np.isclose(
        artifacts.metrics.blind_spot_pct + artifacts.metrics.detection_plus_pct,
        100.0,
    ):
        raise ValueError(
            "blind_spot_pct and detection_plus_pct must sum to 100 over open cells."
        )

def _validate_phase_dependencies(
    floorplan: FloorPlanInput,
    phase01_artifacts: CandidateGenerationArtifacts,
    phase04_artifacts: OptimizationArtifacts,
) -> None:
    """Confirm that phase 01 and phase 04 still match the current floor plan."""

    if phase01_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-01 artifacts do not match the current floorplan shape.")
    if phase04_artifacts.grid_shape != floorplan.shape:
        raise ValueError("Phase-04 artifacts do not match the current floorplan shape.")
    if phase04_artifacts.open_cell_count != len(phase01_artifacts.open_cell_indices):
        raise ValueError(
            "Phase-04 open_cell_count does not match the phase-01 open-cell count."
        )
    if phase04_artifacts.candidate_count != len(phase01_artifacts.candidate_cell_indices):
        raise ValueError(
            "Phase-04 candidate_count does not match the phase-01 candidate count."
        )

def _validate_visualization_artifact_structure(
    artifacts: VisualizationArtifacts,
) -> None:
    """Validate the standalone structure, dtypes, and bounds of phase-05 artifacts."""

    if len(artifacts.grid_shape) != 2:
        raise ValueError("grid_shape must contain exactly two integers.")
    if artifacts.open_cell_count < 0:
        raise ValueError("open_cell_count must be non-negative.")
    if artifacts.candidate_count < 0:
        raise ValueError("candidate_count must be non-negative.")
    if artifacts.configuration_count < 0:
        raise ValueError("configuration_count must be non-negative.")
    if artifacts.solved_k <= 0:
        raise ValueError("solved_k must be positive.")
    if artifacts.selected_camera_count < 0:
        raise ValueError("selected_camera_count must be non-negative.")

    _validate_final_open_cell_scores(
        artifacts.final_open_cell_scores,
        open_cell_count=artifacts.open_cell_count,
    )
    _validate_selected_configuration_ordinals(
        artifacts.selected_configuration_ordinals,
        configuration_count=artifacts.configuration_count,
        solved_k=artifacts.solved_k,
    )
    _validate_selected_candidate_ordinals(
        artifacts.selected_candidate_ordinals,
        candidate_count=artifacts.candidate_count,
        expected_length=artifacts.selected_camera_count,
    )
    _validate_selected_candidate_coords(
        artifacts.selected_candidate_coords_rc,
        expected_length=artifacts.selected_camera_count,
        grid_shape=artifacts.grid_shape,
    )
    _validate_selected_angle_ordinals(
        artifacts.selected_angle_ordinals,
        expected_length=artifacts.selected_camera_count,
    )
    _validate_selected_angles_deg(
        artifacts.selected_angles_deg,
        expected_length=artifacts.selected_camera_count,
    )
    _validate_best_configuration_ordinals(
        artifacts.best_configuration_ordinals,
        open_cell_count=artifacts.open_cell_count,
        selected_configuration_ordinals=artifacts.selected_configuration_ordinals,
    )
    _validate_final_score_grid(
        artifacts.final_score_grid,
        grid_shape=artifacts.grid_shape,
    )
    _validate_blind_spot_mask(
        artifacts.blind_spot_mask,
        grid_shape=artifacts.grid_shape,
    )

def _validate_selected_configuration_ordinals(
    selected_configuration_ordinals: NDArray[np.int32],
    *,
    configuration_count: int,
    solved_k: int,
) -> None:
    """Validate dtype, ordering, uniqueness, bounds, and budget for config ordinals."""

    if selected_configuration_ordinals.dtype != np.int32:
        raise TypeError("selected_configuration_ordinals must use dtype np.int32.")
    if selected_configuration_ordinals.ndim != 1:
        raise ValueError("selected_configuration_ordinals must be a 1D array.")
    if len(selected_configuration_ordinals) > solved_k:
        raise ValueError(
            "The selected configuration count must not exceed the solved camera budget."
        )
    if len(selected_configuration_ordinals) > 1 and not np.all(
        selected_configuration_ordinals[:-1] < selected_configuration_ordinals[1:]
    ):
        raise ValueError(
            "selected_configuration_ordinals must be strictly increasing and unique."
        )
    if selected_configuration_ordinals.size and (
        selected_configuration_ordinals[0] < 0
        or selected_configuration_ordinals[-1] >= configuration_count
    ):
        raise ValueError(
            "selected_configuration_ordinals contains an out-of-range configuration "
            "ordinal."
        )

def _validate_selected_candidate_ordinals(
    selected_candidate_ordinals: NDArray[np.int32],
    *,
    candidate_count: int,
    expected_length: int,
) -> None:
    """Validate dtype, shape, length, bounds, and no-duplicate candidate selection."""

    if selected_candidate_ordinals.dtype != np.int32:
        raise TypeError("selected_candidate_ordinals must use dtype np.int32.")
    if selected_candidate_ordinals.ndim != 1:
        raise ValueError("selected_candidate_ordinals must be a 1D array.")
    if len(selected_candidate_ordinals) != expected_length:
        raise ValueError(
            "selected_candidate_ordinals length must equal selected_camera_count."
        )
    if selected_candidate_ordinals.size and (
        np.min(selected_candidate_ordinals) < 0
        or np.max(selected_candidate_ordinals) >= candidate_count
    ):
        raise ValueError(
            "selected_candidate_ordinals contains an out-of-range candidate ordinal."
        )
    if len(np.unique(selected_candidate_ordinals)) != len(selected_candidate_ordinals):
        raise ValueError("No candidate ordinal may appear more than once.")

def _validate_selected_candidate_coords(
    selected_candidate_coords_rc: NDArray[np.int32],
    *,
    expected_length: int,
    grid_shape: tuple[int, int],
) -> None:
    """Validate dtype, shape, length, and in-grid bounds for selected coordinates."""

    if selected_candidate_coords_rc.dtype != np.int32:
        raise TypeError("selected_candidate_coords_rc must use dtype np.int32.")
    if selected_candidate_coords_rc.ndim != 2 or selected_candidate_coords_rc.shape[1] != 2:
        raise ValueError("selected_candidate_coords_rc must have shape (N, 2).")
    if selected_candidate_coords_rc.shape[0] != expected_length:
        raise ValueError(
            "selected_candidate_coords_rc length must equal selected_camera_count."
        )
    if selected_candidate_coords_rc.size == 0:
        return

    rows = selected_candidate_coords_rc[:, 0]
    cols = selected_candidate_coords_rc[:, 1]
    if (
        np.min(rows) < 0
        or np.max(rows) >= grid_shape[0]
        or np.min(cols) < 0
        or np.max(cols) >= grid_shape[1]
    ):
        raise ValueError("selected_candidate_coords_rc contains an out-of-grid cell.")

def _validate_selected_angle_ordinals(
    selected_angle_ordinals: NDArray[np.int16],
    *,
    expected_length: int,
) -> None:
    """Validate dtype, shape, and length for selected angle ordinals."""

    if selected_angle_ordinals.dtype != np.int16:
        raise TypeError("selected_angle_ordinals must use dtype np.int16.")
    if selected_angle_ordinals.ndim != 1:
        raise ValueError("selected_angle_ordinals must be a 1D array.")
    if len(selected_angle_ordinals) != expected_length:
        raise ValueError(
            "selected_angle_ordinals length must equal selected_camera_count."
        )

def _validate_selected_angles_deg(
    selected_angles_deg: NDArray[np.float32],
    *,
    expected_length: int,
) -> None:
    """Validate dtype, shape, and length for selected physical angles."""

    if selected_angles_deg.dtype != np.float32:
        raise TypeError("selected_angles_deg must use dtype np.float32.")
    if selected_angles_deg.ndim != 1:
        raise ValueError("selected_angles_deg must be a 1D array.")
    if len(selected_angles_deg) != expected_length:
        raise ValueError("selected_angles_deg length must equal selected_camera_count.")

def _validate_best_configuration_ordinals(
    best_configuration_ordinals: NDArray[np.int32],
    *,
    open_cell_count: int,
    selected_configuration_ordinals: NDArray[np.int32],
) -> None:
    """Validate target-to-best-configuration bookkeeping against selection bounds."""

    if best_configuration_ordinals.dtype != np.int32:
        raise TypeError("best_configuration_ordinals must use dtype np.int32.")
    if best_configuration_ordinals.ndim != 1:
        raise ValueError("best_configuration_ordinals must be a 1D array.")
    if len(best_configuration_ordinals) != open_cell_count:
        raise ValueError("best_configuration_ordinals length must equal open_cell_count.")

    selected_lookup = set(selected_configuration_ordinals.tolist())
    for configuration_ordinal in best_configuration_ordinals:
        if int(configuration_ordinal) == -1:
            continue
        if int(configuration_ordinal) not in selected_lookup:
            raise ValueError(
                "best_configuration_ordinals must point only to selected "
                "configuration ordinals or -1."
            )

def _validate_final_score_grid(
    final_score_grid: NDArray[np.int8],
    *,
    grid_shape: tuple[int, int],
) -> None:
    """Validate dtype and shape for the reconstructed full-grid score surface."""

    if final_score_grid.dtype != np.int8:
        raise TypeError("final_score_grid must use dtype np.int8.")
    if final_score_grid.shape != grid_shape:
        raise ValueError("final_score_grid shape must equal grid_shape.")

def _validate_blind_spot_mask(
    blind_spot_mask: NDArray[np.bool_],
    *,
    grid_shape: tuple[int, int],
) -> None:
    """Validate dtype and shape for the explicit blind-spot mask."""

    if blind_spot_mask.dtype != np.bool_:
        raise TypeError("blind_spot_mask must use dtype np.bool_.")
    if blind_spot_mask.shape != grid_shape:
        raise ValueError("blind_spot_mask shape must equal grid_shape.")
